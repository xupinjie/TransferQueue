# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NIXL storage manager (Phase 1: persistent staging arena).

Reuses everything from ``AsyncSimpleStorageManager`` (hash routing, ``put_data`` /
``get_data`` orchestration, controller handshake, ``notify_data_update``,
checkpoint). The only override is the per-storage-unit data transfer: the manager
acts as a NIXL **initiator** and READs/WRITEs a serialized payload between its own
register-once ``NixlArena`` and the unit's arena. Signalling stays on ZMQ; NIXL
agent metadata is exchanged once per (manager, unit) pair over the same channel.

Robustness:
- Unit arena full -> unit replies ``status="BUSY"``; the manager retries a few
  times with backoff, then falls back to the plain ZMQ data plane.
- Payload larger than the arena -> ``status="FALLBACK"`` (GET) or detected upfront
  (PUT); the manager uses the ZMQ data plane for that single transfer.
- NIXL disabled / init failed -> everything uses the parent ZMQ path.
"""

import asyncio
import os
import pickle
from typing import Any, Callable
from uuid import uuid4

import zmq
from omegaconf import DictConfig

from transfer_queue.storage.managers.base import StorageManagerFactory
from transfer_queue.storage.managers.simple_storage_manager import (
    AsyncSimpleStorageManager,
    with_storage_unit_socket,
)
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.nixl_utils import NixlArena
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo

logger = get_logger(__name__)

DEFAULT_NIXL_ARENA_MB = 512
# How many times to retry a prepare that returned BUSY (unit arena full) before
# falling back to the ZMQ data plane, and the backoff between retries (seconds).
TQ_NIXL_BUSY_MAX_RETRIES = int(os.environ.get("TQ_NIXL_BUSY_MAX_RETRIES", 5))
TQ_NIXL_BUSY_BACKOFF = float(os.environ.get("TQ_NIXL_BUSY_BACKOFF", 0.01))
# How many times to retry local arena allocation before falling back.
TQ_NIXL_LOCAL_ALLOC_RETRIES = int(os.environ.get("TQ_NIXL_LOCAL_ALLOC_RETRIES", 100))
TQ_NIXL_LOCAL_ALLOC_BACKOFF = float(os.environ.get("TQ_NIXL_LOCAL_ALLOC_BACKOFF", 0.005))


class _NixlFallback(Exception):
    """Internal signal: this transfer cannot use NIXL; use the ZMQ data plane instead."""


@StorageManagerFactory.register("NixlStorage")
class NixlStorageManager(AsyncSimpleStorageManager):
    """Storage manager that moves bulk data over NIXL while reusing SimpleStorage control."""

    def __init__(self, controller_info: ZMQServerInfo, config: DictConfig):
        super().__init__(controller_info, config)

        self._nixl_enabled = False
        # Set only when NIXL initializes; guarded at every use by ``_nixl_enabled``.
        # Typed non-Optional to keep call sites clean for mypy.
        self._nixl_agent: Any = None
        self._arena: NixlArena = None  # type: ignore[assignment]
        self._nixl_agent_name = f"tq-nixl-mgr-{self.storage_manager_id}"
        # storage_unit_id -> remote NIXL agent name, populated during handshake.
        self._nixl_su_agent_names: dict[str, str] = {}
        self._nixl_ready: set[str] = set()
        self._nixl_handshake_lock = asyncio.Lock()

        use_nixl = bool(config.get("use_nixl", True))
        nixl_backends = list(config.get("nixl_backends", ["UCX"]))
        nixl_arena_mb = int(config.get("nixl_arena_mb", DEFAULT_NIXL_ARENA_MB))

        if use_nixl:
            try:
                from nixl import nixl_agent, nixl_agent_config

                cfg = nixl_agent_config(
                    enable_prog_thread=True,
                    enable_listen_thread=False,
                    backends=nixl_backends,
                )
                self._nixl_agent = nixl_agent(self._nixl_agent_name, cfg)
                self._arena = NixlArena(self._nixl_agent, nixl_arena_mb * 1024 * 1024)
                self._nixl_enabled = True
                logger.info(
                    f"[{self.storage_manager_id}]: NIXL agent '{self._nixl_agent_name}' + "
                    f"{nixl_arena_mb}MB arena initialized (backends={nixl_backends})."
                )
            except Exception as e:
                logger.warning(
                    f"[{self.storage_manager_id}]: failed to initialize NIXL "
                    f"({type(e).__name__}: {e}); falling back to SimpleStorage ZMQ data plane."
                )

    # ---- NIXL metadata handshake (once per storage unit) ----------------------------

    async def _ensure_nixl_handshake(self, target_storage_unit: str) -> None:
        if target_storage_unit in self._nixl_ready:
            return
        async with self._nixl_handshake_lock:
            if target_storage_unit in self._nixl_ready:
                return
            response = await self._nixl_handshake_with_su(target_storage_unit=target_storage_unit)
            if response.request_type != ZMQRequestType.NIXL_HANDSHAKE_RESPONSE:
                raise RuntimeError(
                    f"NIXL handshake with {target_storage_unit} failed: {response.body.get('message', 'unknown error')}"
                )
            self._nixl_agent.add_remote_agent(response.body["agent_metadata"])
            self._nixl_su_agent_names[target_storage_unit] = response.body["agent_name"]
            self._nixl_ready.add(target_storage_unit)
            logger.info(
                f"[{self.storage_manager_id}]: NIXL handshake with {target_storage_unit} "
                f"(remote agent '{response.body['agent_name']}') complete."
            )

    @with_storage_unit_socket
    async def _nixl_handshake_with_su(self, target_storage_unit: str, socket: zmq.Socket = None) -> ZMQMessage:
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.NIXL_HANDSHAKE,  # type: ignore[arg-type]
            sender_id=self.storage_manager_id,
            receiver_id=target_storage_unit,
            body={
                "agent_name": self._nixl_agent_name,
                "agent_metadata": self._nixl_agent.get_agent_metadata(),
            },
        )
        await socket.send_multipart(request_msg.serialize())
        return ZMQMessage.deserialize(await socket.recv_multipart(copy=False))

    # ---- NIXL transfer primitive (arena sub-range <-> remote descs) -----------------

    async def _nixl_transfer(
        self, operation: str, local_offset: int, nbytes: int, serialized_remote_descs: bytes, remote_agent: str
    ) -> None:
        """One-sided NIXL READ/WRITE between a local arena range and remote descriptors."""
        local_descs = self._arena.xfer_descs(local_offset, nbytes)
        remote_descs = self._nixl_agent.deserialize_descs(serialized_remote_descs)
        handle = self._nixl_agent.initialize_xfer(operation, local_descs, remote_descs, remote_agent)
        try:
            state = self._nixl_agent.transfer(handle)
            if state == "ERR":
                raise RuntimeError(f"NIXL {operation} failed to post.")
            while state != "DONE":
                state = self._nixl_agent.check_xfer_state(handle)
                if state == "ERR":
                    raise RuntimeError(f"NIXL {operation} entered error state.")
                if state != "DONE":
                    await asyncio.sleep(0)
        finally:
            self._nixl_agent.release_xfer_handle(handle)

    async def _alloc_local(self, nbytes: int) -> int:
        """Allocate a local arena range, waiting briefly if the arena is momentarily full."""
        for _ in range(TQ_NIXL_LOCAL_ALLOC_RETRIES):
            offset = self._arena.allocate(nbytes)
            if offset is not None:
                return offset
            await asyncio.sleep(TQ_NIXL_LOCAL_ALLOC_BACKOFF)
        raise _NixlFallback()

    # ---- GET (manager READs from unit) ----------------------------------------------

    async def _get_from_single_storage_unit(
        self,
        global_indexes: list[int],
        fields: list[str],
        target_storage_unit: str,
        socket: zmq.Socket = None,
    ):
        if not self._nixl_enabled:
            return await super()._get_from_single_storage_unit(
                global_indexes, fields, target_storage_unit=target_storage_unit
            )
        await self._ensure_nixl_handshake(target_storage_unit)
        try:
            return await self._get_nixl(global_indexes, fields, target_storage_unit=target_storage_unit)
        except _NixlFallback:
            logger.debug(f"[{self.storage_manager_id}]: NIXL get falling back to ZMQ for {target_storage_unit}.")
            return await super()._get_from_single_storage_unit(
                global_indexes, fields, target_storage_unit=target_storage_unit
            )

    @with_storage_unit_socket
    async def _get_nixl(
        self,
        global_indexes: list[int],
        fields: list[str],
        target_storage_unit: str,
        socket: zmq.Socket = None,
    ):
        # 1. Ask the unit to stage the requested data; retry on BUSY, fall back on FALLBACK.
        response_msg = await self._prepare_with_retry(
            socket,
            lambda request_id: ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_NIXL,  # type: ignore[arg-type]
                sender_id=self.storage_manager_id,
                receiver_id=target_storage_unit,
                body={"global_indexes": global_indexes, "fields": fields, "request_id": request_id},
            ),
            ZMQRequestType.GET_DATA_NIXL_RESPONSE,
            target_storage_unit,
        )
        request_id = response_msg.body["request_id"]
        payload_len = response_msg.body["payload_len"]
        serialized_descs = response_msg.body["serialized_descs"]

        # The unit has now allocated a range; it MUST be released regardless of what
        # happens next (local-alloc fallback, transfer error, ...), so wrap in finally.
        local_offset = None
        try:
            # 2. READ the staged payload into a local arena range.
            local_offset = await self._alloc_local(payload_len)
            await self._nixl_transfer(
                "READ", local_offset, payload_len, serialized_descs, self._nixl_su_agent_names[target_storage_unit]
            )
            storage_unit_data: dict[str, Any] = pickle.loads(self._arena.read_bytes(local_offset, payload_len))
        finally:
            if local_offset is not None:
                self._arena.free(local_offset)
            # 3. Always release the unit's arena range.
            release_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_NIXL_RELEASE,  # type: ignore[arg-type]
                sender_id=self.storage_manager_id,
                receiver_id=target_storage_unit,
                body={"request_id": request_id},
            )
            await socket.send_multipart(release_msg.serialize())
            await socket.recv_multipart(copy=False)

        # 4. Same shape SimpleStorage returns: (fields, {field: [items]}).
        return fields, storage_unit_data

    # ---- PUT (manager WRITEs to unit) -----------------------------------------------

    async def _put_to_single_storage_unit(
        self,
        global_indexes: list[int],
        storage_data: dict[str, Any],
        target_storage_unit: str,
        data_parser: Callable[[Any], Any] | None = None,
        socket: zmq.Socket = None,
    ):
        # data_parser runs distributedly on the unit in SimpleStorage; NIXL does not carry
        # it, so use the ZMQ path when a parser is requested.
        if not self._nixl_enabled or data_parser is not None:
            return await super()._put_to_single_storage_unit(
                global_indexes, storage_data, target_storage_unit=target_storage_unit, data_parser=data_parser
            )

        payload = pickle.dumps(storage_data, protocol=pickle.HIGHEST_PROTOCOL)
        # Payload larger than the whole arena can never be staged: use the ZMQ path.
        if len(payload) > self._arena.size:
            return await super()._put_to_single_storage_unit(
                global_indexes, storage_data, target_storage_unit=target_storage_unit, data_parser=None
            )

        await self._ensure_nixl_handshake(target_storage_unit)
        try:
            return await self._put_nixl(global_indexes, payload, target_storage_unit=target_storage_unit)
        except _NixlFallback:
            logger.debug(f"[{self.storage_manager_id}]: NIXL put falling back to ZMQ for {target_storage_unit}.")
            return await super()._put_to_single_storage_unit(
                global_indexes, storage_data, target_storage_unit=target_storage_unit, data_parser=None
            )

    @with_storage_unit_socket
    async def _put_nixl(
        self,
        global_indexes: list[int],
        payload: bytes,
        target_storage_unit: str,
        socket: zmq.Socket = None,
    ):
        payload_len = len(payload)
        # 1. Reserve the local range and fill it BEFORE the unit allocates anything, so a
        #    local-alloc fallback leaves no unit-side state to clean up.
        local_offset = await self._alloc_local(payload_len)
        try:
            self._arena.write_bytes(local_offset, payload)

            # 2. Ask the unit to allocate a receive range; retry on BUSY, fall back on FALLBACK.
            #    On BUSY/FALLBACK the unit did not allocate, so no commit is owed.
            response_msg = await self._prepare_with_retry(
                socket,
                lambda request_id: ZMQMessage.create(
                    request_type=ZMQRequestType.PUT_DATA_NIXL,  # type: ignore[arg-type]
                    sender_id=self.storage_manager_id,
                    receiver_id=target_storage_unit,
                    body={"global_indexes": global_indexes, "payload_len": payload_len, "request_id": request_id},
                ),
                ZMQRequestType.PUT_DATA_NIXL_RESPONSE,
                target_storage_unit,
            )
            request_id = response_msg.body["request_id"]
            serialized_descs = response_msg.body["serialized_descs"]

            # 3. The unit now holds a range; it MUST be committed (which also frees it on the
            #    unit side) regardless of transfer outcome, so send commit in finally.
            try:
                await self._nixl_transfer(
                    "WRITE",
                    local_offset,
                    payload_len,
                    serialized_descs,
                    self._nixl_su_agent_names[target_storage_unit],
                )
            finally:
                commit_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.PUT_DATA_NIXL_COMMIT,  # type: ignore[arg-type]
                    sender_id=self.storage_manager_id,
                    receiver_id=target_storage_unit,
                    body={"request_id": request_id},
                )
                await socket.send_multipart(commit_msg.serialize())
                commit_resp = ZMQMessage.deserialize(await socket.recv_multipart(copy=False))

            if commit_resp.request_type != ZMQRequestType.PUT_DATA_NIXL_COMMIT_RESPONSE:
                raise RuntimeError(
                    f"NIXL put commit to {target_storage_unit} failed: "
                    f"{commit_resp.body.get('message', 'unknown error')}"
                )
        finally:
            self._arena.free(local_offset)

    # ---- shared prepare/retry logic -------------------------------------------------

    async def _prepare_with_retry(
        self, socket, build_request, expected_response_type, target_storage_unit: str
    ) -> ZMQMessage:
        """Send a prepare request (fresh request_id each attempt); handle OK/BUSY/FALLBACK.

        Returns the OK response with ``request_id`` injected into its body. Raises
        ``_NixlFallback`` on FALLBACK or after BUSY retries are exhausted.
        """
        for attempt in range(TQ_NIXL_BUSY_MAX_RETRIES + 1):
            request_id = uuid4().hex
            await socket.send_multipart(build_request(request_id).serialize())
            response_msg = ZMQMessage.deserialize(await socket.recv_multipart(copy=False))

            if response_msg.request_type != expected_response_type:
                raise RuntimeError(
                    f"NIXL prepare to {target_storage_unit} failed: {response_msg.body.get('message', 'unknown error')}"
                )

            status = response_msg.body.get("status")
            if status == "OK":
                response_msg.body["request_id"] = request_id
                return response_msg
            if status == "FALLBACK":
                raise _NixlFallback()
            if status == "BUSY":
                if attempt < TQ_NIXL_BUSY_MAX_RETRIES:
                    await asyncio.sleep(TQ_NIXL_BUSY_BACKOFF)
                    continue
                raise _NixlFallback()
            raise RuntimeError(f"NIXL prepare to {target_storage_unit} returned unexpected status: {status}")

        # Unreachable: the loop always returns (OK) or raises; kept for mypy's control flow.
        raise _NixlFallback()
