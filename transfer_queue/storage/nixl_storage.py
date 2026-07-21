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

"""NIXL-backed storage unit (Phase 1: persistent staging arena).

Storage and control logic are inherited unchanged from ``SimpleStorageUnitBase``:
data still lives in the in-memory ``StorageUnitData`` dict, and the ZMQ server /
worker loop / handshake / checkpoint paths are reused. The addition is a data
plane that moves bulk bytes over NIXL (UCX/RDMA).

Design "X" — landing-zone arena:
- The unit owns ONE large CPU buffer (``NixlArena``) registered with NIXL **once**
  at construction. Because it is registered before any peer exchanges metadata,
  the manager's metadata snapshot already contains it, and every transfer targets
  a sub-range of this one region — no per-request registration.
- The unit is the passive **target**; the manager (initiator) drives every
  transfer. GET: the unit serializes the requested data into an allocated arena
  range and the manager READs it. PUT: the unit allocates a receive range and the
  manager WRITEs into it, then a ZMQ commit makes the unit copy it out and
  deserialize into the dict. All signalling stays on ZMQ.
- Many managers can target different arena offsets concurrently without conflict;
  only allocation is serialized (in the single-threaded worker). When the arena
  is full a prepare returns ``status="BUSY"`` (manager retries); when a payload is
  larger than the whole arena it returns ``status="FALLBACK"`` (manager uses the
  plain ZMQ data plane for that transfer).

Payloads are ``pickle``-serialized (dict storage unchanged); tensor-native
zero-copy is a later phase.
"""

import pickle
from typing import Any

import ray

from transfer_queue.storage.simple_storage import SimpleStorageUnitBase
from transfer_queue.utils.common import limit_pytorch_auto_parallel_threads
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.nixl_utils import NixlArena
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

logger = get_logger(__name__)

TQ_NUM_THREADS_DEFAULT = 8
DEFAULT_NIXL_ARENA_MB = 512


@ray.remote(num_cpus=1)
class NixlStorageUnit(SimpleStorageUnitBase):
    """A ``SimpleStorageUnitBase`` whose bulk data plane is served by NIXL.

    Args:
        storage_unit_size: Same meaning as ``SimpleStorageUnit``.
        use_nixl: When False (or when NIXL init fails), the unit degrades to a
            plain storage unit and NIXL operations are rejected; the paired
            ``NixlStorageManager`` then uses the ZMQ data plane.
        nixl_backends: NIXL backend plugins to initialize (default ``["UCX"]``).
        nixl_arena_mb: Size of the register-once staging arena, in MB.
    """

    def __init__(
        self,
        storage_unit_size: int | None = None,
        use_nixl: bool = True,
        nixl_backends: tuple[str, ...] | list[str] = ("UCX",),
        nixl_arena_mb: int = DEFAULT_NIXL_ARENA_MB,
    ):
        super().__init__(storage_unit_size=storage_unit_size)

        # request_id -> (offset, nbytes[, global_indexes]) for an in-flight transfer.
        self._nixl_pending: dict[str, tuple] = {}
        self._nixl_enabled = False
        # Set only when NIXL initializes; guarded at every use by ``_nixl_enabled`` /
        # ``_require_nixl``. Typed non-Optional to keep call sites clean for mypy.
        self._nixl_agent: Any = None
        self._arena: NixlArena = None  # type: ignore[assignment]
        self._nixl_agent_name = f"tq-nixl-su-{self.storage_unit_id}"

        if use_nixl:
            try:
                from nixl import nixl_agent, nixl_agent_config

                cfg = nixl_agent_config(
                    enable_prog_thread=True,
                    enable_listen_thread=False,  # metadata is exchanged over ZMQ, not NIXL's listener
                    backends=list(nixl_backends),
                )
                self._nixl_agent = nixl_agent(self._nixl_agent_name, cfg)
                # Register the arena now, before any peer fetches this agent's metadata.
                self._arena = NixlArena(self._nixl_agent, nixl_arena_mb * 1024 * 1024)
                self._nixl_enabled = True
                logger.info(
                    f"[{self.storage_unit_id}]: NIXL agent '{self._nixl_agent_name}' + "
                    f"{nixl_arena_mb}MB arena initialized (backends={list(nixl_backends)})."
                )
            except Exception as e:
                logger.warning(
                    f"[{self.storage_unit_id}]: failed to initialize NIXL "
                    f"({type(e).__name__}: {e}); falling back to SimpleStorage ZMQ data plane."
                )

    # ---- worker-loop extension hook -------------------------------------------------

    def _handle_extended_operation(self, operation, request_msg: ZMQMessage) -> ZMQMessage | None:
        """Dispatch the NIXL operations added on top of SimpleStorageUnit."""
        if operation == ZMQRequestType.NIXL_HANDSHAKE:  # type: ignore[arg-type]
            return self._handle_nixl_handshake(request_msg)
        elif operation == ZMQRequestType.GET_DATA_NIXL:  # type: ignore[arg-type]
            return self._handle_get_nixl_prepare(request_msg)
        elif operation == ZMQRequestType.GET_DATA_NIXL_RELEASE:  # type: ignore[arg-type]
            return self._handle_get_nixl_release(request_msg)
        elif operation == ZMQRequestType.PUT_DATA_NIXL:  # type: ignore[arg-type]
            return self._handle_put_nixl_prepare(request_msg)
        elif operation == ZMQRequestType.PUT_DATA_NIXL_COMMIT:  # type: ignore[arg-type]
            return self._handle_put_nixl_commit(request_msg)
        return None

    def _require_nixl(self) -> None:
        if not self._nixl_enabled or self._nixl_agent is None or self._arena is None:
            raise RuntimeError(
                f"[{self.storage_unit_id}]: received a NIXL operation but NIXL is not enabled on this unit."
            )

    # ---- metadata handshake ---------------------------------------------------------

    def _handle_nixl_handshake(self, request_msg: ZMQMessage) -> ZMQMessage:
        """Register the manager's NIXL metadata and return this unit's metadata."""
        try:
            self._require_nixl()
            self._nixl_agent.add_remote_agent(request_msg.body["agent_metadata"])
            return ZMQMessage.create(
                request_type=ZMQRequestType.NIXL_HANDSHAKE_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "agent_name": self._nixl_agent_name,
                    "agent_metadata": self._nixl_agent.get_agent_metadata(),
                },
            )
        except Exception as e:
            logger.error(f"[{self.storage_unit_id}]: NIXL handshake failed: {type(e).__name__}: {e}")
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_GET_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"message": f"NIXL handshake failed on {self.storage_unit_id}: {e}"},
            )

    # ---- GET path (unit = data source; manager READs) -------------------------------

    def _handle_get_nixl_prepare(self, request_msg: ZMQMessage) -> ZMQMessage:
        """Serialize requested data into an arena range; return its descriptors or BUSY/FALLBACK."""
        request_id = request_msg.body["request_id"]
        try:
            self._require_nixl()
            fields = request_msg.body["fields"]
            global_indexes = request_msg.body["global_indexes"]

            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS_DEFAULT, info=f"[{self.storage_unit_id}] _handle_get_nixl"
            ):
                result_data = self.storage_data.get_data(fields, global_indexes)

            payload = pickle.dumps(result_data, protocol=pickle.HIGHEST_PROTOCOL)
            payload_len = len(payload)

            if payload_len > self._arena.size:
                return self._nixl_status_response(ZMQRequestType.GET_DATA_NIXL_RESPONSE, "FALLBACK")

            offset = self._arena.allocate(payload_len)
            if offset is None:
                return self._nixl_status_response(ZMQRequestType.GET_DATA_NIXL_RESPONSE, "BUSY")

            self._arena.write_bytes(offset, payload)
            self._nixl_pending[request_id] = (offset, payload_len)
            return ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_NIXL_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "status": "OK",
                    "serialized_descs": self._arena.serialized_descs(offset, payload_len),
                    "payload_len": payload_len,
                },
            )
        except Exception as e:
            logger.error(f"[{self.storage_unit_id}]: NIXL get prepare failed: {type(e).__name__}: {e}")
            return ZMQMessage.create(
                request_type=ZMQRequestType.GET_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"message": f"Failed to prepare NIXL get on {self.storage_unit_id}: {e}"},
            )

    def _handle_get_nixl_release(self, request_msg: ZMQMessage) -> ZMQMessage:
        """Free a GET arena range after the manager finished reading it."""
        request_id = request_msg.body["request_id"]
        pending = self._nixl_pending.pop(request_id, None)
        if pending is not None:
            self._arena.free(pending[0])
        return ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA_NIXL_RELEASE_RESPONSE,  # type: ignore[arg-type]
            sender_id=self.storage_unit_id,
            body={},
        )

    # ---- PUT path (unit = data sink; manager WRITEs) --------------------------------

    def _handle_put_nixl_prepare(self, request_msg: ZMQMessage) -> ZMQMessage:
        """Allocate a receive range; return its descriptors or BUSY/FALLBACK."""
        request_id = request_msg.body["request_id"]
        try:
            self._require_nixl()
            payload_len = request_msg.body["payload_len"]
            global_indexes = request_msg.body["global_indexes"]

            if payload_len > self._arena.size:
                return self._nixl_status_response(ZMQRequestType.PUT_DATA_NIXL_RESPONSE, "FALLBACK")

            offset = self._arena.allocate(payload_len)
            if offset is None:
                return self._nixl_status_response(ZMQRequestType.PUT_DATA_NIXL_RESPONSE, "BUSY")

            self._nixl_pending[request_id] = (offset, payload_len, global_indexes)
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_NIXL_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"status": "OK", "serialized_descs": self._arena.serialized_descs(offset, payload_len)},
            )
        except Exception as e:
            logger.error(f"[{self.storage_unit_id}]: NIXL put prepare failed: {type(e).__name__}: {e}")
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"message": f"Failed to prepare NIXL put on {self.storage_unit_id}: {e}"},
            )

    def _handle_put_nixl_commit(self, request_msg: ZMQMessage) -> ZMQMessage:
        """Copy the received arena range out and deserialize it into the dict (storage unchanged)."""
        request_id = request_msg.body["request_id"]
        pending = self._nixl_pending.pop(request_id, None)
        if pending is None:
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"message": f"NIXL put commit for unknown request_id {request_id}."},
            )

        offset, payload_len, global_indexes = pending
        try:
            field_data: dict[str, Any] = pickle.loads(self._arena.read_bytes(offset, payload_len))
            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS_DEFAULT, info=f"[{self.storage_unit_id}] _handle_put_nixl_commit"
            ):
                self.storage_data.put_data(field_data, global_indexes)
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_NIXL_COMMIT_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={},
            )
        except Exception as e:
            logger.error(f"[{self.storage_unit_id}]: NIXL put commit failed: {type(e).__name__}: {e}")
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"message": f"Failed to commit NIXL put on {self.storage_unit_id}: {e}"},
            )
        finally:
            self._arena.free(offset)

    # ---- helpers --------------------------------------------------------------------

    def _nixl_status_response(self, response_type, status: str) -> ZMQMessage:
        return ZMQMessage.create(
            request_type=response_type,  # type: ignore[arg-type]
            sender_id=self.storage_unit_id,
            body={"status": status},
        )
