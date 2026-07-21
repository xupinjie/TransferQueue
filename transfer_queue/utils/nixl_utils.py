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

"""Persistent NIXL staging arena.

A single large CPU buffer that is registered with the NIXL agent **once** at
construction and reused for the process lifetime. A variable-size sub-allocator
hands out byte ranges (offsets) inside it; each range is addressed as a NIXL
transfer descriptor sub-region of the one registered memory region — so no
per-request registration is needed, and the registration is already present in
the agent metadata by the time peers exchange it.

Concurrency: many peers may RDMA into *different* offsets of the same buffer at
once without conflict (the NIC DMAs non-overlapping ranges independently). The
only serialized part is allocation/free, guarded by a short lock.

This is the landing-zone form (design "X"): data is copied in/out of the arena
around each transfer; the backing dict storage is unchanged. A future zero-copy
form would store data directly in the arena.
"""

import threading

import torch

_DEFAULT_ALIGN = 256


class NixlArena:
    """One register-once CPU buffer plus a first-fit byte-range allocator.

    Args:
        agent: A ``nixl_agent`` used to register the buffer and build descriptors.
        size_bytes: Total arena size in bytes.
        align: Allocation alignment (offsets and sizes are rounded up to this).
    """

    def __init__(self, agent, size_bytes: int, align: int = _DEFAULT_ALIGN):
        if size_bytes <= 0:
            raise ValueError(f"NixlArena size must be positive, got {size_bytes}")
        self._agent = agent
        self._size = size_bytes
        self._align = align
        self._buf = torch.empty(size_bytes, dtype=torch.uint8)
        self._reg = agent.register_memory(self._buf)
        if not self._reg:
            raise RuntimeError("NIXL register_memory returned an empty descriptor list for the arena.")

        self._lock = threading.Lock()
        # Free list of (offset, size) blocks, kept sorted by offset and coalesced.
        self._free: list[tuple[int, int]] = [(0, size_bytes)]
        # offset -> allocated (aligned) size, for free().
        self._alloc: dict[int, int] = {}

    @property
    def size(self) -> int:
        return self._size

    def _align_up(self, n: int) -> int:
        return (n + self._align - 1) // self._align * self._align

    def allocate(self, nbytes: int) -> int | None:
        """Reserve an aligned range of at least ``nbytes``; return its offset or None if full."""
        need = self._align_up(nbytes)
        with self._lock:
            for i, (off, sz) in enumerate(self._free):
                if sz >= need:
                    if sz == need:
                        self._free.pop(i)
                    else:
                        self._free[i] = (off + need, sz - need)
                    self._alloc[off] = need
                    return off
            return None

    def free(self, offset: int) -> None:
        """Return a previously allocated range to the free list and coalesce neighbours."""
        with self._lock:
            size = self._alloc.pop(offset, None)
            if size is None:
                return
            self._free.append((offset, size))
            self._free.sort()
            merged: list[tuple[int, int]] = []
            for off, sz in self._free:
                if merged and merged[-1][0] + merged[-1][1] == off:
                    merged[-1] = (merged[-1][0], merged[-1][1] + sz)
                else:
                    merged.append((off, sz))
            self._free = merged

    def view(self, offset: int, nbytes: int) -> "torch.Tensor":
        """Contiguous uint8 view of ``[offset, offset+nbytes)`` for memcpy in/out."""
        return self._buf[offset : offset + nbytes]

    def xfer_descs(self, offset: int, nbytes: int):
        """NIXL transfer descriptor list for a sub-range (local use)."""
        return self._agent.get_xfer_descs(self.view(offset, nbytes))

    def serialized_descs(self, offset: int, nbytes: int) -> bytes:
        """Serialized transfer descriptors for a sub-range (to send to a peer)."""
        return self._agent.get_serialized_descs(self.xfer_descs(offset, nbytes))

    def write_bytes(self, offset: int, payload: bytes) -> None:
        """Copy ``payload`` into the arena at ``offset`` (host memcpy)."""
        self.view(offset, len(payload)).copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))

    def read_bytes(self, offset: int, nbytes: int) -> bytes:
        """Copy ``nbytes`` out of the arena at ``offset`` into a fresh ``bytes``."""
        return self.view(offset, nbytes).numpy().tobytes()

    def close(self) -> None:
        try:
            self._agent.deregister_memory(self._reg)
        except Exception:
            pass
