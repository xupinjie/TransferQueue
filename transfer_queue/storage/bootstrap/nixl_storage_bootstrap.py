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

import math
from typing import Any

from omegaconf import DictConfig

from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider
from transfer_queue.storage.nixl_storage import NixlStorageUnit
from transfer_queue.utils.common import get_placement_group
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.zmq_utils import process_zmq_server_info

logger = get_logger(__name__)


@StorageBootstrapProvider.register_provider("NixlStorage")
def initialize_nixl_storage(conf: DictConfig) -> dict[str, Any]:
    """Initialize NIXL storage units.

    Mirrors the SimpleStorage bootstrap (same placement / capacity logic) but
    launches ``NixlStorageUnit`` actors and forwards NIXL options.
    """

    nixl_storage_handles = {}
    backend_conf = conf.backend.NixlStorage
    num_data_storage_units = backend_conf.num_data_storage_units
    total_storage_size = backend_conf.get("total_storage_size", None)
    use_nixl = backend_conf.get("use_nixl", True)
    nixl_backends = list(backend_conf.get("nixl_backends", ["UCX"]))
    nixl_arena_mb = int(backend_conf.get("nixl_arena_mb", 512))
    storage_placement_group = get_placement_group(num_data_storage_units, num_cpus_per_actor=1)

    # Compute per-unit capacity: None means unlimited
    storage_unit_size = (
        math.ceil(total_storage_size / num_data_storage_units) if total_storage_size is not None else None
    )

    for storage_unit_rank in range(num_data_storage_units):
        storage_node = NixlStorageUnit.options(  # type: ignore[attr-defined]
            placement_group=storage_placement_group,
            placement_group_bundle_index=storage_unit_rank,
            name=f"TransferQueueStorageUnit#{storage_unit_rank}",
        ).remote(
            storage_unit_size=storage_unit_size,
            use_nixl=use_nixl,
            nixl_backends=nixl_backends,
            nixl_arena_mb=nixl_arena_mb,
        )
        nixl_storage_handles[f"TransferQueueStorageUnit#{storage_unit_rank}"] = storage_node
        logger.info(f"TransferQueueStorageUnit#{storage_unit_rank} (NIXL) has been created.")

    storage_zmq_info = process_zmq_server_info(nixl_storage_handles)
    backend_name = conf.backend.storage_backend
    conf.backend[backend_name].zmq_info = storage_zmq_info

    return nixl_storage_handles
