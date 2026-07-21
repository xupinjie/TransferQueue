# NIXL 后端设计文档（基于 SimpleStorage）

> 状态：**设计草案，待讨论**。本文只描述设计，不含实现。
> 目标读者：TransferQueue 维护者。

## 1. 背景与目标

TransferQueue 现有两类 NVIDIA 相关后端：

- **SimpleStorage**：数据以 Python dict 形式存放在若干 `SimpleStorageUnit`（Ray actor）进程内存中，manager 与 SU 之间通过 **ZMQ over TCP** 传输控制消息 **和** 张量数据。
- **MooncakeStore**：KV 语义后端，数据交给 Mooncake transfer engine，支持 RDMA / GDR。

本设计在 **SimpleStorage 之上引入 NIXL 后端**：

- **存储与控制逻辑完全沿用 SimpleStorage**：路由（`global_idx % num_su`）、`StorageUnitData` 内存 dict、handshake、`notify_data_update`、checkpoint 等一律不改。
- **只把"数据面"（bulk 张量搬运）从 ZMQ/TCP 换成 NIXL**（UCX/RDMA，后续可扩展 GDR）。

当前阶段目标是**功能接入**，不追求极致性能；但架构要能在后续平滑演进到零拷贝 / GDR。

## 2. 现状：SimpleStorage 数据通路

关键文件：
- [transfer_queue/storage/simple_storage.py](../../transfer_queue/storage/simple_storage.py) — `SimpleStorageUnit`（ROUTER/DEALER + worker 线程，dict 存储）。
- [transfer_queue/storage/managers/simple_storage_manager.py](../../transfer_queue/storage/managers/simple_storage_manager.py) — `AsyncSimpleStorageManager`（路由 + 每 SU 一个 ZMQ 请求）。

`get_data` 现有流程（`put_data` 对称）：

1. Manager（运行在 client 侧：trainer/inference worker 或 driver）收到 `get_data(metadata)`。
2. 按 `global_idx % num_su` 把请求分组到各 SU（`_group_by_hash`）。
3. 对每个 SU 发 ZMQ `GET_DATA` 请求（`_get_from_single_storage_unit`）。
4. SU worker 线程从内存 dict 取张量，**序列化后经 ZMQ multipart 返回**。
5. Manager 反序列化、按 batch 位置拼回 `TensorDict`。

**要害**：第 4/5 步中，张量的**原始字节**是塞进 ZMQ 消息体走 TCP 的。这正是 NIXL 要替换的部分。控制信息（`global_indexes`、`fields`、`field_schema`、错误处理、`notify`）继续走 ZMQ。

## 3. NIXL 能力速览

参考 [basic_two_peers.py](../../deps/nixl/examples/python/basic_two_peers.py) 与 `nixl._api`：

- `nixl_agent(name, nixl_agent_config(...))`：每进程一个 agent，可开 listener 线程用于元数据交换。
- `register_memory(tensor / buffer)` → `reg_descs`：把一段内存注册为 RDMA MR（较贵，应尽量一次性）。
- `get_xfer_descs([tensor_rows...])`：构造传输描述符（地址+长度+设备）。
- 元数据交换：`get_agent_metadata()` / `add_remote_agent(bytes)`，或 `send_local_metadata` / `fetch_remote_metadata`（走 NIXL 自己的 socket）。
- 单边传输：`initialize_xfer("READ"|"WRITE", local_descs, remote_descs, peer_name, notif)` → `transfer(handle)` → 轮询 `check_xfer_state(handle)` 直到 `"DONE"`。

**核心约束**：传输前，**双方待传内存都必须已注册**，且 initiator 需要拿到对端目标区域的 descriptors（地址/长度）。这决定了下面的内存注册方案。

## 4. 总体架构

```
        ┌─────────────────────────────┐         ┌──────────────────────────────┐
        │  NixlStorageManager (client)│         │  NixlStorageUnit (Ray actor) │
        │  = AsyncSimpleStorageManager│         │  = SimpleStorageUnit + NIXL  │
        │    + nixl_agent (initiator) │         │    + nixl_agent (target)     │
        └─────────────┬───────────────┘         └───────────────┬──────────────┘
                      │  ① 控制/元数据 (ZMQ, 不变)             │
                      │─────────────────────────────────────►│  路由/dict存储/notify
                      │      请求 + descriptors + schema        │  全部沿用 SimpleStorage
                      │                                        │
                      │  ② bulk 张量字节 (NIXL: UCX/RDMA)      │
                      │◄════════════════ READ ════════════════│  (get)
                      │═════════════════ WRITE ═══════════════►│  (put)
```

- **控制面（ZMQ）不变**：沿用 `put_get_socket` ROUTER + worker 线程，请求/响应、错误、handshake、notify 全部复用。
- **数据面（NIXL）新增**：Manager 恒为 **initiator**（与现有"manager 主动请求"模型一致）；SU 恒为被动 **target**，只暴露已注册的 buffer。
  - `get_data`：Manager 发起 **READ**（从 SU buffer 读到本地 buffer）。
  - `put_data`：Manager 发起 **WRITE**（把本地 buffer 写入 SU buffer）。

## 5. 关键设计决策（含备选方案）

### 5.1 角色：Manager = initiator，SU = target ✅

理由：现有语义就是 manager 主动 push/pull，SU 被动响应。让 manager 做 initiator 可完全复用现有请求/响应时序，SU 侧只需在 worker 里多一步"准备 buffer + 回 descriptor"。反向（SU 做 initiator）需要 SU 主动连 manager，破坏现有模型，否决。

### 5.2 内存注册策略（**主要分歧点，需讨论**）

SU 的数据是运行时动态生成的任意 shape/dtype 张量，存在 dict 里，**天生不在已注册的 MR 中**。三种方案：

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **A. 固定 staging buffer**（推荐） | SU 启动时预注册一块固定大小的 pinned CPU buffer；传输时把目标张量 **序列化/拷贝进 staging buffer**（连续），把 offset/len 作为 descriptor 回给 manager；manager 也预注册一块本地 staging buffer，NIXL 传输后再反序列化 | 注册一次、生命周期简单、鲁棒；与 Mooncake GDR 的 staging buffer 思路一致；实现最快 | 多一次 memcpy；buffer 满需分块；仍有序列化开销 |
| **B. 按需注册真实张量** | 每次请求把 dict 里的真实张量临时 `register_memory`，传完再 `deregister` | 数据面零拷贝 | 每请求注册/反注册开销高；生命周期复杂（传输完成前不能反注册/释放）；碎片化多 descriptor |
| **C. blob-over-NIXL**（最小改动） | 完全保留现有序列化逻辑，只是把"序列化后的整块 bytes"从 ZMQ 帧改成经 NIXL 传一块（按需注册那块 bytes） | 改动面最小、复用现有序列化 | 仍非零拷贝；本质只把 TCP 换成 RDMA 传输 |

**推荐：方案 A（staging buffer）**，作为可长期演进的目标形态。它把"注册"与"动态数据"解耦，一次注册即可，且天然为后续 GDR（把 staging buffer 换到 GPU）留了口子。方案 C 可作为 Phase 0 快速打通验证。

### 5.3 传输粒度：序列化 blob vs 张量原生

- **Phase 0 / 方案 C+A**：继续用现有序列化（msgpack/pickle 张量）得到连续字节，NIXL 只搬字节。**功能优先、风险低**。
- **Phase 1 目标**：张量原生（tensor-native）——ZMQ 只传 shape/dtype/stride/offset 元数据，NIXL 直接搬张量原始 buffer，实现数据面零拷贝，才是 NIXL 的真正收益点，也是 GDR 的前提。

当前阶段先做 Phase 0，把 Phase 1 作为明确的后续演进项。

### 5.4 NIXL agent 元数据交换：复用 ZMQ ✅

NIXL 两种元数据交换方式：(a) 自带 listener + `fetch_remote_metadata(ip,port)`；(b) 手动 `get_agent_metadata()` bytes → 对端 `add_remote_agent()`。

**推荐 (b) 复用现有 ZMQ 通道**：在 handshake / 首次请求时，把 SU 的 `get_agent_metadata()` bytes 随 `ZMQServerInfo` 或 ZMQ 响应带给 manager，manager `add_remote_agent`。好处：不额外开 NIXL 监听端口、bootstrap 与现有 ZMQ 流程一致、多节点端口管理更简单。每次请求的 **per-transfer descriptors** 也走 ZMQ 响应体传回。

## 6. 数据流详解（方案 A + Phase 0）

### get_data（Manager READ）

1. Manager `_get_from_single_storage_unit`：发 ZMQ 新请求 `GET_DATA_NIXL{global_indexes, fields}`。
2. SU worker：从 dict 取数据 → 序列化进 SU staging buffer（连续）→ 回 ZMQ 响应 `{nixl_descs(serialized), payload_len, ser_meta}`。
3. Manager：确保本地 staging buffer ≥ payload_len（否则分块）→ `initialize_xfer("READ", local_descs, remote_descs, su_agent, notif)` → `transfer` → await 轮询 `DONE`。
4. Manager：从本地 buffer 反序列化 → 拼回 `TensorDict`（复用现有 `_pack_field_values` 逻辑）。
5. SU：收到 NIXL 完成 notif 后可复用/释放 staging 区（或用 slot 机制）。

### put_data（Manager WRITE）

1. Manager：序列化选中的 field 切片进本地 staging buffer。
2. 发 ZMQ `PUT_DATA_NIXL{global_indexes, payload_len, ser_meta}`；SU 回可写入的 `remote_descs`（其 staging buffer 区域）。
3. Manager：`initialize_xfer("WRITE", ...)` → `transfer` → await `DONE`。
4. SU：NIXL notif 触发后，从 staging buffer 反序列化 → `StorageUnitData.put_data`（**存储逻辑完全不变**）→ 回 ZMQ ACK。
5. Manager：收到 ACK 后 `notify_data_update`（**不变**）。

> 注意点：ZMQ 请求/响应与 NIXL 传输的**顺序编排 + 背压**（staging buffer 并发复用、分块、await 轮询融入 asyncio）是实现重点。首版可用"每 SU 单飞行 + buffer 加锁串行化"简化。

## 7. 模块与代码结构（尽量子类化，最小新增）

```
transfer_queue/storage/
├── nixl_storage.py                     # NixlStorageUnit(SimpleStorageUnit)：+nixl agent, +staging buffer, +GET/PUT_DATA_NIXL 处理
├── managers/
│   └── nixl_storage_manager.py         # NixlStorageManager(AsyncSimpleStorageManager)：override _get/_put_to_single_storage_unit
└── bootstrap/
    └── nixl_storage_bootstrap.py       # @register_provider("NixlStorage")：起 NixlStorageUnit + 收集 zmq_info & nixl agent metadata
```

- `NixlStorageUnit` **子类化** `SimpleStorageUnit`：复用 ZMQ 服务端 / worker 循环 / dict 存储 / checkpoint；仅新增 nixl agent 初始化、staging buffer、两个新 operation 分支。
- `NixlStorageManager` **子类化** `AsyncSimpleStorageManager`：复用 `_group_by_hash`、`put_data`/`get_data` 外层、`notify`；仅 override 单 SU 传输方法走 NIXL。
- 工厂注册：`@StorageManagerFactory.register("NixlStorage")`，与 `StorageBootstrapProvider.register_provider("NixlStorage")` 对齐（参考 [__init__.py](../../transfer_queue/storage/__init__.py) 与 [simple_storage_bootstrap.py](../../transfer_queue/storage/bootstrap/simple_storage_bootstrap.py)）。

### 新增 ZMQRequestType
`GET_DATA_NIXL` / `GET_DATA_NIXL_RESPONSE` / `PUT_DATA_NIXL` / `PUT_DATA_NIXL_RESPONSE`（在 `utils/zmq_utils.py` 的枚举里加）。控制/错误路径复用现有 `*_ERROR` 类型。

### 新增配置（`conf.backend.NixlStorage`）
| 字段 | 默认 | 说明 |
|---|---|---|
| `num_data_storage_units` | — | 同 SimpleStorage |
| `total_storage_size` | None | 同 SimpleStorage |
| `nixl_backends` | `["UCX"]` | NIXL agent 使用的后端插件 |
| `staging_buffer_mb` | 256 | 每个 SU / manager 的注册 staging buffer 大小；`0` 时回退纯 ZMQ 数据面 |
| `device_name` | "" | RDMA NIC，空为自动选择 |

## 8. 分阶段实施计划

- **Phase 0（功能打通）**：方案 C + A 的最小组合——复用序列化，NIXL 只搬连续 blob，DRAM↔DRAM（UCX/RDMA）。跑通 UT + e2e，正确性对齐 SimpleStorage。
- **Phase 1（张量原生零拷贝）**：ZMQ 只传元数据，NIXL 直搬张量 buffer；staging buffer 分块 + 并发复用。
- **Phase 2（GDR）**：staging buffer 落 GPU，GPU↔NIC 直传，复用 mooncake_gdr 的"CUDA context 已初始化才启用、否则回退"的策略。

## 9. 风险与开放问题（请讨论）

1. **NIXL 在 Ray actor 内的可用性**：UCX 插件 / RDMA 设备在容器内是否可见、是否需按 SU 绑定特定 NIC（参考 EOS 8×CX-7 拓扑）。需在计算节点实测。
2. **staging buffer 并发模型**：首版是否接受"每 SU 串行传输"以换取简单？还是一开始就做多 slot。
3. **传输粒度**：Phase 0 是否可接受"仍走序列化"（即短期收益仅 TCP→RDMA），还是希望直接上 tensor-native。
4. **元数据交换**：确认走 ZMQ 携带 nixl metadata（方案 5.4b），不额外开 NIXL listener。
5. **回退策略**：`staging_buffer_mb=0` 或 NIXL 不可用时，是否自动回退到 SimpleStorage 的 ZMQ 数据面（利于单一配置覆盖 GPU/CPU worker）。

## 10. Phase 0 实现说明（历史，已被 §10.5 Phase 1 取代）

> ⚠️ Phase 0 的"按需注册"数据面**已被 Phase 1 的常驻 arena 取代**（原因见 §10.5：按需注册在 NIXL 元数据快照语义下很可能失败）。以下保留作演进记录。

代码最初实现时相对设计有一处调整：

- **内存注册：改用"按需注册"而非固定 staging buffer**（即从方案 A 调整为方案 B 的注册方式 + 保留序列化）。原因：进入代码层后发现固定 staging buffer 会引入 **分块 + 跨网络往返的单 buffer 加锁** 复杂度，与"正确性优先"冲突。按需注册（每请求 `register_memory` 一块 exact-size buffer、传完 `deregister`）反而**没有锁、没有分块、并发天然隔离**（各请求用 `request_id` 各自的 buffer），最稳。代价是每次传输有注册开销——这正是 Phase 1 要换回 staging buffer 优化的点。
- **序列化**：`pickle`（`storage_data.get_data` 结果 / manager 选中的 field 切片 dict）。
- **角色/信令**：manager 恒为 initiator；`GET` = manager READ + 之后 ZMQ `RELEASE`；`PUT` = manager WRITE + 之后 ZMQ `COMMIT`。全部信令走 ZMQ，未用 NIXL notif。
- **元数据交换**：懒式，每个 (manager, SU) 对首次传输前经 ZMQ 交换 `get_agent_metadata()` 并双向 `add_remote_agent`。
- **`data_parser`**：Phase 0 NIXL 不承载；`data_parser is not None` 的 put 自动回退走父类 ZMQ 数据面（功能不丢）。
- **回退**：`use_nixl=False` 或 NIXL import 失败时，manager/unit 均退化为 SimpleStorage 的 ZMQ 数据面。
- **代码改动**：
  - `SimpleStorageUnit` 拆为 plain `SimpleStorageUnitBase` + 薄 `@ray.remote` 子类（Ray 不允许继承已装饰的 actor 类），行为不变。
  - `SimpleStorageUnitBase` 新增 no-op 钩子 `_handle_extended_operation`（默认返回 None，行为不变），供子类扩展 operation。
  - 新增文件：`storage/nixl_storage.py`、`storage/managers/nixl_storage_manager.py`、`storage/bootstrap/nixl_storage_bootstrap.py`；三处 `__init__` 注册；`zmq_utils.py` 新增 NIXL 请求枚举。

### Phase 0 配置示例
```yaml
backend:
  storage_backend: NixlStorage
  NixlStorage:
    num_data_storage_units: 4
    total_storage_size: null      # 同 SimpleStorage，null=无限
    use_nixl: true                # false 时回退纯 ZMQ 数据面
    nixl_backends: ["UCX"]        # NIXL agent 后端插件
```

## 10.5 Phase 1 实现说明（已落地：落地区 arena，方案 X）

Phase 1 用**常驻注册 arena（落地区）**取代 Phase 0 的按需注册，dict 存储不变。

**动机（含修正 Phase 0 的隐患）**：NIXL 元数据是快照——`add_remote_agent` 只在 handshake 那一刻拿到对端已注册区域；xfer desc 不带 rkey，rkey 靠 initiator 从这份快照查。Phase 0 在 handshake **之后**才每请求注册，manager 快照里没有这些 buffer → 传输很可能失败。arena 在 handshake **之前**注册一次，快照即包含它，之后所有传输打它的**子段**（NIXL 支持子段寻址），彻底消除注册顺序依赖，且注册一次。

**为什么 arena 是 SU 自持的大 buffer 而非 per-manager 私有**：per-manager 私有落地区在"百台机器同时写"时内存爆炸（`N×buffer`）。SU 自持一块大 buffer，多个 manager 并发 RDMA 写它的**不同 offset** 天然无冲突（网卡各自 DMA 非重叠区间），只有"分配 offset"需串行（在 SU 单线程 worker 内，极快）。

**结构**：
- `utils/nixl_utils.py::NixlArena`——一块 CPU `uint8` buffer，`__init__` 时 `register_memory` 一次；first-fit + 合并的变长分配器（256B 对齐）；`allocate/free/view/xfer_descs/serialized_descs/write_bytes/read_bytes`。manager 和 SU **各持一块**（大小 `nixl_arena_mb`，默认 512）。
- SU（`NixlStorageUnit`）：GET prepare 把数据 pickle 进 arena 分配段、回 descs；GET release 释放段。PUT prepare 分配接收段、回 descs；PUT commit 把段拷出反序列化进 dict、释放段。**dict 存储/clear/checkpoint 全不变**。
- manager（`NixlStorageManager`）：本地 arena 分配段做 READ/WRITE 的本地端；恒 initiator。

**背压与回退**：
- SU arena 满 → prepare 回 `status="BUSY"` → manager 退避重试（`TQ_NIXL_BUSY_MAX_RETRIES` 次），超限回退 ZMQ 数据面。
- 单 payload > arena → `status="FALLBACK"`（GET）或 manager 上传前自查（PUT）→ 该次走 ZMQ 数据面。
- manager 本地 arena 暂满 → 短退避重试（`TQ_NIXL_LOCAL_ALLOC_RETRIES`），超限回退 ZMQ。
- 资源生命周期：SU 一旦 prepare 成功分配了段，manager 必发对应 release（GET）/commit（PUT）释放它——即使本地分配失败或传输报错也在 finally 中补发。

**已知限制**：
- 仍是 pickle 序列化（非 tensor-native 零拷贝）；数据仍住 dict，传输前后各一次 arena↔dict 的 memcpy。零拷贝（数据直接住 arena、`key→(offset,len,dtype,shape)`）是"方案 Y"，留作后续。
- `data_parser != None` 的 put 回退 ZMQ（不变）。
- 极端情况：WRITE 成功后 commit 的网络发送本身报错，可能残留一个 SU 段（RDMA 致命错误场景，非稳态问题）。

### Phase 1 配置示例
```yaml
backend:
  storage_backend: NixlStorage
  NixlStorage:
    num_data_storage_units: 4
    total_storage_size: null
    use_nixl: true
    nixl_backends: ["UCX"]
    nixl_arena_mb: 512        # 每个 manager / SU 的常驻注册 arena 大小(MB)
```

## 11. 测试计划

- **UT**：`NixlStorageUnit` 的 put/get/clear 正确性（mock NIXL 或 loopback agent）。
- **e2e**：复用 run-tests / run-perftest 框架，新增 `NixlStorage` backend；与 SimpleStorage 做数值对齐 + 单/多节点。
- **性能对比**：接入 `scripts/performance_test`，与 SimpleStorage(TCP) / Mooncake(RDMA,GDR) 三方对比。
```
```
