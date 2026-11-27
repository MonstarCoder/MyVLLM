<h1><img src="assets/myvllm-logo.png" alt="" width="88" height="88" align="absmiddle"> MyVLLM</h1>

MyVLLM 是面向 **Qwen3.8-27B 纯文本生成**的推理引擎，目前处于开发阶段。权重使用 BF16，Gated DeltaNet（GDN）递归状态按模型配置使用 FP32。[模型配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)

## 开发入口

阅读[架构与学习文档](docs/README.md)了解引擎的执行流程。运行 `python -m unittest discover -s tests -v` 验证请求调度。关键改动的文档要求见[开发约定](AGENTS.md)。

## 功能清单

已勾选的功能附有对应的测试或实验结果。

### 模型与生成

- [x] **文本模型与权重加载** — 从官方 checkpoint 加载文本主干，用同一组输入 token 对照 Transformers 的 prefill logits，记录数值误差。[27B 实测](docs/cpu-model.md#实测记录)
- [x] **增量 decode** — 同一序列逐 token 解码与整段重算对齐，覆盖门控全注意力和 GDN 的卷积、递归状态更新。[分块与整段对齐测试](tests/test_cpu_model.py)
- [ ] **离线生成与采样** — 支持文本和 token ID 输入；验证 greedy、temperature、top-k、top-p，以及 EOS 和最大输出长度。

### 缓存

- [ ] **分页 KV** — 实现物理页池、请求页表和槽位映射。在页边界及其前后各一个 token 的位置，验证分页缓存与连续缓存的输出一致。
- [ ] **GDN 状态池** — 两个请求交错解码时分别与独立运行对齐；结束或取消后，复用槽位的新请求与从空状态启动的结果一致。
- [ ] **混合模型前缀复用** — 按块哈希查找共享前缀，从 KV 和 GDN 快照都有效的位置恢复。验证共享前缀后分叉、部分命中和淘汰，记录实际省去的 prefill token 数。

### 调度与执行

- [ ] **连续批处理** — 活跃批次逐轮接纳新请求，完成的请求及时退出。回放长短请求交错到达的轨迹，核对输出、页数和槽位回收。
- [ ] **分块 prefill** — 按每轮 token 预算推进长 prompt，中间块跳过采样；验证不同切块位置的最终 logits，并观察插入 decode 后的 token 间隔。
- [ ] **投机解码（greedy）** — 首版用请求历史中的 n-gram 提议后续 token，由 Qwen3.8 一次前向验证整段草稿。覆盖无草稿、全部接受、部分接受、首个即拒绝、EOS 和输出长度上限。逐 token 对照普通 greedy 解码，并验证 KV、GDN 卷积与递归状态都恢复到已确认且实际执行过的 token 边界。
- [ ] **原生 MTP 草稿** — 加载 Qwen3.8-27B checkpoint 的 `mtp.*` 参数，使用目标模型隐藏状态和已选 token 生成候选。先对齐单步草稿 logits，再验证多步提议与拒绝后的续算。复用 greedy 验证与状态回滚路径。[官方权重索引](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/model.safetensors.index.json)
- [ ] **DFlash 块草稿（DFlash2）** — 接入与 Qwen3.8-27B 配套的独立草稿模型，按其配置采集目标层隐藏状态并并行提议一块候选。覆盖部分接受后下一块的特征与缓存位置对齐，逐 token 对照普通 greedy 输出。[草稿模型配置](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/blob/main/config.json)
- [ ] **高性能算子接入** — 验证分页 attention 和 GDN 优化前后的数值误差，分别测量 prefill 和 decode 的算子耗时。
- [ ] **Decode CUDA Graph** — 覆盖不同 batch 桶及 eager 回退；验证填充槽位不会修改活跃请求的状态，测量开关前后的 decode 耗时。
- [ ] **CPU/GPU 重叠调度** — 在新请求加入、取消及结束时验证结果归属和资源回收，用 trace 检查 CPU 调度与 GPU 执行的重叠区间。
- [ ] **张量并行** — 在真实双卡上验证 TP=2 权重切分、集合通信和各 rank 的请求顺序，以 TP=1 输出为正确性参照。

### 服务入口

- [ ] **在线生成** — 验证 `/generate`、`/v1/models` 和 `/v1/chat/completions`；greedy 模式下，流式拼接结果与非流式输出一致。
- [ ] **请求取消** — 客户端在 prefill 或 decode 阶段断开后，请求退出调度，其他请求继续生成，资源可被下一条请求复用。
- [ ] **交互式对话** — 终端支持多轮文本生成和重置历史，重置后的首轮结果与新会话一致。

### 性能评测

- [ ] **可复现实验** — 固定同一权重 revision、BF16、硬件、并行度、显存预算和请求回放；记录两套引擎版本、启动参数、预热方式和逐轮原始结果。
- [ ] **离线对测** — 单请求短入长出测逐 token 延迟，多请求长入短出测 prefill 吞吐；分别报告 MyVLLM 与 vLLM 的结果。
- [ ] **在线对测** — 在相同到达率下回放长短混合请求和共享前缀冷、热负载，比较 p95 首 token 延迟、p95 输出 token 间隔及完成率。
- [ ] **性能追平** — 在预先固定的场景中逐项验证吞吐至少达到 vLLM、延迟至多等于 vLLM。记录重复测量的波动和未达标项，独立跟踪此项与对测完成状态。
- [ ] **消融实验** — 在相同请求序列下分别开关前缀缓存和重叠调度，记录吞吐、首 token 延迟和显存占用的变化。
- [ ] **投机解码消融** — 在相同请求回放和输出预算下分别测量关闭投机、n-gram、MTP、DFlash2 四种模式。记录各草稿方案的提议耗时、验证耗时与接受率，以及各模式每次目标模型前向产出的输出 token 数、单请求生成延迟、聚合吞吐和峰值显存。分别报告低并发与高并发结果。
