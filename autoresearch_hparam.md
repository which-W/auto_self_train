# autoresearch_hparam

这是一个让 LLM 自主对 Transformer 语言模型进行超参数调优的实验框架。

## 背景与目标

目标是在固定数据集上找到最优超参数组合，使验证集损失（val_loss）最低。所有实验通过 wandb 追踪，每次实验都是独立 run。

**核心原则：小步探索、大步确认。**
- **探索阶段**（`--total_steps 500`）：快速 5 分钟内测试一个超参数方向是否有潜力
- **确认阶段**（`--total_steps 5000`）：对有潜力的方向做完整验证

---

## Setup

在开始实验前，依次完成以下步骤：

1. **确定 run tag**：基于今天日期提议一个 tag（如 `apr6`）。分支 `autoresearch/<tag>` 不能已存在——这是全新实验。

2. **创建分支**：
   ```bash
   git checkout -b autoresearch/<tag>
   ```

3. **读取关键文件**：
   - `train.py` —— 训练脚本，唯一允许修改的文件
   - `dataset_process.py` —— 数据处理脚本（只读，不修改）
   - 确认 `--train_data_path` 和 `--valid_data_path` 的实际路径（数据需为 `.npy` memmap 格式，`np.uint16` 类型）

4. **初始化结果记录文件**：
   ```bash
   echo -e "commit\tval_loss\tmemory_gb\twandb_run\tstatus\tdescription" > results.tsv
   ```

5. **确认 wandb 配置**：检查 wandb 是否已登录（`wandb login`），项目名使用 `transformer-lm` 或在 `--wandb_project` 中指定。

6. **确认就绪后，启动实验循环。**

---

## 实验规范

### 允许修改的内容（`train.py`）

所有以下内容均可调整：

| 类别 | 参数 |
|------|------|
| 模型架构 | `--d_model`, `--n_head`, `--n_layer`, `--d_ff`, `--max_seq_len`, `--vocab_size`, `--theta` |
| 正则化结构 | `--no_rms_norm`, `--norm_rope` (pre/post), `--no_rope` |
| 激活函数 | `--ffn_type` (swiglu/silu) — **注意：当前 TransformerBlock 硬编码使用 SwiGLU，此参数暂不生效** |
| 优化器 | `--max_lr`, `--min_lr`, `--weight_decay`, `--warmup_steps` |
| 训练过程 | `--batch_size`, `--max_grad_norm`, `--eval_steps` |
| 检查点 | `--checkpoint_dir`, `--save_interval`, `--resume_from` |
| 其他 | `train.py` 中任何可改的代码逻辑 |

### 不允许修改的内容

- 数据处理脚本（`dataset_process.py`）
- 评估逻辑
- 安装新的依赖包

---

## 两阶段实验策略

### 阶段一：快速探索（每次约 2–3 分钟）

使用较小 `total_steps` 快速验证方向：

```bash
uv run train.py \
  --train_data_path <路径> \
  --valid_data_path <路径> \
  --total_steps 500 \
  --eval_interval 100 \
  --log_interval 50 \
  --use_wandb \
  --wandb_project transformer-lm \
  --wandb_run_name "explore_<tag>_<实验描述>" \
  <其他超参数> \
  > run.log 2>&1
```

探索阶段的判断标准：
- val_loss 比当前最优下降 > 0.005 → 值得进入确认阶段
- val_loss 持平或上升 → 放弃

### 阶段二：完整确认（每次约 5 分钟）

对有潜力的配置使用更大 `total_steps` 做完整评估：

```bash
uv run train.py \
  --train_data_path <路径> \
  --valid_data_path <路径> \
  --total_steps 5000 \
  --eval_interval 500 \
  --log_interval 100 \
  --use_wandb \
  --wandb_project transformer-lm \
  --wandb_run_name "confirm_<tag>_<实验描述>" \
  <其他超参数> \
  > run.log 2>&1
```

---

## Wandb Run 命名规范

每次实验生成独立 run，命名格式：

```
<阶段>_<tag>_<描述>
```

示例：
- `explore_apr6_lr3e3_swiglu`
- `confirm_apr6_lr3e3_swiglu`
- `explore_apr6_post_norm_no_rope`
- `confirm_apr6_d_model768_n_layer8`

**命名要求**：
- 不能重名（相同 project 下）
- 描述要能唯一标识这次改动
- 探索与确认用不同前缀区分

---

## 读取结果

```bash
# 读取验证集损失（主要指标）—— train.py 输出格式为 "Validation Loss:"
grep "Validation Loss:" run.log | tail -5

# 如果脚本最后有汇总输出
tail -n 20 run.log

# 确认 wandb run 已记录
grep "wandb" run.log | tail -3
```

如果 `grep` 输出为空，说明运行崩溃了：

```bash
tail -n 50 run.log
```

---

## 记录结果（results.tsv）

TSV 格式，列定义如下：

```
commit	val_loss	memory_gb	wandb_run	status	description
```

| 列 | 说明 |
|----|------|
| `commit` | git commit hash（7位） |
| `val_loss` | 最终验证集损失，崩溃填 `0.000000` |
| `memory_gb` | 峰值显存（MB ÷ 1024，保留1位小数），崩溃填 `0.0` |
| `wandb_run` | wandb run 名称（确认阶段填完整名，探索阶段可填 `explore_only`） |
| `status` | `keep` / `discard` / `crash` |
| `description` | 简短描述，不含逗号，使用 tab 分隔 |

示例：

```
commit	val_loss	memory_gb	wandb_run	status	description
a1b2c3d	2.453100	12.3	explore_apr6_baseline	keep	baseline with default params
b2c3d4e	2.401500	12.4	confirm_apr6_lr6e4	keep	increase max_lr to 6e-4
c3d4e5f	2.510000	12.3	explore_apr6_post_norm	discard	post-norm worse than pre-norm
d4e5f6g	0.000000	0.0	explore_apr6_silu	crash	silu variant OOM
```

**注意**：`results.tsv` 不加入 git 追踪（`.gitignore` 或直接不 `git add`）。

---

## 实验循环

```
LOOP FOREVER:

1. 查看当前 git 状态（分支、最新 commit）
2. 选择一个超参数方向进行实验（见下方实验方向清单）
3. 修改 train.py（或仅改命令行参数）
4. git commit（描述本次改动）
5. 【探索阶段】用小 total_steps 跑一次，读取结果
6. 判断是否值得进入确认阶段：
   - 值得 → 跑完整 total_steps，记录 results.tsv，保留 commit
   - 不值得 → git reset 回上一个 keep 状态，记录 discard
7. 如果崩溃：
   - 简单 bug（typo、import）→ 修复后重跑
   - 根本性问题（OOM、架构错误）→ 记录 crash，reset，换方向
8. 回到步骤 1
```

**关键规则**：
- 一旦进入循环，**绝不停下来问人类是否继续**。人类可能在睡觉。
- 如果没有新想法，参考下方实验方向清单，或组合之前的 near-miss。
- 若 10 分钟内还没完成一次完整 eval，kill 进程，记录 crash，换方向。

---

## 推荐实验顺序

### 第一步：确立 baseline

**第一次实验永远是 baseline**，使用 `train.py` 默认参数（不加任何修改），仅添加 wandb 参数：

```bash
uv run train.py \
  --train_data_path <路径> \
  --valid_data_path <路径> \
  --total_steps 5000 \
  --eval_interval 500 \
  --log_interval 100 \
  --use_wandb \
  --wandb_project transformer-lm \
  --wandb_run_name "confirm_<tag>_baseline" \
  > run.log 2>&1
```

### 第二步：学习率扫描（优先级最高）

学习率对 Transformer 影响最大，优先扫描：

| 探索 run | `max_lr` | `min_lr` |
|----------|----------|----------|
| explore_lr_low | `1e-4` | `1e-5` |
| explore_lr_mid | `3e-4` | `3e-5` |（baseline） |
| explore_lr_high | `6e-4` | `6e-5` |
| explore_lr_1e3 | `1e-3` | `1e-4` |

### 第三步：模型大小 vs. 深度

在同等参数量下，宽而浅 vs. 窄而深（**6GB 显存限制下需缩小规模**）：

| 实验 | `d_model` | `n_layer` | `n_head` | `d_ff` | `max_seq_len` | `batch_size` |
|------|-----------|-----------|----------|--------|---------------|--------------|
| baseline | 512 | 4 | 8 | 2048 | 256 | 4 |
| wider_shallow | 512 | 3 | 8 | 2048 | 256 | 4 |
| deeper_narrow | 384 | 6 | 6 | 1536 | 256 | 4 |
| minimal | 256 | 4 | 4 | 1024 | 256 | 8 |

### 第四步：结构消融

| 实验 | 改动 | 命令行开关 | 说明 |
|------|------|-----------|------|
| post_norm | 改为 post-norm | `--norm_rope post` | 注意：当前 TransformerBlock 仅实现 pre-norm，此参数传给了 TransformerLM 但 block 内部未使用 |
| no_rope | 禁用 RoPE | `--no_rope` | 将 theta 设为 None |
| silu | 改用 SiLU | `--ffn_type silu` | **暂不生效**，需先修改 TransformerBlock 支持动态 FFN 选择 |
| no_rms | 移除 RMSNorm | `--no_rms_norm` | 将 final norm 替换为 Identity |

### 第五步：batch size 与 warmup

| 实验 | `batch_size` | `warmup_steps` | 说明 |
|------|-------------|----------------|------|
| baseline_batch | 4 | 1000 | 6GB 显存下的安全值 |
| small_batch | 2 | 500 | 用于更大模型 |
| large_batch | 8 | 2000 | 仅适用于 minimal 配置（d_model=256） |

### 第六步：组合最优配置

将前面各阶段发现的最优设置组合起来，做一次综合实验。

---

## 简洁性原则

- 微小提升（< 0.001）但代码复杂度大幅增加 → **放弃**
- 无提升但代码更简洁 → **保留**（简化胜利）
- 提升显著（> 0.005）即使代码略有增加 → **保留**
- 相同 val_loss，参数量更少 → **保留**

---

## 显存限制

**硬件约束**：
- 总 VRAM：16GB
- 实际可用 N 卡内存：**6144 MiB（6GB）**
- 安全上限：建议单次实验显存占用不超过 **5.5 GB**（预留 10% 余量）

### 显存监控

实验开始前确认当前显存状态：

```bash
# 查看当前显存占用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

### 6GB 显存下的推荐配置

基于 6GB 可用显存，以下配置可安全运行：

| 配置 | `d_model` | `n_layer` | `n_head` | `d_ff` | `max_seq_len` | `batch_size` | 预估显存 |
|------|-----------|-----------|----------|--------|---------------|--------------|----------|
| 最小 | 256 | 4 | 4 | 1024 | 256 | 8 | ~2.5 GB |
| baseline | 512 | 4 | 8 | 2048 | 256 | 4 | ~4.5 GB |
| 上限 | 512 | 6 | 8 | 2048 | 512 | 2 | ~5.5 GB |

**显存超限处理**：
- OOM 时优先减小 `batch_size`，其次减小 `max_seq_len`
- `d_model` 和 `n_layer` 对显存影响最大，调整需谨慎
- 预估显存公式（粗略）：`显存(GB) ≈ batch_size × seq_len × d_model × n_layer × 0.000002`

## 超时与崩溃处理

- **超时**：单次 run 超过 10 分钟 → `kill` 进程，记录 `crash`，`git reset`
- **OOM**：记录 `crash`，尝试减小 `batch_size`、`max_seq_len` 或 `d_model`
- **显存超限**：预估显存 > 5.5 GB 的配置禁止运行，先缩小规模
- **连续 3 次 crash**：退回已知最优配置，换一个完全不同的方向
- **运行卡住**：检查 GPU 是否空闲，检查数据路径是否正确

---

## 实验结束条件

人类手动中断为止。在此之前：

1. val_loss 长期不下降（连续 10 次 discard） → 尝试更激进的架构变化
2. 所有推荐实验方向已尝试 → 尝试组合，或参考 Transformer 相关论文中的改进
3. **不要停下来问人类**

---

## 快速参考：命令模板

### 探索阶段（快速）

```bash
uv run train.py \
  --train_data_path DATA_TRAIN \
  --valid_data_path DATA_VAL \
  --total_steps 500 \
  --eval_interval 100 \
  --log_interval 50 \
  --batch_size 4 \
  --d_model 512 \
  --n_head 8 \
  --n_layer 4 \
  --d_ff 2048 \
  --max_seq_len 256 \
  --max_lr 3e-4 \
  --min_lr 3e-5 \
  --warmup_steps 100 \
  --use_wandb \
  --wandb_project transformer-lm \
  --wandb_run_name "explore_TAG_DESC" \
  > run.log 2>&1
```

### 确认阶段（完整）

```bash
uv run train.py \
  --train_data_path DATA_TRAIN \
  --valid_data_path DATA_VAL \
  --total_steps 5000 \
  --eval_interval 500 \
  --log_interval 100 \
  --batch_size 4 \
  --d_model 512 \
  --n_head 8 \
  --n_layer 4 \
  --d_ff 2048 \
  --max_seq_len 256 \
  --max_lr 3e-4 \
  --min_lr 3e-5 \
  --warmup_steps 2000 \
  --use_wandb \
  --wandb_project transformer-lm \
  --wandb_run_name "confirm_TAG_DESC" \
  > run.log 2>&1
```

---

## 已知问题与限制

### 1. TransformerBlock 未使用实验参数
当前 `TransformerBlock` 硬编码了以下行为，**命令行实验参数传入了但不会改变实际行为**：
- **Pre-norm only**：`--norm_rope` 参数传递给 `TransformerLM`，但 `TransformerBlock` 内部仅实现 pre-norm 结构
- **SwiGLU only**：`--ffn_type` 参数传递给 `TransformerLM`，但 `TransformerBlock` 硬编码使用 `SwiGLU`
- **RMSNorm 在 block 内始终启用**：`--no_rms_norm` 仅影响 `TransformerLM` 的 `ln_final`，block 内的 `ln1`/`ln2` 不受影响

若要让消融实验生效，需要修改 `TransformerBlock` 支持：
- 动态选择 pre-norm / post-norm 结构
- 动态选择 SwiGLU / SiLU 激活函数
- 可选禁用 block 内的 RMSNorm

### 2. 数据格式要求
- 数据需为 `np.memmap` 格式，`dtype=np.uint16`（用于支持 > 32767 的词表）
- 文件扩展名不限，但内容必须是连续的 token ID 序列
- 使用 `dataset_process.py` 生成符合格式的数据文件

### 3. 学习率调度器
- 使用 `CosineAnnealingWarmupScheduler.get_lr_cosine_shedule()` 方法（注意方法名拼写为 `shedule` 而非 `schedule`）
- 预热阶段：从 0 线性增长到 `max_lr`
- 退火阶段：余弦衰减到 `min_lr`

---

