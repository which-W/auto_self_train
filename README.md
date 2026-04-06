# auto_self_train

一个受 autoresearch 启发的 LLM 自主超参数调优框架。核心 idea：让 AI 在无人干预的情况下，自动设计实验、运行训练、分析结果、迭代方向，最终找到最优超参数组合。

---

## 核心 Idea

传统超参数搜索（网格搜索、随机搜索、贝叶斯优化）是机械的参数遍历。本项目换了一条路：

> **让一个 LLM 像研究员一样思考**——先跑 baseline，再提出假设，设计实验验证，根据结果调整方向，不断循环直到找到最优配置。

整个过程无需人工干预：LLM 自己决定下一个实验是什么、判断是否有潜力、决定放弃还是深入。

---

## 自动调参工作流

```
┌─────────────────────────────────────────────────────────────┐
│                    实验循环 (LOOP FOREVER)                   │
│                                                             │
│  1. 读 git 状态 ──► 2. 选方向 ──► 3. 改参数 ──► 4. commit  │
│       │                                                    │
│       ▼                                                    │
│  5. 探索实验 (500 steps, ~2min)                             │
│       │                                                    │
│       ▼                                                    │
│  6. 判断: val_loss 下降 > 0.005 ?                           │
│     ┌─────┴──────┐                                         │
│     ▼            ▼                                         │
│   值得          不值得                                      │
│     │            │                                         │
│     ▼            ▼                                         │
│  确认实验      git reset ──► 记录 discard                   │
│  (5000 steps)                                               │
│     │                                                    │
│     ▼                                                    │
│  记录 results.tsv ──► 回到步骤 1                            │
└─────────────────────────────────────────────────────────────┘
```

### 关键设计原则

| 原则 | 说明 |
|------|------|
| **小步探索、大步确认** | 探索阶段只跑 500 steps（约 2 分钟），有潜力才跑 5000 steps |
| **绝不停下来问人类** | 一旦进入循环，自动运行，人类可能在睡觉 |
| **git 驱动的实验管理** | 每次实验一个 commit，失败就 reset，保持工作区干净 |
| **TSV 结果追踪** | `results.tsv` 记录每次实验的 commit、val_loss、状态、描述 |
| **Wandb 可视化** | 每个实验独立 run，云端同步，随时查看曲线 |

---

## 两阶段实验策略

### 阶段一：探索（Explore）

```bash
uv run train.py \
  --train_data_path data/TinyStories-train.bin \
  --valid_data_path data/TinyStories-valid.bin \
  --total_steps 500 \
  --eval_interval 100 \
  --log_interval 50 \
  --batch_size 4 \
  --d_model 512 --n_head 8 --n_layer 4 --d_ff 2048 \
  --max_seq_len 256 \
  --max_lr 3e-4 --min_lr 3e-5 \
  --use_wandb \
  --wandb_project transformer-lm \
  --wandb_run_name "explore_apr6_实验描述" \
  > run.log 2>&1
```

- **目的**：快速判断一个超参数方向是否有潜力
- **时间**：约 2-3 分钟
- **判断标准**：val_loss 比当前最优下降 > 0.005 → 进入确认

### 阶段二：确认（Confirm）

```bash
uv run train.py \
  --train_data_path data/TinyStories-train.bin \
  --valid_data_path data/TinyStories-valid.bin \
  --total_steps 5000 \
  --eval_interval 500 \
  --log_interval 100 \
  --batch_size 4 \
  --d_model 512 --n_head 8 --n_layer 4 --d_ff 2048 \
  --max_seq_len 256 \
  --max_lr 3e-4 --min_lr 3e-5 --warmup_steps 2000 \
  --use_wandb \
  --wandb_project transformer-lm \
  --wandb_run_name "confirm_apr6_实验描述" \
  > run.log 2>&1
```

- **目的**：对有潜力的配置做完整验证
- **时间**：约 5-10 分钟
- **判断标准**：val_loss 确认优于之前 → keep，否则 discard

---

## 实验搜索空间

### 模型架构参数

| 参数 | 范围 | 说明 |
|------|------|------|
| `d_model` | 256 / 384 / 512 / 768 | 模型维度 |
| `n_layer` | 3 / 4 / 6 | Transformer 层数 |
| `n_head` | 4 / 6 / 8 / 12 | 注意力头数 |
| `d_ff` | 1024 / 2048 / 4096 | 前馈网络维度 |
| `max_seq_len` | 256 / 512 | 最大序列长度 |

### 优化器参数

| 参数 | 范围 | 说明 |
|------|------|------|
| `max_lr` | 1e-4 / 3e-4 / 6e-4 / 1e-3 | 最大学习率 |
| `min_lr` | max_lr / 10 | 最小学习率 |
| `weight_decay` | 0.01 / 0.1 | 权重衰减 |
| `warmup_steps` | 100 / 500 / 2000 | 预热步数 |

### 训练过程参数

| 参数 | 范围 | 说明 |
|------|------|------|
| `batch_size` | 2 / 4 / 8 | 批次大小 |

### 结构消融开关

| 参数 | 效果 | 说明 |
|------|------|------|
| `--no_rope` | 禁用 RoPE 位置编码 | 验证位置编码的必要性 |
| `--no_rms_norm` | 移除最终 RMSNorm | 验证归一化的必要性 |
| `--norm_rope post` | 改为 post-norm | 归一化位置（当前仅 pre-norm 生效）|
| `--ffn_type silu` | 改用 SiLU | 激活函数（当前仅 SwiGLU 生效）|

---

## 实验推荐的搜索顺序

LLM 按照固定顺序逐步探索，确保不遗漏关键方向：

```
1. Baseline（默认参数，确立基准线）
       │
2. 学习率扫描（优先级最高，影响最大）
   ├── lr=1e-4（低）
   ├── lr=3e-4（中，baseline）
   ├── lr=6e-4（高）
   └── lr=1e-3（极高）
       │
3. 模型大小 vs. 深度（宽而浅 vs. 窄而深）
   ├── n_layer=3（更浅）
   ├── d_model=384, n_layer=6（窄而深）
   └── d_model=256, batch_size=8（最小）
       │
4. 结构消融
   ├── 禁用 RoPE
   ├── 禁用 RMSNorm
   ├── 改为 post-norm
   └── 改用 SiLU
       │
5. Batch size 与 warmup
       │
6. 组合最优配置
```

---

## 结果记录与追踪

### results.tsv

每次实验记录到 `results.tsv`（TSV 格式，不加入 git 追踪）：

```
commit    val_loss    memory_gb    wandb_run    status    description
356d05b   0.7737      -            confirm_apr6_baseline   keep    baseline d512_nhead8_nlayer4...
-         1.2393      -            explore_apr6_lr_low_1e4 discard lr 1e-4 too low
-         1.0782      -            explore_apr6_lr_high_6e4 discard lr 6e-4 too high
```

| 列 | 说明 |
|----|------|
| `commit` | git commit hash（7位） |
| `val_loss` | 最终验证集损失，崩溃填 `0.000000` |
| `memory_gb` | 峰值显存（GB），崩溃填 `0.0` |
| `wandb_run` | wandb run 名称 |
| `status` | `keep` / `discard` / `crash` |
| `description` | 简短描述 |

### Wandb Run 命名

格式：`<阶段>_<tag>_<描述>`

```
explore_apr6_lr_low_1e4
confirm_apr6_baseline
explore_apr6_no_rope
explore_apr6_d_model_768
```

---

## 崩溃与异常处理

| 情况 | 处理方式 |
|------|----------|
| **超时**（> 10 分钟） | kill 进程，记录 `crash`，git reset |
| **OOM** | 记录 `crash`，减小 batch_size 或 d_model |
| **显存超限**（> 5.5 GB） | 禁止运行，先缩小规模 |
| **连续 3 次 crash** | 退回已知最优配置，换方向 |
| **运行卡住** | 检查 GPU 状态、数据路径 |

---

## 简洁性原则

LLM 在决策时遵循以下原则：

| 场景 | 决策 |
|------|------|
| 提升 < 0.001 但代码复杂大增 | 放弃 |
| 无提升但代码更简洁 | 保留（简化胜利）|
| 提升 > 0.005 即使代码略增 | 保留 |
| 相同 val_loss，参数量更少 | 保留 |

---

## 实验结果（apr6 轮次）

### 最优配置

```yaml
d_model:       512
n_head:        8
n_layer:       4
d_ff:          2048
max_seq_len:   256
batch_size:    4
max_lr:        3.0e-4
min_lr:        3.0e-5
warmup_steps:  2000
weight_decay:  0.01
最终 val_loss:  0.7737
```

### 全部实验对比（14 次）

| 实验 | val_loss | 差值 | 结论 |
|------|----------|------|------|
| **baseline** | **0.7737** | **-** | **最优** |
| lr=1e-4 | 1.2393 | +0.46 | 学习率过低 |
| lr=6e-4 | 1.0782 | +0.30 | 学习率偏高 |
| lr=1e-3 | 1.0686 | +0.29 | 学习率过高 |
| n_layer=3 | 1.1450 | +0.37 | 模型太浅 |
| d_model=256 | 1.1887 | +0.41 | 模型太小 |
| d_model=384, n_layer=6 | 1.1732 | +0.39 | 窄而深不如宽而浅 |
| no_rope | 1.0954 | +0.32 | RoPE 必须 |
| no_rms_norm | 1.0170 | +0.24 | RMSNorm 必须 |
| n_layer=6 | 1.1687 | +0.39 | 增加层数无益 |
| d_ff=4096 | 1.0849 | +0.31 | FFN 增大无收益 |
| seq_len=512 | 1.1001 | +0.32 | 序列太长 |
| wd=0.1 | 1.1188 | +0.34 | weight_decay 过大 |
| d_model=768 | 1.1637 | +0.39 | 模型太大 |

### 关键发现

1. **Baseline 即最优**——14 次探索实验均未超越默认配置
2. **RoPE 和 RMSNorm 是必须的**——移除后 val_loss 分别恶化 0.32 和 0.24
3. **lr=3e-4 是甜蜜点**——偏离该值均导致性能显著下降
4. **d_model=512, n_layer=4 是最佳平衡**——更大或更小的模型都更差

---

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 启动实验循环

阅读 `autoresearch_hparam.md` 了解完整实验规范，然后：

```bash
# 创建实验分支
git checkout -b autoresearch/apr6

# 初始化结果文件
echo -e "commit\tval_loss\tmemory_gb\twandb_run\tstatus\tdescription" > results.tsv

# 开始第一轮实验（baseline → 学习率扫描 → 结构探索 → ...）
```

### 3. 查看结果

```bash
# 查看 results.tsv
column -t results.tsv

# 查看 wandb 实验看板
# https://wandb.ai/wengzu-love-tongji-university/transformer-lm
```

---

## 项目结构（自动调参相关）

```
auto_self_train/
├── autoresearch_hparam.md   # 自动调参实验规范文档（核心指南）
├── hparam_config.yaml       # 实验结果超参数配置
├── results.tsv              # 实验结果记录表
├── train.py                 # 训练脚本（实验唯一允许修改的文件）
├── dataset_process.py       # 数据处理脚本（只读）
├── pyproject.toml           # 项目依赖配置
└── data/
    ├── TinyStories-train.bin
    └── TinyStories-valid.bin
```

---

## 硬件约束

| 项目 | 值 |
|------|-----|
| 总 VRAM | 16 GB |
| 实际可用 N 卡内存 | 6144 MiB（6 GB）|
| 安全上限 | 5.5 GB（预留 10%）|
| 推荐 batch_size | 4（baseline）|
| 推荐 max_seq_len | 256 |

---

## 已知限制

1. **TransformerBlock 硬编码行为**——`--norm_rope` 和 `--ffn_type` 参数传入但不生效（block 内部仅实现 pre-norm + SwiGLU）
2. **数据格式**——必须为 `np.memmap` + `dtype=np.uint16`
3. **学习率调度器方法名拼写**——`get_lr_cosine_shedule`（非 `schedule`）
