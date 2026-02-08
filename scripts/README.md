# Scripts

## BCI Agent 训练（当前主线）

### 单配置训练

```bash
# Stage 1: 对齐训练（Qwen 冻结，训练 encoder + embeddings）
bash scripts/train_bci_agent_s1.sh

# Stage 2: 指令微调（LoRA + 混合数据）
bash scripts/train_bci_agent_s2.sh
```

默认使用 REVE + FBCCA FiLM 编码器，6x GPU DeepSpeed ZeRO-2。

### 消融实验

4 种编码器配置 × 2 阶段 = 8 个实验：

| 配置 | backbone | FBCCA | 说明 |
|------|----------|-------|------|
| `reve_fbcca` | REVE (512d) | FiLM | 完整模型 |
| `reve_only` | REVE (512d) | 无 | 无频率信息 |
| `labram_fbcca` | LaBraM (200d) | FiLM | LaBraM + 频率增强 |
| `labram_only` | LaBraM (200d) | 无 | 纯 LaBraM baseline |

```bash
# 全部跑（S1 → S2，约 1.5-2 小时）
bash scripts/train_ablation.sh all

# 只跑 Stage 1
bash scripts/train_ablation.sh s1

# 只跑某个配置
bash scripts/train_ablation.sh s1 labram_fbcca

# S1 跑完后单独跑 S2
bash scripts/train_ablation.sh s2

# 单个配置完整流程
bash scripts/train_ablation.sh all reve_fbcca
```

输出目录：`output_ablation_{配置名}_{s1|s2}/`

### CLI 参数

```bash
python main_bci_agent.py \
    --stage 1|2 \
    --encoder_type reve|labram \     # 编码器选择
    --use_fbcca / --no_fbcca \       # FBCCA FiLM 开关
    --exclude_bad_subjects \          # 排除 BETA 低质量受试者
    --batch_size 64 \
    --epochs 10 \
    --deepspeed configs/ds_zero2.json
```

### 收敛判断

| eval_loss | 状态 |
|-----------|------|
| ~3.69 | 随机（ln40） |
| ~1.5-2.0 | 初步收敛 |
| ~0.3-0.5 | 90% 准确率，可进 Stage 2 |
| <0.3 | 检查是否过拟合 |

### 数据质量

`--exclude_bad_subjects` 过滤 FBCCA <30% 的 BETA 受试者（S11,S41,S55,S59,S64），
运行时 mask 过滤，不需要重新预处理 .pt 文件。

---

## 旧版脚本（仅供参考）

| 脚本 | 说明 |
|------|------|
| `train_e2e.sh` | E2E pipeline (REVE → Qwen-VL-8B) |
| `train_hybrid.sh` | Plan A: 自定义 decoder |
| `train_hybrid_qwen.sh` | Plan B: Qwen-0.6B |
| `preprocess_e2e.sh` | 预处理 .mat → .pt |
| `train_unsloth.sh` | Unsloth 4-bit 单卡训练 |
| `extract_embeddings.sh` | 提取 REVE embeddings |
| `probe_fbcca_beta.py` | BETA FBCCA linear probe |
| `ssvep_sensitivity.py` | 受试者 SSVEP 质量分析 |
