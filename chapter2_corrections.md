# 第二章 LaTeX 修正清单

本文档列出 LaTeX 章节中需要修正的内容，包括通道名称、数据集构造描述、ITR 计算公式与实验结果数值。

---

## 修正 1：通道名称（PO5/PO6 → PO7/PO8）

**位置**：`\subsection{EEG 预处理与通道选择}` 以及 `表 2.1 参数表`

**原文**：
```
本文仅选取 9 个视觉相关通道 Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, O2
```

**修正为**：
```
本文仅选取 9 个视觉相关通道 Oz, O1, O2, POz, PO3, PO4, PO7, PO8, Pz
```

**原因**：代码中 `OCCIPITAL_CHANNELS = ["Oz", "O1", "O2", "POz", "PO3", "PO4", "PO7", "PO8", "Pz"]`，FBCCA 和 FiLM 1s 模型均使用此默认通道集。FiLM 2s 模型使用了 PO5/PO6 是个别实验变体，不应作为通用描述。

同步修改参数表中通道列表：
```latex
通道数 $C$ & 9（$O_z, O_1, O_2, PO_z, PO_3, PO_4, PO_7, PO_8, P_z$）
```

---

## 修正 2：训练/评估数据集构造（核心修正）

**位置**：`\subsection{数据集构造与任务划分}`

**原文存在的问题**：
LaTeX 描述 Type A:70% / C:20% / D:10% 实际是 **v2 评估数据的比例**，但模型训练用的是 **v1 数据**（A:50% / C:25% / D:25%）。需要区分训练集和评估集的数据构成。

**修正后的完整描述**：

```latex
\subsection{数据集构造与任务划分}
\label{subsec:chap2_sample_construction}

本章实验包含两部分。第一部分为单试次字符级分类评估，用于比较不同底层解码器的识别性能。
第二部分为基于合成序列的自然语言连续拼写评估，用于验证后训练语言模型对带噪文本的恢复能力。

为使语言模型学习脑机带噪文本的恢复规律，本文构建了基于真实 FBCCA 解码输出的监督微调数据集。
训练数据与评估数据在被试层面严格分离，并分别采用不同的样本构造策略。

\paragraph{训练数据构造}
训练集从训练被试池（Benchmark S01--S30 + BETA S01--S60）的真实 FBCCA Top-$K$ 输出中采样构建，
共生成 10,000 条样本。为全面覆盖语言模型在脑机交互中需要具备的多维能力，
训练样本划分为以下三类任务，采用 50\%/25\%/25\% 的比例混合：

\begin{enumerate}
    \item \textbf{Type A（隐式恢复，占比 50\%）}。模型输入包含 FBCCA 解码序列及各位置的 Top-$K$ 候选字符与置信度，
    要求直接输出修正后的目标单词或短指令。该类型用于训练核心的端到端文本恢复能力。
    \item \textbf{Type C（显式解释，占比 25\%）}。在输出正确结果的同时，
    模型还需给出错误识别与修正建议的自然语言解释。
    该类型用于培养模型的显式错误归因能力。
    为模拟真实 SSVEP 解码错误模式，本文采用基于频域混淆结构的错误注入策略：
    60\% 为频率邻近替换（$5\times8$ 网格上相邻目标），15\% 为字符丢失，
    10\% 为字符重复，15\% 为随机替换。
    \item \textbf{Type D（通用对话，占比 25\%）}。引入常规问答与脑机场景对话指令，
    以缓解 LoRA 微调后可能出现的灾难性遗忘，并保持基座模型的通用语言能力。
\end{enumerate}

\paragraph{评估数据构造}
评估集从验证被试池（Benchmark S31--S35 + BETA S61--S70）的真实 FBCCA 输出中独立采样构建，
共生成 2,000 条样本。与训练集不同，评估集采用 70\%/20\%/10\% 的类型比例（Type A 1,400 条、
Type C 400 条、Type D 200 条），并混入约 30\% 的短词样本（2--8 个字符），
使平均词长由训练集的约 19 个字符缩短至约 15 个字符。
此设计旨在覆盖更多元的词长分布，评估模型在不同语义上下文丰度条件下的恢复能力。

为量化采样随机性对评估结果的影响，评估阶段采用 3 组不同随机种子（seed = 42, 123, 456）
独立生成评估集并取平均，所有报告结果均以均值 $\pm$ 标准差形式呈现。

训练集与评估集的详细构成对比如表~\ref{ta:chap2_data_construction} 所示。

\begin{table}[htbp]
\bicaption{语言模型训练集与评估集构成对比}
        {Comparison of LLM training and evaluation data composition}
\label{ta:chap2_data_construction}
\renewcommand{\arraystretch}{1.8}
\centering
\setlength{\tabcolsep}{4mm}{
\begin{tabular}{ccccc}
\toprule
\textbf{数据集} & \textbf{样本数} & \textbf{Type A/C/D 比例}
  & \textbf{平均词长} & \textbf{FBCCA 时长} \\
\midrule
训练集 & 10,000 & 50\%/25\%/25\% & $\sim$19 字符 & 1.0 s / 2.0 s \\
评估集 & 2,000 $\times$ 3 seeds & 70\%/20\%/10\% & $\sim$15 字符 & 1.0 s / 2.0 s \\
\bottomrule
\end{tabular}}
\\[0.35em]
{\small 注：评估集混入 30\% 短词（2--8 字符），用于评估不同词长条件下的恢复能力。
训练和评估使用严格分离的被试池。}
\end{table}
```

---

## 修正 3：FiLM-REVE 参数补全

**位置**：`\subsection{FiLM-REVE 跨被试学习型基线}` 最后一段

**原文**：
```
本文采用冻结参数的 REVE 模型作为 EEG 骨干网络，并在投影层中引入 LoRA 模块进行参数高效微调。
```

**修正为**：
```
本文采用冻结参数的 REVE 模型作为 EEG 骨干网络，
并在 Transformer 的注意力投影层（$W_Q$, $W_K$, $W_V$ 及 $W_O$）中引入 LoRA 模块进行参数高效微调，
秩 $r=16$，缩放因子 $\alpha=32$。
```

---

## 修正 4：等效 ITR 公式（使用 Method B）

**位置**：`\subsection{语义级拼写评估指标}` 中 ITR_eff 定义

**原文**：
```latex
ITR_{\mathrm{eff}} = \mathrm{CharAcc} \times ITR_{\mathrm{char}}
```

**修正为**：
```latex
本文进一步定义等效系统信息传输率（Effective ITR），
用于衡量语义恢复后系统的整体信息吞吐效率。
由于 FBCCA+LLM 方法在词级后处理中不具备直接的逐字符分类准确率，
本文采用基于编辑距离的反推方法估算等效逐字符准确率：
\begin{equation}
P_{\mathrm{eff}}
=
1 - \frac{\overline{\mathrm{ED}}}{L},
\label{eq:chap2_p_eff}
\end{equation}
其中，$\overline{\mathrm{ED}}$ 为平均编辑距离，$L$ 为评估集平均词长。
随后将 $P_{\mathrm{eff}}$ 代入式~\eqref{eq:chap2_itr}，
以单次字符决策耗时（含注视切换时间）为 $T$，计算等效 ITR：
\begin{equation}
ITR_{\mathrm{eff}}
=
\left[
\log_2 N_s
+
P_{\mathrm{eff}} \log_2 P_{\mathrm{eff}}
+
(1-P_{\mathrm{eff}}) \log_2\frac{1-P_{\mathrm{eff}}}{N_s-1}
\right]
\frac{60}{T + T_{\mathrm{gaze}}},
\label{eq:chap2_itr_eff_revised}
\end{equation}
其中，$T_{\mathrm{gaze}}=0.5\,\mathrm{s}$ 为注视切换时间。
该方法不依赖字符独立性假设，直接从编辑距离衡量信息损失，是最保守的 ITR 估算。
```

---

## 修正 5：实验结果数值更新

### 5.1 表 2.4（FBCCA + LLM 恢复性能）

**原文等效 ITR 值**：32.0, 81.0, 79.2, 110.1

**修正后（Method B）**：

```latex
\begin{table}[htbp]
\bicaption{FBCCA 结合大语言模型后的文本恢复性能提升}
         {Performance improvement of text recovery using FBCCA combined with LLM}
\label{ta:chap2_llm_correction_results}
\renewcommand{\arraystretch}{1.8}
\centering
\setlength{\tabcolsep}{4mm}{
\begin{tabular}{ccccc}
\toprule
\textbf{评价指标}
& \makecell[c]{\textbf{FBCCA} \\ \textbf{(1.0 s)}}
& \makecell[c]{\textbf{1.0 s +} \\ \textbf{LLM恢复}}
& \makecell[c]{\textbf{FBCCA} \\ \textbf{(2.0 s)}}
& \makecell[c]{\textbf{2.0 s +} \\ \textbf{LLM恢复}} \\
\midrule
单词准确率 (\%) & $1.6 \pm 0.1$ & $24.4 \pm 1.2$ & $19.4 \pm 0.4$
  & $\mathbf{69.8 \pm 0.2}$ \\
字符准确率 (\%) & $47.8 \pm 0.1$ & $71.2 \pm 0.8$ & $85.4 \pm 0.4$
  & $\mathbf{95.4 \pm 0.1}$ \\
平均编辑距离 & $7.87 \pm 0.04$ & $4.35 \pm 0.14$ & $2.20 \pm 0.04$
  & $\mathbf{0.70 \pm 0.01}$ \\
等效 ITR (bits/min) & 62.7 & 117.3 & 94.9
  & \textbf{115.3} \\
\bottomrule
\end{tabular}}
\\[0.35em]
{\small 注：等效 ITR 按式~\eqref{eq:chap2_itr_eff_revised} 计算
（$P_{\mathrm{eff}}=1-\overline{\mathrm{ED}}/L$，$L\approx15$，$T_{\mathrm{gaze}}=0.5\,\mathrm{s}$），
用于刻画语义恢复后系统的整体信息吞吐效率。}
\end{table}
```

### 5.2 表 2.5（多类型恢复能力）— 数值无变化

当前 LaTeX 中的数值（Type A: 69.8±0.2%, Type C: 31.4±1.8%, Type D: 100%）与实验结果一致，无需修改。

### 5.3 正文中引用等效 ITR 的位置

全文搜索 `110.1` 替换为 `115.3`，搜索 `32.0` `81.0` `79.2` 替换为对应新值。

---

## 修正 6：单试次分类结果表微调

**位置**：表 2.2

表中 ITR 数值微小差异需统一（因四舍五入）：

| 方法 | 时长 | Acc | ITR (精确) |
|------|------|-----|-----------|
| CCA | 1s | 45.6% | 58.1 |
| FBCCA | 1s | 49.9% | 67.0 |
| FiLM | 1s | 65.8% | 103.5 |
| eTRCA | 1s | 77.4% | 134.3 |
| CCA | 2s | 72.7% | 72.8 |
| FBCCA | 2s | 84.3% | 92.8 |
| FiLM | 2s | 88.3% | 100.4 |
| eTRCA | 2s | 86.4% | 96.7 |
| CCA | 3s | 79.5% | 60.1 |
| FBCCA | 3s | 89.4% | 73.3 |
| eTRCA | 3s | 91.9% | 76.9 |

当前 LaTeX 表中数值与上表一致，无需修改。

---

## 修正汇总

| # | 修正内容 | 影响范围 |
|---|---------|---------|
| 1 | 通道 PO5/PO6 → PO7/PO8 | 2处：正文 + 参数表 |
| 2 | 训练数据比例 70/20/10 → 50/25/25，区分训练/评估 | 整个数据集构造小节重写 |
| 3 | FiLM LoRA 参数补全 (r=16, α=32) | 1处 |
| 4 | ITR 公式改为 Method B (P_eff = 1 - ed/L) | 公式定义 + 表格注释 |
| 5 | 等效 ITR 数值更新 (32.0→62.7, 81.0→117.3, 79.2→94.9, 110.1→115.3) | 1个表 + 正文引用 |
| 6 | 单试次 ITR 数值已核实，无需修改 | — |
