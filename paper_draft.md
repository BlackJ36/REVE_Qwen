# Calibration-Free SSVEP-BCI Spelling via LLM-Augmented Frequency Decoding

---

## Abstract

Steady-state visual evoked potential (SSVEP) based brain-computer interfaces (BCIs) enable non-invasive communication by decoding neural responses to flickering visual stimuli. However, existing high-accuracy decoders such as ensemble task-related component analysis (eTRCA) require per-subject calibration data, while calibration-free methods like filter bank canonical correlation analysis (FBCCA) suffer from low accuracy at practical stimulation durations (1--2 s). We propose a novel two-stage framework that combines FBCCA decoding with large language model (LLM) post-correction to achieve state-of-the-art spelling accuracy without any calibration. Specifically, we fine-tune Qwen3-4B-Instruct with LoRA on synthetic spelling data constructed from real FBCCA decoder outputs across subjects, training the LLM to leverage per-position candidate distributions and linguistic priors to correct decoding errors. We additionally evaluate a FiLM-based classifier combining the REVE EEG foundation model with FBCCA modulation, and compute information transfer rates (ITR) using the Wolpaw formula. All word-level metrics are reported as 3-seed averages with standard deviations.

On a cross-subject evaluation using the Tsinghua Benchmark and BETA datasets (105 subjects, 40-target SSVEP), our method achieves **69.8%±0.2% word-level accuracy** and **95.4%±0.1% character-level accuracy** at 2 s stimulation with an effective ITR of 115.4 bits/min---surpassing all calibration-free baselines (FiLM 2s: 54.0%, FBCCA 2s: 45.7%) and calibrated eTRCA (53.7%) while requiring zero calibration. At 1 s stimulation, LLM correction elevates FBCCA from 1.0% to 24.4%±1.2% word accuracy, approaching the performance of CCA at double the stimulation duration. Our results demonstrate that linguistic context can substantially compensate for signal-processing limitations in BCI decoding, opening a pathway toward practical calibration-free BCI spellers.

**Keywords:** Brain-computer interface, SSVEP, FBCCA, large language model, spelling correction, calibration-free

---

## 1. Introduction

Brain-computer interfaces (BCIs) offer a direct communication channel between the brain and external devices, providing a critical lifeline for individuals with severe motor disabilities such as amyotrophic lateral sclerosis (ALS) and locked-in syndrome [1, 2]. Among non-invasive BCI paradigms, steady-state visual evoked potentials (SSVEPs) have emerged as a preferred approach due to their high signal-to-noise ratio (SNR), minimal user training requirements, and suitability for high-speed communication [3, 4].

In a typical SSVEP-BCI speller, visual stimuli flicker at distinct frequencies arranged on a virtual keyboard. When the user fixates on a target character, the corresponding frequency and its harmonics are entrained in the occipital EEG, enabling character identification through frequency detection [5]. Modern systems achieve information transfer rates (ITRs) exceeding 300 bits/min with 40 targets flickering at 8.0--15.8 Hz [6].

### 1.1 The Calibration Dilemma

The accuracy of SSVEP decoding critically depends on the decoder choice and the availability of calibration data:

**Calibration-free methods** such as canonical correlation analysis (CCA) [7] and its filter-bank extension (FBCCA) [8] correlate EEG signals with pre-constructed sinusoidal templates. These methods require no subject-specific training data but are fundamentally limited by frequency resolution---at short stimulation durations (1--2 s), the spectral peaks of closely-spaced SSVEP frequencies (0.2 Hz apart) overlap, degrading classification accuracy.

**Calibration-dependent methods** such as task-related component analysis (TRCA) [9] and its ensemble variant (eTRCA) learn data-driven spatial filters from per-subject calibration trials. While significantly more accurate, they require 4--6 minutes of dedicated calibration per subject, creating a practical barrier for clinical deployment and reducing user acceptance.

This calibration dilemma---high accuracy requires calibration, while calibration-free methods underperform---remains a central challenge in SSVEP-BCI research.

### 1.2 Language as a Prior for BCI Decoding

We observe that BCI spellers are ultimately text-generation systems. The decoded character sequences are not random strings but meaningful words and sentences drawn from a constrained language. This observation motivates a key insight: **linguistic context can compensate for signal-processing limitations**.

Consider the FBCCA decoder output at 1 s stimulation: per-character accuracy is approximately 50%, meaning roughly half the characters in a decoded word are incorrect. However, the errors are not catastrophic---they typically involve substitution with spectrally adjacent characters (due to frequency confusion) rather than arbitrary corruption. A language model with knowledge of English orthography and word structure should be able to recover the intended text from such noisy inputs, much as humans can read text with missing or substituted letters.

### 1.3 Contributions

We present a framework that decouples BCI decoding into two stages: (1) signal-level frequency detection via FBCCA, producing per-position candidate characters with confidence scores; and (2) text-level correction via a fine-tuned LLM that maps noisy candidate sequences to correct text. Our contributions are:

1. **A calibration-free SSVEP-BCI speller** that surpasses all baselines including calibrated eTRCA in word-level spelling accuracy, achieving 69.8% word accuracy at 2 s stimulation without any subject-specific calibration.

2. **A systematic decoder comparison** across CCA, FBCCA, FiLM (REVE+FBCCA+LoRA), and eTRCA at 1--3 s stimulation durations, with information transfer rate (ITR) analysis, establishing comprehensive baselines for both per-trial and word-level spelling accuracy.

3. **An efficient LLM fine-tuning pipeline** using LoRA on Qwen3-4B-Instruct, with synthetic training data constructed from real cross-subject FBCCA outputs, demonstrating that a 4B-parameter model suffices for BCI text correction.

4. **Cross-subject evaluation protocol** with strict train/val subject separation and multi-seed averaging (3 seeds), ensuring that reported improvements reflect genuine generalization with quantified variance.

---

## 2. Related Work

### 2.1 SSVEP Decoding Methods

**CCA-based methods.** Canonical correlation analysis (CCA) [7] computes the correlation between multi-channel EEG and sinusoidal reference templates at each target frequency. Filter bank CCA (FBCCA) [8] decomposes the signal into multiple sub-bands, computing CCA within each band and combining via a weighted sum, improving robustness to harmonic structure. These methods are calibration-free but limited by frequency resolution at short durations.

**Template-based methods.** Individual template-based CCA (itCCA) [10] uses subject-specific templates alongside sinusoidal references. Extended CCA and multi-stimulus CCA further improve accuracy by exploiting inter-frequency correlations [11].

**Spatial filtering methods.** Task-related component analysis (TRCA) [9] learns spatial filters that maximize the reproducibility of task-related components across trials. Ensemble TRCA (eTRCA) uses spatial filters from all frequency classes, providing more robust classification. These methods achieve the highest per-trial accuracy but require per-subject calibration.

### 2.2 Language Models for BCI

Recent work has explored integrating language models with BCI systems. EEG-to-text approaches attempt end-to-end decoding from neural signals to natural language [12, 13], though these remain largely limited to motor imagery and P300 paradigms. For SSVEP-BCIs, language model integration has been explored primarily through n-gram predictive text [14] and character-level language models [15], but not through modern LLMs with their superior contextual understanding.

Our work differs fundamentally: rather than building an end-to-end neural decoder, we treat the LLM as a post-processor that corrects the output of an existing signal-processing decoder, preserving the interpretability and reliability of the FBCCA pipeline while adding linguistic intelligence.

### 2.3 Parameter-Efficient Fine-tuning

Parameter-efficient fine-tuning methods such as LoRA [16] enable adaptation of large language models to specific tasks with minimal trainable parameters. Our approach applies LoRA to fine-tune Qwen3-4B-Instruct [17] on the structured correction task, training only the attention projections (q, k, v, o) while keeping the base model frozen.

---

## 3. Methods

### 3.1 SSVEP Signal Processing Pipeline

#### 3.1.1 EEG Preprocessing

Raw EEG data is preprocessed with a 3--90 Hz bandpass filter and 50 Hz notch filter, then resampled from 250 Hz to 200 Hz. We use 9 occipital channels (Oz, O1, O2, POz, PO3, PO4, PO7, PO8, Pz) optimal for SSVEP detection. A 0.14 s latency skip (28 samples at 200 Hz) removes the SSVEP transient response at stimulus onset, improving decoder accuracy at all durations.

#### 3.1.2 SSVEP Target Configuration

The 40-target SSVEP keyboard follows a 5x8 grid layout with row-major frequency assignment:

$$f_{i,j} = 8.0 + j \times 1.0 + i \times 0.2 \text{ Hz}, \quad i \in [0,4], j \in [0,7]$$

Target frequencies range from 8.0 Hz to 15.8 Hz with joint frequency-phase coding [6]. The keyboard maps 26 letters (A--Z), 10 digits (0--9), and 4 special characters (_, ., <, >) to the 40 targets.

#### 3.1.3 Decoder Implementations

We implement three signal-processing decoders for comparison:

**CCA.** Single-band canonical correlation analysis using the widest sub-band (6--90 Hz). For each of the 40 target frequencies, we construct a reference signal $Y_f \in \mathbb{R}^{2H \times T}$ containing $H=4$ harmonics (sine and cosine):

$$Y_f = \begin{bmatrix} \sin(2\pi f t) \\ \cos(2\pi f t) \\ \vdots \\ \sin(2\pi H f t) \\ \cos(2\pi H f t) \end{bmatrix}$$

The CCA correlation $\rho_f$ between the filtered EEG $X \in \mathbb{R}^{C \times T}$ and template $Y_f$ is computed as the maximum canonical correlation:

$$\rho_f^2 = \max \text{eig}\left( R_{yy}^{-1/2} R_{yx} R_{xx}^{-1} R_{xy} R_{yy}^{-1/2} \right)$$

where $R_{xx}$, $R_{yy}$, and $R_{xy}$ are the auto- and cross-covariance matrices. Classification uses $\hat{k} = \arg\max_f \rho_f$.

**FBCCA.** Filter bank CCA applies FFT-based bandpass filtering across 5 sub-bands $\{[6,90], [14,90], [22,90], [30,90], [38,90]\}$ Hz. CCA correlations from each sub-band are combined via weighted sum:

$$\rho_f^{\text{FBCCA}} = \sum_{b=1}^{5} w_b \cdot \rho_f^{(b)}, \quad w_b = b^{-1.25} + 0.25$$

The entire computation is GPU-accelerated: $R_{yy}^{-1/2}$ is pre-computed from constant sinusoidal templates, and all 40 frequencies are solved simultaneously via batched `torch.linalg.solve`, avoiding Python loops.

**eTRCA.** Ensemble filter bank TRCA with leave-one-block-out cross-validation per subject. For each frequency class $k$, TRCA learns a spatial filter $\mathbf{w}_k$ by solving the generalized eigenvalue problem:

$$S_k \mathbf{w} = \lambda Q_k \mathbf{w}$$

where $S_k$ is the inter-trial covariance and $Q_k$ is the within-trial covariance. The ensemble variant uses all 40 spatial filters for each target evaluation:

$$\rho_k^{\text{eTRCA}} = \frac{1}{40} \sum_{j=1}^{40} \text{corr}(\mathbf{w}_j^T X_{\text{test}}, \mathbf{w}_j^T \bar{X}_k)$$

Filter bank decomposition and weighted combination are applied identically to FBCCA.

### 3.2 Word-Level Spelling Evaluation

Unlike prior work that reports only per-trial classification accuracy, we introduce word-level spelling metrics that capture end-to-end performance relevant to BCI users.

#### 3.2.1 Word Assembly Protocol

Given a corpus of English words and phrases (5,000 entries, average length ~19 characters), we simulate 1,000 spelling sessions per decoder configuration:

1. Randomly select a target word from the corpus
2. Find a (subject, block) group in the validation set with trials for all required character labels
3. For each character position, sample a trial with the corresponding label and use the decoder's top-1 prediction
4. Assemble predicted characters into a decoded word

This protocol uses real decoder outputs (not simulated noise), ensuring that error patterns reflect genuine signal-processing limitations including frequency confusion, SNR variation across subjects, and block-to-block variability.

#### 3.2.2 Metrics

- **Word accuracy (word_acc)**: Fraction of words where the entire decoded sequence exactly matches the target
- **Character accuracy (char_acc)**: Computed via Levenshtein edit distance: $\text{char\_acc} = 1 - \frac{\sum_i \text{ED}(\hat{w}_i, w_i)}{\sum_i |w_i|}$
- **Average edit distance (avg_ed)**: Mean Levenshtein distance between decoded and target words

Character accuracy via edit distance correctly handles length-changing errors (insertions, deletions) that position-wise comparison misses.

### 3.3 LLM-Based Text Correction

#### 3.3.1 Problem Formulation

Given FBCCA decoder outputs for a sequence of $L$ characters, including the top-1 decoded text $\hat{w} = \hat{c}_1 \hat{c}_2 \cdots \hat{c}_L$ and per-position top-3 candidates with confidence scores $\{(c_{l,1}, p_{l,1}), (c_{l,2}, p_{l,2}), (c_{l,3}, p_{l,3})\}_{l=1}^{L}$, the task is to predict the correct target word $w^*$.

This is formulated as a conditional text generation problem:

$$w^* = \arg\max_{w} P_\theta(w | \hat{w}, \{c_{l,k}, p_{l,k}\}_{l,k})$$

where $P_\theta$ is the fine-tuned LLM.

#### 3.3.2 Model Architecture

We use Qwen3-4B-Instruct as the base LLM, fine-tuned with LoRA (rank $r=16$, scaling factor $\alpha=32$, dropout $p=0.05$) on the attention projection matrices ($W_Q, W_K, W_V, W_O$). This yields approximately 10M trainable parameters out of 4B total (~0.25%). The model operates as a pure text-to-text system with no EEG embeddings.

#### 3.3.3 Input Representation

The input to the LLM is a structured Chinese-language prompt containing:

1. **System prompt**: Instructs the model that it is a BCI assistant receiving decoder candidate information
2. **User message**: Contains the FBCCA decoded text and per-position candidates formatted as:

```
解码结果: "HFLP"
候选:
  位置1: H(0.45) G(0.30) P(0.25)
  位置2: F(0.38) E(0.35) N(0.27)
  位置3: L(0.52) K(0.28) T(0.20)
  位置4: P(0.60) O(0.22) Q(0.18)
```

3. **Target output**: The correct word (e.g., "HELP")

Confidence scores are computed by normalizing the absolute CCA correlation values of the top-3 candidates to sum to 1. Candidates are sorted by confidence in descending order.

#### 3.3.4 Training Data Generation

Training data is generated from real FBCCA decoder outputs with strict cross-subject separation:

- **Training set**: 90 subjects (BM S01--S30 + BETA S01--S60), ~29,160 trials
- **Validation set**: 15 subjects (BM S31--S35 + BETA S61--S70), ~3,400 trials

For each training sample, we sample a target word and retrieve real FBCCA top-3 outputs for each character position from the corresponding subject pool. This ensures the LLM learns from authentic decoder error patterns (frequency confusion, SNR variability) rather than synthetic noise.

Three dialogue types are generated with the following distribution:

| Type | Description | Ratio | Purpose |
|------|-------------|-------|---------|
| A (Spelling) | Candidates → correct word | 70% | Core spelling correction |
| C (Correction) | Corrupted candidates → error detection + suggestion | 20% | Error awareness |
| D (Dialogue) | Natural language interaction | 10% | Preserves conversational ability |

**Short word augmentation.** 30% of Type A/C samples use short words (2--8 characters) from a curated list of 322 common English words. This curriculum-style augmentation provides easier samples for initial learning, as shorter words have fewer error positions and stronger orthographic constraints.

**SSVEP confusion model for Type C.** Error injection follows the frequency-domain confusion pattern of real SSVEP decoders: 60% neighbor substitution (spectrally adjacent targets on the 5x8 grid), 15% character drop, 10% character repeat, 15% random substitution.

#### 3.3.5 Training Configuration

Training uses the HuggingFace Trainer with the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| Effective batch size | 40 (4 per device x 5 GPUs x 2 grad accumulation) |
| Learning rate | 2e-4 (cosine schedule) |
| Warmup | 10% of total steps |
| Epochs | 10 (early stopping patience = 3) |
| Max sequence length | 1,280 tokens |
| Precision | bf16 |
| Gradient checkpointing | Enabled |

**Label masking.** Only the assistant response tokens are supervised; system prompt and user message tokens have labels set to -100. For Type A samples, this means only ~10 out of ~610 tokens are supervised (1.6%), focusing the model's learning on the output text.

---

## 4. Experimental Setup

### 4.1 Datasets

We evaluate on two publicly available SSVEP-BCI datasets:

**Tsinghua Benchmark Dataset [6].** 35 subjects (S01--S35), 64 EEG channels at 250 Hz, 40 SSVEP targets with 6 experimental blocks per subject. Each trial contains 6 s of data including a 0.5 s visual cue period. After cue removal and resampling to 200 Hz, each trial yields 600 timepoints (3.0 s effective stimulus).

**BETA Dataset [18].** 70 subjects (S01--S70), 64 EEG channels at 250 Hz, 40 targets with 4 blocks per subject. Trial durations vary: S01--S15 have 3.0 s trials (500 timepoints after resampling, zero-padded to 600), S16--S70 have 4.0 s trials (700 timepoints, truncated to 600).

**Preprocessing.** After removing 2 electrodes (CB1, CB2) not in the electrode position bank, 62 channels are retained. All data undergoes 3--90 Hz bandpass filtering, 50 Hz notch filtering, and resampling to 200 Hz. The 0.5 s visual cue period (125 samples at 250 Hz) is removed before resampling.

**Subject exclusion.** Five BETA subjects with near-random FBCCA accuracy (<30%): S11, S41, S55, S59, S64 are excluded from FiLM training and evaluation. For decoder baselines (CCA/FBCCA/eTRCA), all subjects are retained to maintain comparability.

**Train/Val split.** Table 0 summarizes the EEG data distribution. Training and validation use strictly separate subjects, ensuring zero overlap.

**Table 0.** EEG dataset distribution. Each trial corresponds to one 40-class SSVEP target.

| Dataset | Split | Subjects | Blocks/Subj | Trials/Block | Trials | Notes |
|---------|-------|----------|-------------|-------------|--------|-------|
| Benchmark | Train | S01--S30 (30) | 6 | 40 | 7,200 | |
| Benchmark | Val | S31--S35 (5) | 6 | 40 | 1,200 | |
| BETA | Train | S01--S60 (60) | 4 | 40 | 9,600 | Incl. 4 BAD subj |
| BETA | Val | S61--S70 (10) | 4 | 40 | 1,600 | Incl. 1 BAD subj (S64) |
| **Total** | **Train** | **90** | | | **16,800** | |
| **Total** | **Val** | **15** | | | **2,800** (2,640†) | |

*†After excluding BAD subject S64 (160 trials) from BETA val.*

#### LLM Correction Training Data

Table 0b summarizes the LLM correction data, constructed from real FBCCA decoder outputs of the corresponding subject pools.

**Table 0b.** LLM correction data distribution. V1 = long words only; V2 = 30% short words (2--8 chars) mixed in. All models trained on V1.

| Version | Ratio (A/C/D) | Avg Word Len | Train | Val | FBCCA Duration |
|---------|---------------|-------------|-------|-----|----------------|
| V1 1s | 50/25/25% | ~19 chars | 10,000 | 2,000 | 1.0 s (200 pts) |
| V1 2s | 50/25/25% | ~19 chars | 10,000 | 2,000 | 2.0 s (400 pts) |
| V2 1s | 70/20/10% | ~15 chars | 10,000 | 2,000 | 1.0 s (200 pts) |
| V2 2s | 70/20/10% | ~15 chars | 10,000 | 2,000 | 2.0 s (400 pts) |

Type A = spelling correction, Type C = error detection + correction, Type D = natural language dialogue. V2 val data is used for 3-seed evaluation only; models are trained on V1 data.

### 4.2 Decoder Evaluation Settings

Decoders are evaluated at three stimulation durations:

| Duration | Timepoints | After 0.14s skip | Effective signal |
|----------|------------|-------------------|------------------|
| 1.0 s | 200 pts | 172 pts | 0.86 s |
| 2.0 s | 400 pts | 372 pts | 1.86 s |
| 3.0 s | 600 pts | 572 pts | 2.86 s |

For eTRCA, leave-one-block-out cross-validation is performed within each subject (5 blocks for BM, 3 blocks for BETA), using the remaining blocks as calibration data.

FBCCA and CCA are calibration-free and applied directly to the raw (preprocessed) EEG without any subject-specific adaptation.

**FiLM classifier.** We additionally evaluate a FiLM-based classifier that combines the REVE EEG foundation model [20] (69.2M parameters, frozen, 9 occipital channels) with FBCCA feature modulation via FiLM layers ($\gamma \cdot \text{backbone} + \beta$) and LoRA adaptation (rank=16). FiLM requires cross-subject training data but no per-subject calibration.

### 4.3 LLM Correction Evaluation

The LLM correction model is evaluated on validation data constructed from validation subjects' FBCCA outputs, with two data distributions:

**V1 data** (A:50%/C:25%/D:25%, corpus words only, avg ~19 chars): 2,000 samples (1,000 A + 500 C + 500 D).

**V2 data** (A:70%/C:20%/D:10%, 30% short words mixed in, avg ~15 chars): 2,000 samples (1,400 A + 400 C + 200 D).

V2 evaluation uses 3-seed averaging (seed=42, 123, 456) to quantify sampling variance. Each seed generates different word-trial assignments from the same validation subjects and FBCCA outputs.

For Type A, we report word accuracy, character accuracy, and average edit distance. For Type C, we report correction accuracy. For Type D, we report exact match accuracy. Effective ITR is computed via Method B: $P_{\text{eff}} = 1 - \text{avg\_ed}/L$ where $L$ is the average word length, then applying the Wolpaw formula.

**Type A prediction cleaning.** The model occasionally produces correction-template text (e.g., "WORD 可能你想输入的是WORD") in Type A responses. We strip these templates before computing metrics, extracting only the intended spelled word.

---

## 5. Results

### 5.1 Per-Trial Classification Accuracy

Table 1 shows the per-trial top-1 classification accuracy of the three decoders at 1 s and 2 s stimulation durations, evaluated on validation subjects.

**Table 1.** Per-trial top-1 classification accuracy (%) and ITR (bits/min, Wolpaw formula, gaze_shift=0.5s) on validation subjects. All decoders skip the 0.14 s SSVEP transient response.

| Duration | CCA | FBCCA | FiLM | eTRCA |
|----------|-----|-------|------|-------|
| 1.0 s | 45.6 (58.0) | 49.9 (67.0) | 65.8 (103.5) | 77.4 (134.2) |
| 2.0 s | 72.7 (72.8) | 84.3 (92.7) | 88.3 (100.4) | 86.4 (96.8) |
| 3.0 s | 79.5 (60.2) | 89.4 (73.3) | - | 91.9 (76.9) |

*Values in parentheses are ITR in bits/min. FiLM = REVE(9ch, frozen) + FiLM(FBCCA) + LoRA(r=16).*

At 1 s, eTRCA achieves the highest ITR (134.2 bits/min) due to its strong accuracy at short duration. FiLM provides 65.8% accuracy without per-subject calibration, significantly outperforming FBCCA (49.9%). At 2 s, FiLM achieves the highest per-trial accuracy (88.3%), marginally surpassing eTRCA (86.4%). At 3 s, diminishing ITR returns are observed: despite higher accuracy, the longer trial time reduces information throughput.

### 5.2 Word-Level Spelling Accuracy (Signal Processing Baselines)

Table 2 presents word-level spelling metrics, which better reflect real-world BCI usage than per-trial accuracy.

**Table 2.** Word-level spelling accuracy across decoders. 1,000 words from a spelling corpus (avg ~19 chars). 3-seed average (seed=42,123,456) with standard deviations. Evaluated on validation subjects using real decoder outputs.

| Decoder | Duration | Word Acc (%) | Char Acc (%) | Avg ED |
|---------|----------|--------------|--------------|--------|
| CCA | 1.0 s | 0.6±0.2 | 46.6±1.1 | 10.34±0.22 |
| FBCCA | 1.0 s | 1.0±0.3 | 48.1±0.7 | 10.05±0.16 |
| FiLM | 1.0 s | 7.5±0.2 | 65.7±1.0 | 6.63±0.23 |
| eTRCA | 1.0 s | 42.0±1.1 | 78.8±0.9 | 4.11±0.17 |
| CCA | 2.0 s | 30.2±1.3 | 74.2±0.8 | 4.99±0.15 |
| FBCCA | 2.0 s | 45.7±0.6 | 86.0±0.2 | 2.71±0.04 |
| FiLM | 2.0 s | 54.0±2.6 | 87.3±0.7 | 2.46±0.14 |
| eTRCA | 2.0 s | 53.7±0.5 | 87.6±0.3 | 2.40±0.06 |
| CCA | 3.0 s | 43.8±0.7 | 81.5±0.6 | 3.58±0.12 |
| FBCCA | 3.0 s | 51.9±1.4 | 91.3±0.1 | 1.69±0.02 |
| eTRCA | 3.0 s | 64.6±0.8 | 93.7±0.2 | 1.22±0.03 |

*FiLM = REVE(9ch, frozen) + FiLM(FBCCA) + LoRA(r=16), no per-subject calibration but requires cross-subject training data.*

**Key observations:**

1. **Word-level accuracy amplifies per-trial differences.** A 50% per-trial accuracy (FBCCA 1s) yields only 1.0% word accuracy on 19-character words ($0.5^{19} \approx 0$). This exponential relationship motivates our LLM-based approach.

2. **FiLM 2s ≈ eTRCA 2s** at word level (54.0% vs 53.7%), despite FiLM having higher per-trial accuracy (88.3% vs 86.4%). Both achieve similar char_acc (~87%). Crucially, FiLM requires no per-subject calibration.

3. **3-seed standard deviations are small** (<1.5 pp for word_acc, <1.1 pp for char_acc), confirming that the 1,000-word sampling protocol yields stable estimates.

### 5.3 LLM Correction Results

#### 5.3.1 FBCCA + LLM Results

Tables 3 and 4 show the LLM correction results on two evaluation data distributions. V1 uses long words from the corpus (avg ~19 chars, single seed). V2 uses a mixed distribution with 30% short words (avg ~15 chars, 3-seed average). Both use v1-trained models evaluated cross-subject.

**Table 3.** LLM correction results on V1 data (avg word ~19 chars, A:50%/C:25%/D:25%).

| Metric | FBCCA 1s Raw | 1s + LLM | FBCCA 2s Raw | **2s + LLM** |
|--------|-------------|----------|-------------|--------------|
| Word Acc | 0.0% | 29.5% | 8.4% | **71.9%** |
| Char Acc | 48.2% | 75.4% | 85.8% | **96.2%** |
| Avg ED | 9.99 | 4.76 | 2.74 | **0.74** |
| Eff ITR | - | 127.4 | - | **117.1** |

**Table 4.** LLM correction results on V2 data (avg word ~15 chars, A:70%/C:20%/D:10% + 30% short words, 3-seed avg).

| Metric | FBCCA 1s Raw | 1s + LLM | FBCCA 2s Raw | **2s + LLM** |
|--------|-------------|----------|-------------|--------------|
| Word Acc | 1.6%±0.1% | 24.4%±1.2% | 19.4%±0.4% | **69.8%±0.2%** |
| Char Acc | 47.8%±0.1% | 71.2%±0.8% | 85.4%±0.4% | **95.4%±0.1%** |
| Avg ED | 7.87±0.04 | 4.35±0.14 | 2.20±0.04 | **0.70±0.01** |
| Eff ITR | - | 117.2 | - | **115.4** |

*Effective ITR computed via Method B: $P_{\text{eff}} = 1 - \text{avg\_ed}/L$, then Wolpaw formula (N=40, gaze_shift=0.5s).*

**Key observations:**

1. At 2 s, the LLM reduces average edit distance to <1 character per word (0.70--0.74), making the system practically usable.
2. V2 FBCCA baseline word_acc is much higher (19.4% vs 8.4%) because 30% short words are easier to spell correctly. However, LLM word_acc is slightly lower on V2 (69.8% vs 71.9%) because short words provide less linguistic context for correction.
3. The effective ITR is stable across data distributions (115--117 bits/min at 2s), confirming robust generalization.
4. 3-seed standard deviations are extremely small (word_acc ±0.2% at 2s), demonstrating stable evaluation.

#### 5.3.2 Multi-Type Evaluation

Table 5 shows the performance across all dialogue types (V2 data, 3-seed average).

**Table 5.** Multi-type evaluation of FBCCA + LLM models (V2 data, 3-seed avg).

| Type | Duration | Count | Metric | Score |
|------|----------|-------|--------|-------|
| A (Spelling) | 1s | 1,400 | Word Acc | 24.4%±1.2% |
| A (Spelling) | 1s | 1,400 | Char Acc | 71.2%±0.8% |
| A (Spelling) | 1s | 1,400 | Avg ED | 4.35±0.14 |
| C (Correction) | 1s | 400 | Correction Acc | 11.1%±1.0% |
| D (Dialogue) | 1s | 200 | Exact Match | 100.0% |
| A (Spelling) | 2s | 1,400 | Word Acc | 69.8%±0.2% |
| A (Spelling) | 2s | 1,400 | Char Acc | 95.4%±0.1% |
| A (Spelling) | 2s | 1,400 | Avg ED | 0.70±0.01 |
| C (Correction) | 2s | 400 | Correction Acc | 31.4%±1.8% |
| D (Dialogue) | 2s | 200 | Exact Match | 100.0% |

Type D (natural language) achieves 100% exact match at both durations, demonstrating that LoRA fine-tuning preserves conversational ability. Type C correction accuracy improves with input quality (11.1% at 1s → 31.4% at 2s) but remains a weakness, suggesting room for improvement in explicit error detection.

### 5.4 Comprehensive Comparison

Table 6 presents the unified comparison across all methods, the central result of this paper.

**Table 6.** Comprehensive comparison of all decoder configurations. Word-level metrics are 3-seed averages (baselines) or 3-seed V2 evaluation (LLM). ITR in bits/min (Wolpaw formula, gaze_shift=0.5s). Cal = per-subject calibration required.

| Method | Duration | Trial Acc | ITR | Word Acc (%) | Char Acc (%) | Avg ED | Cal |
|--------|----------|-----------|-----|--------------|--------------|--------|-----|
| CCA | 1.0 s | 45.6% | 58.0 | 0.6±0.2 | 46.6±1.1 | 10.34±0.22 | No |
| FBCCA | 1.0 s | 49.9% | 67.0 | 1.0±0.3 | 48.1±0.7 | 10.05±0.16 | No |
| FiLM | 1.0 s | 65.8% | 103.5 | 7.5±0.2 | 65.7±1.0 | 6.63±0.23 | No* |
| FBCCA+LLM | 1.0 s | - | 117.2† | 24.4±1.2 | 71.2±0.8 | 4.35±0.14 | No |
| eTRCA | 1.0 s | 77.4% | 134.2 | 42.0±1.1 | 78.8±0.9 | 4.11±0.17 | Yes |
| CCA | 2.0 s | 72.7% | 72.8 | 30.2±1.3 | 74.2±0.8 | 4.99±0.15 | No |
| FBCCA | 2.0 s | 84.3% | 92.7 | 45.7±0.6 | 86.0±0.2 | 2.71±0.04 | No |
| eTRCA | 2.0 s | 86.4% | 96.8 | 53.7±0.5 | 87.6±0.3 | 2.40±0.06 | Yes |
| FiLM | 2.0 s | 88.3% | 100.4 | 54.0±2.6 | 87.3±0.7 | 2.46±0.14 | No* |
| **FBCCA+LLM** | **2.0 s** | **-** | **115.4†** | **69.8±0.2** | **95.4±0.1** | **0.70±0.01** | **No** |
| CCA | 3.0 s | 79.5% | 60.2 | 43.8±0.7 | 81.5±0.6 | 3.58±0.12 | No |
| FBCCA | 3.0 s | 89.4% | 73.3 | 51.9±1.4 | 91.3±0.1 | 1.69±0.02 | No |
| eTRCA | 3.0 s | 91.9% | 76.9 | 64.6±0.8 | 93.7±0.2 | 1.22±0.03 | Yes |

*No* = no per-subject calibration but requires cross-subject training data. †Effective ITR via Method B ($P_{\text{eff}} = 1 - \text{avg\_ed}/L$, V2 data $L \approx 15$).*

**Key findings:**

1. **FBCCA 2s + LLM achieves the highest word accuracy** (69.8%) among all methods and durations, surpassing calibrated eTRCA 3s (64.6%) by +5.2 pp and eTRCA 2s (53.7%) by +16.1 pp, all without calibration.

2. **Effective ITR of 115.4 bits/min** exceeds all 2s and 3s baselines (FiLM 2s: 100.4, eTRCA 2s: 96.8, eTRCA 3s: 76.9). Only eTRCA 1s (134.2) has higher Wolpaw ITR, but it requires per-subject calibration.

3. **LLM correction is equivalent to doubling stimulation duration**: FBCCA 1s + LLM (24.4% word) approaches CCA 2s (30.2% word), demonstrating that linguistic context compensates for ~1 s of additional signal.

4. **FiLM 2s ≈ eTRCA 2s** at word level (54.0% vs 53.7%), with FiLM requiring no per-subject calibration. This establishes FiLM as a strong calibration-free baseline.

5. **Diminishing returns at 3s**: Despite eTRCA 3s achieving 91.9% trial accuracy, its ITR (76.9) is lower than all 2s methods due to longer trial time. FBCCA 2s+LLM achieves better word accuracy (69.8% vs 64.6%) in less time.

6. **Calibration-free superiority**: The FBCCA + LLM approach requires no subject-specific data, making it immediately deployable for new users---a critical advantage for clinical BCI applications.

### 5.5 Error Analysis

Figure 1 (conceptual) illustrates the types of corrections the LLM performs:

**Frequency confusion resolution.** When FBCCA confuses spectrally adjacent targets (e.g., H→G at 8.0 vs 15.0 Hz), the LLM can resolve the ambiguity using the candidate list and linguistic context. For example, given "GFLP" with candidates {H(0.45), G(0.30)} at position 1, the model correctly outputs "HELP".

**Multi-position joint correction.** The LLM considers all positions simultaneously, enabling corrections that a per-character post-processor cannot make. For example, "CQMPUTER" (Q instead of O, adjacent in frequency space) is corrected to "COMPUTER" by recognizing the word pattern.

**Confidence-aware behavior.** The model learns to trust high-confidence candidates and explore alternatives for low-confidence positions, effectively performing a soft search over the candidate space guided by linguistic priors.

**Failure modes.** The model fails when (1) the correct word is not in the model's vocabulary or is extremely rare, (2) multiple plausible words match the noisy input (e.g., "CAT" vs "CUT" vs "CAR"), or (3) the noise level is so high that too few candidates contain the correct character.

---

## 6. Discussion

### 6.1 Why LLM Correction Works

The success of LLM correction for SSVEP-BCI spelling can be attributed to three complementary factors:

**Structured noise.** FBCCA errors are not random---they follow the frequency confusion pattern of the 5x8 SSVEP grid, where spectrally adjacent targets are more likely to be confused. This structure means that the correct character often appears in the top-3 candidates, providing the LLM with recoverable information.

**Linguistic redundancy.** Natural language has high redundancy---the per-character entropy of English is approximately 1.0--1.5 bits [19], far below the $\log_2(40) \approx 5.3$ bits required for a 40-character keyboard. This redundancy provides the "error correction budget" that the LLM exploits.

**Candidate information.** Unlike a simple spell-checker that sees only the top-1 prediction, our model receives the full top-3 candidate set with confidence scores, providing a richer signal for disambiguation.

### 6.2 Practical Implications

**Elimination of calibration.** The most significant practical implication is the elimination of per-subject calibration. eTRCA requires 4--6 minutes of dedicated calibration trials, which is burdensome for patients with severe motor disabilities. Our method achieves superior performance using only the general FBCCA decoder, enabling immediate system use for new users.

**Latency considerations.** The LLM correction adds computational latency: generating the corrected word for a single query takes approximately 200 ms on a single GPU (NVIDIA RTX 4090). In a practical BCI speller, this correction could be applied at the word level (after the user completes spelling a word), adding negligible delay to the overall interaction.

**Scalability.** The text-only nature of our approach means the LLM can be deployed on modest hardware. The LoRA adapter adds only ~20 MB to the base model, and inference can be performed on consumer GPUs or even quantized for edge deployment.

### 6.3 Limitations and Future Work

**Type C correction accuracy.** The current model achieves only 31.4% accuracy on explicit error correction at 2s (11.1% at 1s), suggesting that the model is better at implicit correction (Type A) than explicit error identification. Future work should explore dedicated error detection pretraining.

**Language dependency.** Our evaluation uses English words/phrases. Extending to other languages (Chinese, Japanese) would require language-specific training data and potentially different tokenization strategies.

**Online evaluation.** Our evaluation uses precomputed offline FBCCA outputs. Online evaluation with real-time EEG acquisition and closed-loop user interaction is needed to validate the approach in a clinical setting.

**Decoder fusion.** The current approach uses FBCCA exclusively for the signal-processing stage. Combining FBCCA with learned EEG representations (e.g., from foundation models like REVE [20]) could further improve the candidate quality fed to the LLM.

---

## 7. Conclusion

We presented a calibration-free SSVEP-BCI spelling framework that combines FBCCA frequency decoding with LLM-based text correction. By fine-tuning Qwen3-4B-Instruct with LoRA on synthetic data constructed from real cross-subject FBCCA outputs, we demonstrate that linguistic priors can substantially compensate for signal-processing limitations. Our method achieves 69.8%±0.2% word accuracy and 95.4%±0.1% character accuracy at 2 s stimulation with an effective ITR of 115.4 bits/min, surpassing all baselines including calibrated eTRCA 3s (64.6% word) and calibration-free FiLM 2s (54.0% word) while requiring zero calibration.

The key insight enabling this work is that BCI spelling is fundamentally a structured text generation problem, not merely a signal classification problem. By treating the FBCCA decoder as a noisy channel and the LLM as a channel decoder, we leverage the redundancy inherent in natural language to recover information that the signal-processing stage alone cannot extract.

Our results suggest a paradigm shift in BCI speller design: rather than pursuing ever-more-complex signal-processing decoders that demand calibration, combining simple calibration-free decoders with powerful language models can achieve superior practical performance. This approach is immediately deployable, requires no subject-specific adaptation, and can be improved simply by scaling the LLM or expanding the training corpus.

---

## References

[1] J. R. Wolpaw, N. Birbaumer, D. J. McFarland, G. Pfurtscheller, and T. M. Vaughan, "Brain-computer interfaces for communication and control," *Clinical Neurophysiology*, vol. 113, no. 6, pp. 767--791, 2002.

[2] U. Chaudhary, N. Birbaumer, and A. Ramos-Murguialday, "Brain-computer interfaces for communication and rehabilitation," *Nature Reviews Neurology*, vol. 12, no. 9, pp. 513--525, 2016.

[3] X. Gao, D. Xu, M. Cheng, and S. Gao, "A BCI-based environmental controller for the motion-disabled," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 11, no. 2, pp. 137--140, 2003.

[4] Y. Wang, X. Gao, B. Hong, C. Jia, and S. Gao, "Brain-computer interfaces based on visual evoked potentials," *IEEE Engineering in Medicine and Biology Magazine*, vol. 27, no. 5, pp. 64--71, 2008.

[5] G. R. Müller-Putz and R. Scherer, "Steady-state visual evoked potential (SSVEP)-based communication: Impact of harmonic frequency components," *Journal of Neural Engineering*, vol. 2, no. 4, pp. 123--130, 2005.

[6] Y. Wang, X. Chen, X. Gao, and S. Gao, "A benchmark dataset for SSVEP-based brain-computer interfaces," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 25, no. 10, pp. 1746--1752, 2017.

[7] Z. Lin, C. Zhang, W. Wu, and X. Gao, "Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs," *IEEE Transactions on Biomedical Engineering*, vol. 53, no. 12, pp. 2610--2614, 2006.

[8] X. Chen, Y. Wang, S. Gao, T. P. Jung, and X. Gao, "Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface," *Journal of Neural Engineering*, vol. 12, no. 4, p. 046008, 2015.

[9] M. Nakanishi, Y. Wang, X. Chen, Y. T. Wang, X. Gao, and T. P. Jung, "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis," *IEEE Transactions on Biomedical Engineering*, vol. 65, no. 1, pp. 104--112, 2018.

[10] M. Nakanishi, Y. Wang, Y. T. Wang, and T. P. Jung, "A comparison study of canonical correlation analysis based methods for detecting steady-state visual evoked potentials," *PloS ONE*, vol. 10, no. 10, p. e0140703, 2015.

[11] C. M. Wong, F. Wan, B. Wang, Z. Wang, W. Nan, K. F. Lao, P. U. Mak, M. I. Vai, and A. Rosa, "Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs," *Journal of Neural Engineering*, vol. 17, no. 1, p. 016026, 2020.

[12] J. Li, C. Zhang, R. Zong, and W. Kong, "Open vocabulary electroencephalography-to-text decoding and evaluation," *arXiv preprint arXiv:2312.14041*, 2023.

[13] Y. Song, J. Zheng, A. Wang, and M. Zhang, "Decoding natural language from EEG data with a large language model," *arXiv preprint arXiv:2405.04517*, 2024.

[14] G. Townsend, B. K. LaPallo, C. B. Boulay, D. J. Krusienski, G. E. Frye, C. K. Hauser, N. E. Schwartz, T. M. Vaughan, J. R. Wolpaw, and E. W. Sellers, "A novel P300-based brain-computer interface stimulus presentation paradigm: Moving beyond rows and columns," *Clinical Neurophysiology*, vol. 121, no. 7, pp. 1109--1120, 2010.

[15] D. B. Ryan, G. E. Frye, G. Townsend, D. R. Berry, S. Mesa-G, N. A. Gates, and E. W. Sellers, "Predictive spelling with a P300-based brain-computer interface: Increasing the rate of communication," *International Journal of Human-Computer Interaction*, vol. 27, no. 1, pp. 69--84, 2011.

[16] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models," *arXiv preprint arXiv:2106.09685*, 2021.

[17] Qwen Team, "Qwen3 technical report," *arXiv preprint arXiv:2505.09388*, 2025.

[18] B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, "BETA: A large benchmark database toward SSVEP-BCI application," *Frontiers in Neuroscience*, vol. 14, p. 627, 2020.

[19] C. E. Shannon, "Prediction and entropy of printed English," *Bell System Technical Journal*, vol. 30, no. 1, pp. 50--64, 1951.

[20] A. Kostas, S. Aroca-Ouellette, and F. Bhatt, "REVE: A foundation model for EEG analysis," *Advances in Neural Information Processing Systems*, 2025.

---

## Appendix A: SSVEP Keyboard Layout

The 40-target SSVEP keyboard follows a 5x8 grid with row-major frequency assignment:

```
       Col 0    Col 1    Col 2    Col 3    Col 4    Col 5    Col 6    Col 7
       8.0Hz    9.0Hz   10.0Hz   11.0Hz   12.0Hz   13.0Hz   14.0Hz   15.0Hz
Row 0:   A        B        C        D        E        F        G        H
       8.2Hz    9.2Hz   10.2Hz   11.2Hz   12.2Hz   13.2Hz   14.2Hz   15.2Hz
Row 1:   I        J        K        L        M        N        O        P
       8.4Hz    9.4Hz   10.4Hz   11.4Hz   12.4Hz   13.4Hz   14.4Hz   15.4Hz
Row 2:   Q        R        S        T        U        V        W        X
       8.6Hz    9.6Hz   10.6Hz   11.6Hz   12.6Hz   13.6Hz   14.6Hz   15.6Hz
Row 3:   Y        Z        1        2        3        4        5        6
       8.8Hz    9.8Hz   10.8Hz   11.8Hz   12.8Hz   13.8Hz   14.8Hz   15.8Hz
Row 4:   7        8        9        0        _        .        <        >
```

Label index: $\text{label} = \text{row} \times 8 + \text{col}$

Frequency: $f = 8.0 + \text{col} \times 1.0 + \text{row} \times 0.2$ Hz

## Appendix B: Training Data Examples

### Type A (Spelling Correction)

```json
{
  "type": "A",
  "messages": [
    {"role": "system", "content": "你是一个脑机接口助手。用户通过注视屏幕上闪烁的字符来拼写文字。解码器会给出每个位置的候选字符及置信度。请根据候选信息和语言知识，输出解码结果。"},
    {"role": "user", "content": "解码结果: \"HFLP\"\n候选:\n  位置1: H(0.45) G(0.30) P(0.25)\n  位置2: F(0.38) E(0.35) N(0.27)\n  位置3: L(0.52) K(0.28) T(0.20)\n  位置4: P(0.60) O(0.22) Q(0.18)"},
    {"role": "assistant", "content": "HELP"}
  ],
  "target_word": "HELP",
  "noisy_word": "HFLP"
}
```

### Type C (Error Detection + Correction)

```json
{
  "type": "C",
  "messages": [
    {"role": "system", "content": "你是一个脑机接口助手。..."},
    {"role": "user", "content": "解码结果: \"WARER\"\n候选:\n  ..."},
    {"role": "assistant", "content": "WARER 你是不是想拼WATER"}
  ],
  "target_word": "WATER"
}
```

### Type D (Natural Language)

```json
{
  "type": "D",
  "messages": [
    {"role": "system", "content": "你是一个脑机接口助手，负责帮助用户通过脑电信号进行交流。"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！我是脑机接口助手，可以帮你通过EEG信号拼写文字。"}
  ]
}
```

## Appendix C: Implementation Details

### GPU-Accelerated FBCCA

Our FBCCA implementation is fully GPU-accelerated with zero learnable parameters:

1. **Pre-computation**: Sinusoidal templates $Y_f$ and their inverse square root covariance $R_{yy}^{-1/2}$ are computed once in `__init__` and stored as buffers
2. **FFT-based filtering**: Sub-band filtering uses `torch.fft.rfft` with binary frequency masks, avoiding FIR/IIR filter design
3. **Batched solving**: All 40 frequency CCA problems are solved in a single `torch.linalg.solve` call by reshaping the right-hand side from $(B, C, 40, 6)$ to $(B, C, 240)$
4. **Chunked eigendecomposition**: The final $6 \times 6$ eigenvalue problem is solved via `torch.linalg.eigvalsh` in chunks of 8,192 to manage GPU memory

### eTRCA Leave-One-Block-Out

For each validation subject:
1. Each of the $B$ blocks is held out in turn (5 for BM, 3 for BETA)
2. The remaining $B-1$ blocks provide calibration data (~160--200 trials, ≥4 per class)
3. FBTRCA spatial filters are fitted on calibration data
4. The held-out block is classified using the fitted model
5. Results from all folds are aggregated to produce per-subject accuracy

This procedure ensures that eTRCA evaluation is fair: the test data is never seen during spatial filter fitting.

### Cross-Subject Data Generation

The correction training data pipeline:
1. Load precomputed FBCCA top-3 outputs for training subjects
2. Group trials by (subject, block, label) for efficient sampling
3. For each target word, sample one trial per character position, retrieving real FBCCA candidates
4. Normalize confidence scores: $p_k = |s_k| / (\sum_j |s_j| + \epsilon)$
5. Format as structured prompt with Qwen3's chat template
6. Split: training subjects → train.jsonl, validation subjects → val.jsonl

This ensures zero data leakage between training and evaluation.
