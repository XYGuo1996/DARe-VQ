# DARe-VQ: Scaling Large EMA Codebooks via Delayed Evidence Aggregation and Usage Distortion Guided Reallocation

Official implementation of **DARe-VQ** (**D**elayed Evidence **A**ggregation and **Re**allocation for **V**ector **Q**uantization).

DARe-VQ is a training-time codebook maintenance strategy for large **Exponential Moving Average (EMA)** vector quantizers. It addresses the severe codebook collapse that can occur when naively scaling EMA-updated codebooks, allowing enlarged codebooks to translate into usable representational capacity and improved speech reconstruction quality.

## Installation

Create the environment and install dependencies according to the provided environment file:

```bash
# Example
conda create -n darevq python=3.9
conda activate darevq

pip install -r requirements.txt
```

Please replace the commands above if your repository uses a different environment setup.


## Overview

Naively increasing the size of an EMA-updated codebook does not necessarily increase its effective capacity. In our experiments, scaling a standard EMA codebook from **4K to 16K entries** reduces codebook utilization to only **14.55%** and degrades reconstruction quality.

DARe-VQ addresses this problem with two components:

* **Delayed Multi-Batch Evidence**
  Codebook maintenance is performed only after accumulating assignment evidence over multiple batches, reducing unreliable inactive-code decisions caused by short-term assignment sparsity.

* **Usage Distortion Guided Reallocation**
  Inactive codebook entries are treated as reallocatable capacity. Active clusters are ranked according to their usage-weighted local distortion, and high-priority clusters are locally split to place additional codewords where they are most useful.

DARe-VQ modifies only the **EMA codebook maintenance procedure**. It introduces no additional learnable parameters or auxiliary training objective and leaves the encoder, decoder, quantization operation, training loss, and inference pipeline unchanged.

## Main Results

## Main Results

### Comparison with Standard EMA

We first compare DARe-VQ with standard EMA codebook maintenance under the same training setup. Naively scaling the EMA codebook from 4K to 16K causes severe codebook collapse, reducing utilization to about 14.5% and degrading reconstruction quality. DARe-VQ restores near-complete utilization and consistently outperforms both EMA-16K and EMA-4K across all reconstruction metrics.

| Method          | Evaluation Set          |    UTMOS ↑ |     PESQ ↑ |     STOI ↑ |       F1 ↑ | Util. (%)   |
| --------------- | ----------------------- | ---------: | ---------: | ---------: | ---------: | ----------: |
| EMA-4K          | LibriTTS test-clean     |     3.9872 |     2.3897 |     0.9151 |     0.9390 |      100.00 |
|                 | LibriTTS test-other     |     3.5032 |     2.1281 |     0.8828 |     0.9146 |      100.00 |
|                 | LJSpeech                |     3.8350 |     2.0070 |     0.9011 |     0.9169 |       99.88 |
| EMA-16K         | LibriTTS test-clean     |     3.8509 |     2.1583 |     0.8999 |     0.9316 |       14.55 |
|                 | LibriTTS test-other     |     3.3608 |     1.9435 |     0.8668 |     0.9059 |       14.55 |
|                 | LJSpeech                |     3.6851 |     1.8525 |     0.8871 |     0.9140 |       14.56 |
| **DARe-VQ-16K** | **LibriTTS test-clean** | **4.0358** | **2.4171** | **0.9197** | **0.9417** |       99.76 |
|                 | **LibriTTS test-other** | **3.5544** | **2.1440** | **0.8880** | **0.9174** |       99.76 |
|                 | **LJSpeech**            | **3.9340** | **2.0748** | **0.9092** | **0.9191** |       98.43 |

DARe-VQ-16K increases LibriTTS codebook utilization from **14.55% to 99.76%** and outperforms EMA-16K on every reported reconstruction metric. It also surpasses the healthy EMA-4K baseline, showing that the recovered entries provide useful representational capacity rather than merely increasing the number of active codes.

---

## Scaling to Large Codebooks

DARe-VQ remains effective when scaling the codebook from 16K to 131K entries. The 131K configuration retains near-complete utilization while further improving PESQ, STOI, and V/UV F1 across all evaluation sets.

| Method           | Evaluation Set          |    UTMOS ↑ |     PESQ ↑ |     STOI ↑ |       F1 ↑ | Util. (%)   |
| ---------------- | ----------------------- | ---------: | ---------: | ---------: | ---------: | ----------: |
| DARe-VQ-16K      | LibriTTS test-clean     | **4.0358** |     2.4171 |     0.9197 |     0.9417 |       99.76 |
|                  | LibriTTS test-other     | **3.5544** |     2.1440 |     0.8880 |     0.9174 |       99.76 |
|                  | LJSpeech                | **3.9340** |     2.0748 |     0.9092 |     0.9191 |       98.43 |
| **DARe-VQ-131K** | **LibriTTS test-clean** |     3.9912 | **2.5259** | **0.9246** | **0.9441** |       99.47 |
|                  | **LibriTTS test-other** |     3.4949 | **2.2347** | **0.8944** | **0.9204** |       99.47 |
|                  | **LJSpeech**            |     3.9202 | **2.1960** | **0.9150** | **0.9252** |       95.28 |

Increasing the codebook to **131,072 entries** retains **99.47% utilization on LibriTTS** and **95.28% on LJSpeech**. Compared with DARe-VQ-16K, the 131K model further improves **PESQ, STOI, and F1 on all evaluation sets**, with only a modest reduction in UTMOS.

---

## Component Ablation

We evaluate the contribution of the source-selection criterion and the reallocation strategy using a 16K codebook with (N=10).

* **DARe-VQ:** usage × distortion score + local cluster splitting.
* **Score: Additive:** replaces the multiplicative score with the sum of normalized usage and distortion.
* **Score: Usage only:** selects source clusters using only their usage.
* **Reallocation: Random reset:** retains delayed evidence but replaces local splitting with random replacement.

| Variant                    | Evaluation Set          |    UTMOS ↑ |     PESQ ↑ |     STOI ↑ |       F1 ↑ | Util. (%)   |
| -------------------------- | ----------------------- | ---------: | ---------: | ---------: | ---------: | ----------: |
| **DARe-VQ**                | **LibriTTS test-clean** |     4.0358 | **2.4171** | **0.9197** | **0.9417** |       99.76 |
|                            | **LibriTTS test-other** |     3.5544 | **2.1440** | **0.8880** | **0.9174** |       99.76 |
|                            | **LJSpeech**            |     3.9340 | **2.0748** | **0.9092** | **0.9191** |       98.43 |
| Score: Additive            | LibriTTS test-clean     |     3.9789 |     2.3768 |     0.9160 |     0.9396 |       99.74 |
|                            | LibriTTS test-other     |     3.5049 |     2.1101 |     0.8843 |     0.9152 |       99.74 |
|                            | LJSpeech                |     3.8535 |     2.0210 |     0.9047 |     0.9156 |   **98.58** |
| Score: Usage only          | LibriTTS test-clean     |     3.9762 |     2.2280 |     0.9079 |     0.9364 |   **99.96** |
|                            | LibriTTS test-other     |     3.4807 |     1.9684 |     0.8733 |     0.9109 |   **99.96** |
|                            | LJSpeech                |     3.8400 |     1.9520 |     0.8955 |     0.9181 |       96.53 |
| Reallocation: Random reset | LibriTTS test-clean     | **4.0488** |     2.3491 |     0.9133 |     0.9387 |       23.39 |
|                            | LibriTTS test-other     | **3.5543** |     2.0753 |     0.8808 |     0.9146 |       23.39 |
|                            | LJSpeech                | **3.9211** |     2.0268 |     0.9040 |     0.9153 |       23.39 |

The ablation results show that **high utilization alone is insufficient**. Usage-only and additive scoring recover nearly full utilization but consistently underperform the full usage-distortion product in reconstruction quality. Random reset activates only **23.39%** of the codebook, whereas local splitting raises utilization to nearly 100% while improving PESQ, STOI, and F1. These results highlight the importance of reallocating inactive capacity toward regions that are simultaneously **frequently used and poorly represented**.

## Acknowledgements

This implementation is built upon the **WavTokenizer** framework. We thank the authors of WavTokenizer and related open-source projects for making their code publicly available.
