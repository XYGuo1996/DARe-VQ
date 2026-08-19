# DARe-VQ: Scaling Large EMA Codebooks via Delayed Evidence Aggregation and Usage Distortion Guided Reallocation

Official implementation of **DARe-VQ** (**D**elayed Evidence **A**ggregation and **Re**allocation for **V**ector **Q**uantization).

DARe-VQ is a training-time codebook maintenance strategy for large **Exponential Moving Average (EMA)** vector quantizers. It addresses the severe codebook collapse that can occur when naively scaling EMA-updated codebooks, allowing enlarged codebooks to translate into usable representational capacity and improved speech reconstruction quality.

## Overview

Naively increasing the size of an EMA-updated codebook does not necessarily increase its effective capacity. In our experiments, scaling a standard EMA codebook from **4K to 16K entries** reduces codebook utilization to only **14.55%** and degrades reconstruction quality.

DARe-VQ addresses this problem with two components:

* **Delayed Multi-Batch Evidence**
  Codebook maintenance is performed only after accumulating assignment evidence over multiple batches, reducing unreliable inactive-code decisions caused by short-term assignment sparsity.

* **Usage Distortion Guided Reallocation**
  Inactive codebook entries are treated as reallocatable capacity. Active clusters are ranked according to their usage-weighted local distortion, and high-priority clusters are locally split to place additional codewords where they are most useful.

DARe-VQ modifies only the **EMA codebook maintenance procedure**. It introduces no additional learnable parameters or auxiliary training objective and leaves the encoder, decoder, quantization operation, training loss, and inference pipeline unchanged.

## Main Results

Speech reconstruction experiments are conducted using **LibriTTS** for training and **LibriTTS test-clean/test-other** and **LJSpeech** for evaluation.

| Method      | Codebook Size | LibriTTS Utilization | LJSpeech Utilization |
| ----------- | ------------: | -------------------: | -------------------: |
| EMA         |            4K |              100.00% |               99.88% |
| EMA         |           16K |               14.55% |               14.56% |
| **DARe-VQ** |       **16K** |           **99.76%** |           **98.43%** |
| **DARe-VQ** |      **131K** |           **99.47%** |           **95.28%** |

At **16K entries**, DARe-VQ consistently outperforms both the reproduced EMA-4K baseline and naive EMA-16K scaling in reconstruction quality.

Scaling DARe-VQ to **131K entries** retains near-complete codebook utilization and further improves **PESQ, STOI, and V/UV F1** across all evaluation sets.


## Evidence Window

The evidence window (N) plays an important role in stabilizing codebook maintenance.

| (N) | LibriTTS Utilization | Observation                         |
| --: | -------------------: | ----------------------------------- |
|   1 |                0.02% | Unstable reallocation               |
|   5 |               99.95% | Utilization largely recovered       |
|  10 |               99.76% | Best overall reconstruction quality |
|  20 |               99.93% | Stable but less responsive          |

With (N=1), single-batch evidence can misclassify code activity, causing newly split codes to be repeatedly reclaimed before sufficient support is accumulated. A moderate evidence window stabilizes activity estimation while preserving timely reallocation.


## Installation

Create the environment and install dependencies according to the provided environment file:

```bash
# Example
conda create -n darevq python=3.9
conda activate darevq

pip install -r requirements.txt
```

Please replace the commands above if your repository uses a different environment setup.


## Acknowledgements

This implementation is built upon the **WavTokenizer** framework. We thank the authors of WavTokenizer and related open-source projects for making their code publicly available.
