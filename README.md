# DARe-VQ: Scaling Large EMA Codebooks via Delayed Evidence Aggregation and Usage Distortion Guided Reallocation

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DARe--VQ-yellow)](https://huggingface.co/XiaoyongGuo/DARe-VQ)

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

### Comparison with Standard EMA

We first compare DARe-VQ with standard EMA codebook maintenance under the same training setup. Naively scaling the EMA codebook from 4K to 16K causes severe codebook collapse, reducing utilization to about 14.5% and degrading reconstruction quality. DARe-VQ restores near-complete utilization and consistently outperforms both EMA-16K and EMA-4K across all reconstruction metrics.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Evaluation Set</th>
      <th>UTMOS ↑</th>
      <th>PESQ ↑</th>
      <th>STOI ↑</th>
      <th>F1 ↑</th>
      <th>Util. (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">EMA-4K</td>
      <td>LibriTTS test-clean</td>
      <td>3.9872</td>
      <td>2.3897</td>
      <td>0.9151</td>
      <td>0.9390</td>
      <td rowspan="2">100.00</td>
    </tr>
    <tr>
      <td>LibriTTS test-other</td>
      <td>3.5032</td>
      <td>2.1281</td>
      <td>0.8828</td>
      <td>0.9146</td>
    </tr>
    <tr>
      <td>LJSpeech</td>
      <td>3.8350</td>
      <td>2.0070</td>
      <td>0.9011</td>
      <td>0.9169</td>
      <td>99.88</td>
    </tr>
    <tr>
      <td rowspan="3">EMA-16K</td>
      <td>LibriTTS test-clean</td>
      <td>3.8509</td>
      <td>2.1583</td>
      <td>0.8999</td>
      <td>0.9316</td>
      <td rowspan="2">14.55</td>
    </tr>
    <tr>
      <td>LibriTTS test-other</td>
      <td>3.3608</td>
      <td>1.9435</td>
      <td>0.8668</td>
      <td>0.9059</td>
    </tr>
    <tr>
      <td>LJSpeech</td>
      <td>3.6851</td>
      <td>1.8525</td>
      <td>0.8871</td>
      <td>0.9140</td>
      <td>14.56</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>DARe-VQ-16K</strong></td>
      <td><strong>LibriTTS test-clean</strong></td>
      <td><strong>4.0358</strong></td>
      <td><strong>2.4171</strong></td>
      <td><strong>0.9197</strong></td>
      <td><strong>0.9417</strong></td>
      <td rowspan="2">99.76</td>
    </tr>
    <tr>
      <td><strong>LibriTTS test-other</strong></td>
      <td><strong>3.5544</strong></td>
      <td><strong>2.1440</strong></td>
      <td><strong>0.8880</strong></td>
      <td><strong>0.9174</strong></td>
    </tr>
    <tr>
      <td><strong>LJSpeech</strong></td>
      <td><strong>3.9340</strong></td>
      <td><strong>2.0748</strong></td>
      <td><strong>0.9092</strong></td>
      <td><strong>0.9191</strong></td>
      <td>98.43</td>
    </tr>
  </tbody>
</table>

DARe-VQ-16K increases LibriTTS codebook utilization from **14.55% to 99.76%** and outperforms EMA-16K on every reported reconstruction metric. It also surpasses the healthy EMA-4K baseline, showing that the recovered entries provide useful representational capacity rather than merely increasing the number of active codes.

---

## Scaling to Large Codebooks

DARe-VQ remains effective when scaling the codebook from 16K to 131K entries. The 131K configuration retains near-complete utilization while further improving PESQ, STOI, and V/UV F1 across all evaluation sets.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Evaluation Set</th>
      <th>UTMOS ↑</th>
      <th>PESQ ↑</th>
      <th>STOI ↑</th>
      <th>F1 ↑</th>
      <th>Util. (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">DARe-VQ-16K</td>
      <td>LibriTTS test-clean</td>
      <td><strong>4.0358</strong></td>
      <td>2.4171</td>
      <td>0.9197</td>
      <td>0.9417</td>
      <td rowspan="2">99.76</td>
    </tr>
    <tr>
      <td>LibriTTS test-other</td>
      <td><strong>3.5544</strong></td>
      <td>2.1440</td>
      <td>0.8880</td>
      <td>0.9174</td>
    </tr>
    <tr>
      <td>LJSpeech</td>
      <td><strong>3.9340</strong></td>
      <td>2.0748</td>
      <td>0.9092</td>
      <td>0.9191</td>
      <td>98.43</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>DARe-VQ-131K</strong></td>
      <td><strong>LibriTTS test-clean</strong></td>
      <td>3.9912</td>
      <td><strong>2.5259</strong></td>
      <td><strong>0.9246</strong></td>
      <td><strong>0.9441</strong></td>
      <td rowspan="2">99.47</td>
    </tr>
    <tr>
      <td><strong>LibriTTS test-other</strong></td>
      <td>3.4949</td>
      <td><strong>2.2347</strong></td>
      <td><strong>0.8944</strong></td>
      <td><strong>0.9204</strong></td>
    </tr>
    <tr>
      <td><strong>LJSpeech</strong></td>
      <td>3.9202</td>
      <td><strong>2.1960</strong></td>
      <td><strong>0.9150</strong></td>
      <td><strong>0.9252</strong></td>
      <td>95.28</td>
    </tr>
  </tbody>
</table>

Increasing the codebook to **131,072 entries** retains **99.47% utilization on LibriTTS** and **95.28% on LJSpeech**. Compared with DARe-VQ-16K, the 131K model further improves **PESQ, STOI, and F1 on all evaluation sets**, with only a modest reduction in UTMOS.

---

## Component Ablation

We evaluate the contribution of the source-selection criterion and the reallocation strategy using a 16K codebook with (N=10).

* **DARe-VQ:** usage × distortion score + local cluster splitting.
* **Score: Additive:** replaces the multiplicative score with the sum of normalized usage and distortion.
* **Score: Usage only:** selects source clusters using only their usage.
* **Reallocation: Random reset:** retains delayed evidence but replaces local splitting with random replacement.

<table>
  <thead>
    <tr>
      <th>Variant</th>
      <th>Evaluation Set</th>
      <th>UTMOS ↑</th>
      <th>PESQ ↑</th>
      <th>STOI ↑</th>
      <th>F1 ↑</th>
      <th>Util. (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><strong>DARe-VQ</strong></td>
      <td>LibriTTS test-clean</td>
      <td>4.0358</td>
      <td><strong>2.4171</strong></td>
      <td><strong>0.9197</strong></td>
      <td><strong>0.9417</strong></td>
      <td rowspan="2">99.76</td>
    </tr>
    <tr>
      <td>LibriTTS test-other</td>
      <td><strong>3.5544</strong></td>
      <td><strong>2.1440</strong></td>
      <td><strong>0.8880</strong></td>
      <td><strong>0.9174</strong></td>
    </tr>
    <tr>
      <td>LJSpeech</td>
      <td><strong>3.9340</strong></td>
      <td><strong>2.0748</strong></td>
      <td><strong>0.9092</strong></td>
      <td><strong>0.9191</strong></td>
      <td>98.43</td>
    </tr>
    <tr>
      <td rowspan="3">Score: Additive</td>
      <td>LibriTTS test-clean</td>
      <td>3.9789</td>
      <td>2.3768</td>
      <td>0.9160</td>
      <td>0.9396</td>
      <td rowspan="2">99.74</td>
    </tr>
    <tr>
      <td>LibriTTS test-other</td>
      <td>3.5049</td>
      <td>2.1101</td>
      <td>0.8843</td>
      <td>0.9152</td>
    </tr>
    <tr>
      <td>LJSpeech</td>
      <td>3.8535</td>
      <td>2.0210</td>
      <td>0.9047</td>
      <td>0.9156</td>
      <td>98.58</td>
    </tr>
    <tr>
      <td rowspan="3">Score: Usage only</td>
      <td>LibriTTS test-clean</td>
      <td>3.9762</td>
      <td>2.2280</td>
      <td>0.9079</td>
      <td>0.9364</td>
      <td rowspan="2">99.96</td>
    </tr>
    <tr>
      <td>LibriTTS test-other</td>
      <td>3.4807</td>
      <td>1.9684</td>
      <td>0.8733</td>
      <td>0.9109</td>
    </tr>
    <tr>
      <td>LJSpeech</td>
      <td>3.8400</td>
      <td>1.9520</td>
      <td>0.8955</td>
      <td>0.9181</td>
      <td>96.53</td>
    </tr>
    <tr>
      <td rowspan="3">Reallocation: Random reset</td>
      <td>LibriTTS test-clean</td>
      <td><strong>4.0488</strong></td>
      <td>2.3491</td>
      <td>0.9133</td>
      <td>0.9387</td>
      <td rowspan="2">23.39</td>
    </tr>
    <tr>
      <td>LibriTTS test-other</td>
      <td>3.5543</td>
      <td>2.0753</td>
      <td>0.8808</td>
      <td>0.9146</td>
    </tr>
    <tr>
      <td>LJSpeech</td>
      <td>3.9211</td>
      <td>2.0268</td>
      <td>0.9040</td>
      <td>0.9153</td>
      <td>23.39</td>
    </tr>
  </tbody>
</table>

The ablation results show that **high utilization alone is insufficient**. Usage-only and additive scoring recover nearly full utilization but consistently underperform the full usage-distortion product in reconstruction quality. Random reset activates only **23.39%** of the codebook, whereas local splitting raises utilization to nearly 100% while improving PESQ, STOI, and F1. These results highlight the importance of reallocating inactive capacity toward regions that are simultaneously **frequently used and poorly represented**.

## Acknowledgements

This implementation is built upon the **WavTokenizer** framework. We thank the authors of WavTokenizer and related open-source projects for making their code publicly available.
