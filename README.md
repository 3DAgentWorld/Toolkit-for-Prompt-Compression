<div align=center>
<img src="https://github.com/3DAgentWorld/Toolkit-for-Prompt-Compression/blob/main/imgs/logo_trans.png" width="600" height="150">

[![GitHub license](https://img.shields.io/github/license/3DAgentWorld/Toolkit-for-Prompt-Compression?color=blue)](https://github.com/3DAgentWorld/Toolkit-for-Prompt-Compression/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/3DAgentWorld/Toolkit-for-Prompt-Compression)](https://github.com/3DAgentWorld/Toolkit-for-Prompt-Compression)

---

</div>

# PCToolkit: A Unified Plug-and-Play Prompt Compression Toolkit of Large Language Models

## 🎉News

## 📄Techical Report

You can find more details about PCToolkit in our <a href=''>technical report</a>.

## Contents
- [Introduction](#introduction)
	- [Relevant Repositories](#relevant-repositories)
  	- [Key Features of PCToolkit](#key-features-of-pctoolkit)
  	- [Outlines](#outlines)
- [How to start](#how-to-start)
	- [Downloading models](#downloading-models)
- [How to use](#how-to-use)
- [Reference](#reference)
- [Citation](#citation)

## Introduction

Prompt compression is an innovative method for efficiently condensing input prompts while preserving essential information. To facilitate quick-start services, user-friendly interfaces, and compatibility with common datasets and metrics, we present the Prompt Compression Toolkit (PCToolkit). This toolkit is a unified plug-and-play solution for compressing prompts in Large Language Models (LLMs), featuring cutting-edge prompt compressors, diverse datasets, and metrics for comprehensive performance evaluation. PCToolkit boasts a modular design, allowing for easy integration of new datasets and metrics through portable and user-friendly interfaces. In this paper, we outline the key components and functionalities of PCToolkit.

We conducted evaluations of the compressors within PCToolkit across various natural language tasks, including reconstruction, summarization, mathematical problem-solving, question answering, few-shot learning, synthetic tasks, code completion, boolean expressions, multiple choice questions, and lies recognition.

PCToolkit contains:

- 5 compression methods
- 11 datasets
- 5+ metrics

<div align=center>
<img src="https://github.com/3DAgentWorld/Toolkit-for-Prompt-Compression/blob/main/imgs/architecture.png" width="739" height="380.5">
</div>

### Relevant Repositories

- <a href='https://github.com/liyucheng09/selective_context'>Selective Context</a> ![Github stars](https://img.shields.io/github/stars/liyucheng09/selective_context.svg)
- <a href='https://github.com/microsoft/LLMLingua'>LLMLingua</a></a> ![Github stars](https://img.shields.io/github/stars/microsoft/LLMLingua.svg)
- <a href='https://github.com/microsoft/LLMLingua'>LongLLMLingua</a> ![Github stars](https://img.shields.io/github/stars/microsoft/LLMLingua.svg)
- <a href='https://github.com/complementizer/rl-sentence-compression'>SCRL</a> ![Github stars](https://img.shields.io/github/stars/complementizer/rl-sentence-compression.svg)
- <a href='https://github.com/tingofurro/keep_it_simple'>KiS</a> ![Github stars](https://img.shields.io/github/stars/tingofurro/keep_it_simple.svg)


### Key Features of PCToolkit

(i) **State-of-the-art and reproducible methods.** Encompassing a wide array of mainstream compression techniques, PCToolkit offers a unified interface for various compression methods (compressors). Notably, PCToolkit incorporates a total of five distinct compressors, namely <a href='https://arxiv.org/abs/2310.06201'>Selective Context</a>, <a href='https://arxiv.org/abs/2310.05736'>LLMLingua</a>, <a href='https://arxiv.org/abs/2310.06839'>LongLLMLingua</a>, <a href='https://arxiv.org/abs/2205.08221'>SCRL</a> and <a href='https://arxiv.org/abs/2107.03444'>Keep it Simple</a>.

(ii) **User-friendly interfaces for new compressors, datasets, and metrics.** Facilitating portability and ease of adaptation to different environments, the interfaces within PCToolkit are designed to be easily customizable. This flexibility makes PCToolkit suitable for a wide range of environments and tasks.

(iii) **Modular design.** Featuring a modular structure that simplifies the transition between different methods, datasets, and metrics, PCToolkit is organized into four distinct modules: Compressor, Dataset, Metric and Runner module.

### Outlines

The following table presents an overview of the supported tasks, compressors, and datasets within PCToolkit. Each component are described in detail in our technical report.

| Tasks                 | Supported Compressors                            | Supported Datasets                                   |
|-----------------------|--------------------------------------------------|------------------------------------------------------|
| Reconstruction        | SC, LLMLingua, LongLLMLingua, SCRL, KiS          | BBC, ShareGPT, Arxiv, GSM8K                         |
| Mathematical problems | SC, LLMLingua, LongLLMLingua, SCRL, KiS          | GSM8K, BBH                                          |
| Boolean expressions   | SC, LLMLingua, LongLLMLingua, SCRL, KiS          | BBH                                                 |
| Multiple choice       | SC, LLMLingua, LongLLMLingua, SCRL, KiS          | BBH                                                 |
| Lies recognition      | SC, LLMLingua, LongLLMLingua, SCRL, KiS          | BBH                                                 |
| Summarization         | SC, LLMLingua, LongLLMLingua, SCRL, KiS          | BBC, Arxiv, Gigaword, DUC2004, BNC, Broadcast, Google |
|                       | LLMLingua, LongLLMLingua                         | LongBench                                           |
| Question and Answer   | SC, LLMLingua, LongLLMLingua, SCRL, KiS          | BBH                                                 |
|                       | LLMLingua, LongLLMLingua                         | LongBench                                           |
| Few-shot learning     | LLMLingua, LongLLMLingua                         | LongBench                                           |
| Synthetic tasks       | LLMLingua, LongLLMLingua                         | LongBench                                           |
| Code completion       | LLMLingua, LongLLMLingua                         | LongBench                                           |


## How to start

```shell
git clone https://github.com/3DAgentWorld/Toolkit-for-Prompt-Compression.git
```

```shell
cd Toolkit-for-Prompt-Compression
```

Locate to the current folder and run:

```shell
pip install -r requirements.txt
```

### Downloading models

You should download the models manually. Most of the models can be automatically downloaded from Huggingface Hub. However, you should at least download models for `SCRL` method manually. Just follow the guide inside `/models` folder.

## How to use

For **prompt compression** tasks, follow `pctoolkit/compressors.py`, you can modify the compression methods as well as the parameters for them. There is an example in `pctoolkit/compressors.py`, it will be easy to modify.

Or you can follow the code below:

```python
from pctoolkit.compressors import 
    PromptCompressor

compressor = PromptCompressor(type='SCCompressor', device='cuda')

test_prompt = "test prompt"
ratio = 0.3
result = compressor.compressgo(test_prompt, ratio)
print(result)
```

For **evaluation**, follow `pctoolkit_demo.py`. **Please note that if you want to change the metrics, modify pctoolkit/metrics.py, especially for LongBench dataset**.

```python
from pctoolkit.runners import run
from pctoolkit.datasets import load_dataset
from pctoolkit.metrics import load_metrics
from pctoolkit.compressors import PromptCompressor

compressor = PromptCompressor(type='SCCompressor', device='cuda')
dataset_name = 'arxiv'
dataset = load_dataset(dataset_name)

run(compressor=compressor, dataset=dataset, metrics=load_metrics, ratio=0.1)

```

> Hint: Please do remember to fill in your Huggingface Tokens and API keys for OpenAI in pctoolkit/runners.py. (You can also change the urls if you are using other APIs for OpenAI)

## Reference

1. Li, Yucheng et al. “Compressing Context to Enhance Inference Efficiency of Large Language Models.” Conference on Empirical Methods in Natural Language Processing (2023).

2. Jiang, Huiqiang et al. “LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.” Conference on Empirical Methods in Natural Language Processing (2023).

3. Jiang, Huiqiang et al. “LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.” ArXiv abs/2310.06839 (2023): n. pag.

4. Ghalandari, Demian Gholipour et al. “Efficient Unsupervised Sentence Compression by Fine-tuning Transformers with Reinforcement Learning.” ArXiv abs/2205.08221 (2022): n. pag.

5. [Keep It Simple: Unsupervised Simplification of Multi-Paragraph Text](https://aclanthology.org/2021.acl-long.498) (Laban et al., ACL-IJCNLP 2021)

## Citation

PCToolkit is used in your research or applications, please cite it using the following BibTeX:

```bib
nothing here yet
```
