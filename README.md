# PCToolkit: A Unified Plug-and-Play Prompt Compression Toolkit of Large Language Models

<div align=center>
<img src="https://github.com/3DAgentWorld/Toolkit-for-Prompt-Compression/blob/main/imgs/logo.png" width="400" height="105">
	
---

[![GitHub license](https://img.shields.io/github/license/3DAgentWorld/Toolkit-for-Prompt-Compression?color=blue)](https://github.com/3DAgentWorld/Toolkit-for-Prompt-Compression/blob/main/LICENSE)
</div>

# Contents
- [Methods supporting](#methods-supporting)
- [How to start](#how-to-start)
	- [Downloading models](#downloading-models)
- [How to use](#how-to-use)
- [Relevant Repositories](#relevant-repositories)
- [Reference](#reference)

## Methods supporting

Currently, TPC includes <a href='https://arxiv.org/abs/2310.06201'>Selective Context</a>, <a href='https://arxiv.org/abs/2310.05736'>LLMLingua</a>, <a href='https://arxiv.org/abs/2310.06839'>LongLLMLingua</a>, <a href='https://arxiv.org/abs/2205.08221'>SCRL</a> and <a href='https://arxiv.org/abs/2107.03444'>Keep it Simple</a>.

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

## Relevant Repositories

- <a href='https://github.com/liyucheng09/selective_context'>Selective Context</a> ![Github stars](https://img.shields.io/github/stars/liyucheng09/selective_context.svg)
- <a href='https://github.com/microsoft/LLMLingua'>LLMLingua</a></a> ![Github stars](https://img.shields.io/github/stars/microsoft/LLMLingua.svg)
- <a href='https://github.com/microsoft/LLMLingua'>LongLLMLingua</a> ![Github stars](https://img.shields.io/github/stars/microsoft/LLMLingua.svg)
- <a href='https://github.com/complementizer/rl-sentence-compression'>SCRL</a> ![Github stars](https://img.shields.io/github/stars/complementizer/rl-sentence-compression.svg)
- <a href='https://github.com/tingofurro/keep_it_simple'>KiS</a> ![Github stars](https://img.shields.io/github/stars/tingofurro/keep_it_simple.svg)

## Reference

1. Li, Yucheng et al. “Compressing Context to Enhance Inference Efficiency of Large Language Models.” Conference on Empirical Methods in Natural Language Processing (2023).

2. Jiang, Huiqiang et al. “LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.” Conference on Empirical Methods in Natural Language Processing (2023).

3. Jiang, Huiqiang et al. “LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.” ArXiv abs/2310.06839 (2023): n. pag.

4. Ghalandari, Demian Gholipour et al. “Efficient Unsupervised Sentence Compression by Fine-tuning Transformers with Reinforcement Learning.” ArXiv abs/2205.08221 (2022): n. pag.

5. [Keep It Simple: Unsupervised Simplification of Multi-Paragraph Text](https://aclanthology.org/2021.acl-long.498) (Laban et al., ACL-IJCNLP 2021)
