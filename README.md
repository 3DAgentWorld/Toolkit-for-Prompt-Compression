# Toolkit-for-Prompt-Compression
Toolkit for Prompt Compression (TPC)

## Methods supporting

Currently, TPC includes <a href='https://arxiv.org/abs/2310.06201'>Selective Context</a>, <a href='https://arxiv.org/abs/2310.05736'>LLMLingua</a>, <a href='https://arxiv.org/abs/2310.06839'>LongLLMLingua</a>, <a href='https://arxiv.org/abs/2205.08221'>SCRL</a> and <a href='https://arxiv.org/abs/2107.03444'>Keep it Simple</a>.

## How to start

Locate to the current folder and run:

```shell
pip install -r requirements.txt
```

### Some errors when doing the above operation

It is possible that an error might occur when installing the dependencies. That is because the version of `Spacy` is not very suitablt for the version of `transformers`. To deal with this error, just erase the version numbers of `Spacy` and then run `pip install -r requirements.txt`. After installing all other dependencies, run `pip install spacy==3.7.2`, this time, you will possibly receive an error warning too, but with spacy 3.7.2 successfully installed at the same time.

### Downloading models

You should download the models manually. Most of the models can be automatically downloaded from Huggingface Hub. However, you should at least download models for `SCRL` method manually. Just follow the guide inside `/models` folder.

## How to use

For **prompt compression** tasks, run `compressor.py`, you can modify the compression methods as well as the parameters for them. There is an example in `compressor.py`, it will be easy to modify.

For **methods evaluation** tasks, run `evaluation.py`, you can change the parameter the same as `compressor.py`, and easily change the dataset or add more datasets you want by following the format in `evaluation.py`.

> Hint: Please do remember to fill in your Huggingface Tokens and API keys for OpenAI. (You can also change the urls if you are using other APIs for OpenAI)

# Reference

Li, Yucheng et al. “Compressing Context to Enhance Inference Efficiency of Large Language Models.” Conference on Empirical Methods in Natural Language Processing (2023).

Jiang, Huiqiang et al. “LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.” Conference on Empirical Methods in Natural Language Processing (2023).

Jiang, Huiqiang et al. “LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.” ArXiv abs/2310.06839 (2023): n. pag.

Ghalandari, Demian Gholipour et al. “Efficient Unsupervised Sentence Compression by Fine-tuning Transformers with Reinforcement Learning.” ArXiv abs/2205.08221 (2022): n. pag.

[Keep It Simple: Unsupervised Simplification of Multi-Paragraph Text](https://aclanthology.org/2021.acl-long.498) (Laban et al., ACL-IJCNLP 2021)
