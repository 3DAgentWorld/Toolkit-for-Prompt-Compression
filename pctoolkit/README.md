# PCToolkit

This is the folder of PCToolkit, including `compressors.py`, `datasets.py`, `metrics.py` and `runners.py`. `/methods` folder contains the source codes of each compressor. `pretrain_models` folder contains pretrained models for invoking, we strongly suggest you download the pretrained models for SCRL following the guidance in `pretrain_models`.

## Compressors

`pctoolkit.compressors` module in PCToolkit encompasses five state-of-the-art compression methods tailored for prompt optimization. 

## Datasets

`pctoolkit.datasets` module boasts a diverse collection of over ten datasets, each meticulously curated to cover a wide array of natural language tasks. From tasks like reconstruction, summarization, question answering, to more specialized domains such as code completion and lies recognition, the datasets in PCToolkit offer a comprehensive testing ground for assessing the efficacy of prompt compression techniques.

We implemented `datasets_helper.py` in the main folder, where you can modify it and include more custom datasets.

## Metrics

`pctoolkit.metrics` module plays a crucial role in quantifying the performance of the compression methods across different tasks. All metrics needed can be easily contained inside a list that tells the Runner which metrics are required measuring. 

## Runners

`pctoolkit.runners` module serves as the engine that drives the evaluation process, orchestrating the interaction between the compression methods, datasets, and evaluation metrics. Researchers and practitioners can seamlessly execute experiments, compare results, and analyze the performance of different compression techniques using the Runner component. This streamlined workflow ensures efficient experimentation and evaluation of prompt compression strategies within the toolkit.
