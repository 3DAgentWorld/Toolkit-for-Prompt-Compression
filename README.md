# Toolkit-for-Prompt-Compression
Toolkit for Prompt Compression

## How to start

Locate to the current folder and run

```shell
pip install -r requirements.txt
```

### Some errors when doing the above operation

It is possible that an error might occur when installing the dependencies. That is because the version of `Spacy` is not very suitablt for the version of `transformers`. To deal with this error, just erase the version numbers of `Spacy` and then run `pip install -r requirements.txt`. After installing all other dependencies, run `pip install spacy==3.7.2`, this time, you will possibly receive an error warning too, but with spacy 3.7.2 successfully installed at the same time.

## How to use

For **prompt compression** tasks, run `compressor.py`, you can modify the compression methods as well as the parameters for them. There is an example in `compressor.py`, it will be easy to modify.

For **methods evaluation** tasks, run `evaluation.py`, you can change the parameter the same as `compressor.py`, and easily change the dataset or add more datasets you want by following the format in `evaluation.py`.

> Hint: Please do remember to fill in your Huggingface Tokens and API keys for OpenAI. (You can also change the urls if you are using other APIs for OpenAI)

