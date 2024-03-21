from pctoolkit.runners import run
from pctoolkit.datasets import load_dataset
from pctoolkit.metrics import load_metrics
from pctoolkit.compressors import PromptCompressor

compressor = PromptCompressor(type='SCCompressor', device='cuda')
dataset_name = 'arxiv'
dataset = load_dataset(dataset_name)

run(compressor=compressor, dataset=dataset, metrics=load_metrics, ratio=0.1)
