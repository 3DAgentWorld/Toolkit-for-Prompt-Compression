from datasets import load_dataset


class Dataset:

    def __init__(self, dataset_name: str, subdataset_name: str = ""):
        self.dataset_name = dataset_name
        self.subdataset_name = subdataset_name
        self.data = self.load_data(self.dataset_name, self.subdataset_name)

    def load_data(self, dataset_name: str, subdataset_name: str):
        if dataset_name == "arxiv":
            dataset = load_dataset("parquet", data_files={'test': 'dataset/arxiv/train-00000-of-00001-b334c773bce22cb2.parquet'}, split="test")
            return dataset
        elif dataset_name == "sharegpt":
            dataset = load_dataset("parquet", data_files={'test': 'dataset/sharegpt/train-00000-of-00001-18e3e661ded310e9.parquet'}, split="test")
            return dataset
        elif dataset_name == "bbc":
            dataset = load_dataset("json", data_files={'test': 'dataset/bbc/articles.json'}, split="test")
            return dataset
        elif dataset_name == "GSM":
            dataset = load_dataset("json", data_files={'test': 'dataset/GSM8K/grade_school_math/data/test.jsonl'}, split="test")
            return dataset
        elif dataset_name == "LongBench":
            dataset = load_dataset("json", data_files={'test': f'dataset/LongBench/{subdataset_name}.jsonl'}, split="test")
            return dataset
        elif dataset_name == "BBH":
            dataset = load_dataset("json", data_files={'test': f'dataset/BBH/bbh/{subdataset_name}.json'}, split="test")
            return dataset
        elif dataset_name == "gigaword":
            dataset = load_dataset("json", data_files={'test': 'dataset/SCRL_datasets/gigaword.jsonl'}, split="test")
            return dataset
        elif dataset_name == "duc2004":
            dataset = load_dataset("json", data_files={'test': 'dataset/SCRL_datasets/duc2004.jsonl'}, split="test")
            return dataset
        elif dataset_name == "bnc":
            dataset = load_dataset("json", data_files={'test': 'dataset/SCRL_datasets/bnc.jsonl'}, split="test")
            return dataset
        elif dataset_name == "broadcast":
            dataset = load_dataset("json", data_files={'test': 'dataset/SCRL_datasets/broadcast.jsonl'}, split="test")
            return dataset
        elif dataset_name == "google":
            dataset = load_dataset("json", data_files={'test': 'dataset/SCRL_datasets/google.jsonl'}, split="test")
            return dataset
        else:
            print("Unknown dataset")
            return None

