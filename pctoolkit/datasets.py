import sys
sys.path.append('..')
from datasets_helper import Dataset


def load_dataset(dataset_name: str, subdataset_name: str = ""):

    dataset = Dataset(dataset_name, subdataset_name)
    return dataset
