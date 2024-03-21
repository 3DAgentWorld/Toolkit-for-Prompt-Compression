import os
import json
import datasets
from pathlib import Path


_DESCRIPTION = "Gigaword dataset"
_DOCUMENT = "document"
_ID = "id"


class GigawordDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _DOCUMENT: datasets.Value("string"),
                    _ID: datasets.Value("string"),
                }
            ),
            #supervised_keys=(_DOCUMENT, _SUMMARY),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager._data_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": os.path.join(data_dir, "train.jsonl"), "name": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": os.path.join(data_dir, "val.jsonl"), "name": "validation"}
            ),
        ]

    def _generate_examples(self, path=None, name=None):
        """Yields examples."""
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                x = json.loads(line)
                id = x["id"]
                item = {
                    _ID: id,
                    _DOCUMENT: x["text"],
                }
                yield id, item
