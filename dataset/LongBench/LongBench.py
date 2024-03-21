# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import datasets
import json


_DESCRIPTION = """\
LongBench is a comprehensive benchmark for multilingual and multi-task purposes, with the goal to fully measure and evaluate the ability of pre-trained language pretrained_models to understand long text. This dataset consists of twenty different tasks, covering key long-text application scenarios such as multi-document QA, single-document QA, summarization, few-shot learning, synthetic tasks, and code completion.
"""

_HOMEPAGE = "https://github.com/THUDM/LongBench"


_URL = r"https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"

task_list = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
    "qasper_e",
    "multifieldqa_en_e",
    "hotpotqa_e",
    "2wikimqa_e",
    "gov_report_e",
    "multi_news_e",
    "trec_e",
    "triviaqa_e",
    "samsum_e",
    "passage_count_e",
    "passage_retrieval_en_e",
    "lcc_e",
    "repobench-p_e"
]


class LongBenchConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class LongBench(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        LongBenchConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "input": datasets.Value("string"), 
                "context": datasets.Value("string"), 
                "answers": [datasets.Value("string")], 
                "length": datasets.Value("int32"), 
                "dataset": datasets.Value("string"), 
                "language": datasets.Value("string"), 
                "all_classes": [datasets.Value("string")],
                "_id": datasets.Value("string"), 
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        task_name = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", f"{task_name}.jsonl"
                    ),
                },
            )
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                key = f"{self.config.name}-{idx}"
                item = json.loads(line)
                yield key, {
                    "input": item["input"],
                    "context": item["context"],
                    "answers": item["answers"],
                    "length": item["length"],
                    "dataset": item["dataset"],
                    "language": item["language"],
                    "_id": item["_id"],
                    "all_classes": item["all_classes"],
                }