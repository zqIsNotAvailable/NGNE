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
"""FeTaQA is a Free-form Table Question Answering dataset"""

import json
import os

import datasets

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{Nan2021FeTaQAFT,
  title={FeTaQA: Free-form Table Question Answering},
  author={Nan, Linyong and Hsieh, Chiachun and Mao, Ziming and Lin, Xi Victoria and Verma, Neha and Zhang, Rui and 
  Kryściński, Wojciech and Schoelkopf, Hailey and Kong, Riley and Tang, Xiangru and Mutuma, Mutethia and Rosand, 
  Ben and Trindade, Isabel and Bandaru, Renusree and Cunningham, Jacob and Xiong, Caiming and Radev, Dragomir},
  journal={Transactions of the Association for Computational Linguistics},
  year={2022},
  volume={10},
  pages={35-49}
}
"""

_DESCRIPTION = """\
FeTaQA is a Free-form Table Question Answering dataset with 10K Wikipedia-based {table, question, free-form answer, 
supporting table cells} pairs. It yields a more challenging table QA setting because it requires generating free-form 
text answers after retrieval, inference, and integration of multiple discontinuous facts from a structured knowledge 
source. Unlike datasets of generative QA over text in which answers are prevalent with copies of short text spans from 
the source, answers in our dataset are human-generated explanations involving entities and their high-level relations.
"""

_HOMEPAGE = "https://github.com/Yale-LILY/FeTaQA"

_LICENSE = "CC-BY-SA-4.0 license"

# _URLS = {
#     "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
#     "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
# }

_URL = "https://codeload.github.com/Yale-LILY/FeTaQA/zip/refs/heads/main"


class NewDataset(datasets.GeneratorBasedBuilder):
    """The FeTaQA dataset"""

    # VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    # BUILDER_CONFIGS = [

    def _info(self):
        features = datasets.Features(
            {
                "feta_id": datasets.Value("string"),
                "table_page_title": datasets.Value("string"),
                "table_section_title": datasets.Value("string"),
                "table_array": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        data_dir = os.path.join(dl_manager.download_and_extract(_URL), 'FeTaQA-main')
        # data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data/fetaQA-v1_train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data/fetaQA-v1_dev.jsonl"),
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data/fetaQA-v1_test.jsonl"),
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):

        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                # Yields examples as (key, example) tuples
                yield key, {
                    "feta_id": str(data["feta_id"]),
                    "table_page_title": data["table_page_title"],
                    "table_section_title": data["table_section_title"],
                    "table_array": data["table_array"],
                    "question": data["question"],
                    "answer": data["answer"],
                }

