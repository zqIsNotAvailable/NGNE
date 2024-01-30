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
"""The FinQA is a dataset of numerical reasoning over financial data"""

import json
import os

import datasets

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{chen2021finqa,
  title={FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author={Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Mous
  sa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang},
  journal={Proceedings of EMNLP 2021},
  year={2021}
}
"""

_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

_HOMEPAGE = "https://github.com/czyssrs/FinQA"

_LICENSE = "MIT license"

# _URLS = {
#     "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
#     "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
# }

_URL = "https://codeload.github.com/czyssrs/FinQA/zip/refs/heads/main"


class NewDataset(datasets.GeneratorBasedBuilder):
    """The FinQA dataset"""

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
                "pre_text": datasets.Sequence(datasets.Value("string")),
                "post_text": datasets.Sequence(datasets.Value("string")),
                "table": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                "id": datasets.Value("string"),
                "qa": {"question": datasets.Value("string"),
                       # "program": datasets.Value("string"),
                       # "gold_inds": {},
                       "exe_ans": datasets.Value("string"),
                       # "program_re": datasets.Value("string")
                       },
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

        data_dir = os.path.join(dl_manager.download_and_extract(_URL), 'FinQA-main')
        # data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dataset/train.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dataset/dev.json"),
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dataset/test.json"),
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for key, row in enumerate(data):
                # Yields examples as (key, example) tuples
                yield key, {
                    "pre_text": row["pre_text"],
                    "post_text": row["post_text"],
                    "table": row["table"],
                    "id": row["id"],
                    "qa": {
                        "question": row["qa"]["question"],
                        # "program": row["qa"]["program"],
                        # "gold_inds": row["qa"]["gold_inds"],
                        "exe_ans": str(row["qa"]["exe_ans"]),
                        # "program_re": row["qa"]["program_re"]
                    },
                }

