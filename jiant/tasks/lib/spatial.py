import numpy as np
import torch
import pandas as pd
from dataclasses import dataclass
from typing import List

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    GlueMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import double_sentence_featurize, labels_to_bimap, single_sentence_featurize
from jiant.utils.python.io import read_file_lines


@dataclass
class Example(BaseExample):
    guid: str
    sentence: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            text=tokenizer.tokenize(self.sentence),
            label_id=self.label,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.text,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class SpatialTask(Task):

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = ["0", "1"]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)
    Batch = Batch

    def get_train_examples(self):
        return self._create_examples(path=self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(path=self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(path=self.test_path, set_type="test")

    @classmethod
    def _create_examples(cls, path, set_type):
        examples = []
        df = pd.read_csv(path, index_col=0)
        for i, row in df.iterrows():
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    sentence=row.sentence,
                    label=row.label if set_type != "test" else cls.LABELS[-1],
                )
            )

        return examples