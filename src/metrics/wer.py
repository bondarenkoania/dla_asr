from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric


class WERMetric(BaseMetric):
    def __call__(self, pred_texts: list[str], target_texts: list[str]):
        cers = []
        for target_text, pred_text in zip(target_texts, pred_texts):
            cers.append(self.calc_wer(target_text, pred_text))

        return sum(cers) / len(cers)

    @staticmethod
    def calc_wer(target_text, predicted_text) -> float:
        # Based on seminar materials
        # Don't forget to support cases when target_text == ''
        # TODO
        pass
