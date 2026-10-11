from typing import List

import torch
from torch import Tensor

from src.decoder.base_decoder import BaseDecoder


class CTCGreedyDecoder(BaseDecoder):
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, **kwargs
    ) -> List[str]:
        predictions = torch.argmax(log_probs, dim=-1).cpu().numpy()
        lengths = log_probs_length.detach().cpu().numpy()

        pred_texts = [
            self.text_encoder.ctc_decode(pred_vec[:length])
            for pred_vec, length in zip(predictions, lengths)
        ]

        return pred_texts
