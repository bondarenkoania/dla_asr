from typing import List

import torch
from torch import Tensor

from src.decoder.base_decoder import BaseDecoder


class CTCBeamSearchDecoder(BaseDecoder):
    def __init__(self, beam_size: int, text_encoder):
        super().__init__(text_encoder)
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, **kwargs
    ) -> List[str]:
        # TODO
        pass
