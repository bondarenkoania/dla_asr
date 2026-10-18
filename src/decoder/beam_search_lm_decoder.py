from typing import List

from pyctcdecode import build_ctcdecoder
from torch import Tensor

from src.decoder.base_decoder import BaseDecoder


class CTCBeamSearchLMDecoder(BaseDecoder):
    def __init__(self, beam_size: int, text_encoder, path_to_lm, alpha, beta):
        super().__init__(text_encoder)
        self.beam_size = beam_size
        self.decoder = build_ctcdecoder(
            labels=[label.upper() for label in text_encoder.vocab],
            kenlm_model_path=path_to_lm,
            alpha=alpha,
            beta=beta,
        )

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, **kwargs
    ) -> List[str]:
        log_probs_batch = log_probs.detach().cpu().numpy()
        lengths = log_probs_length.detach().cpu().numpy()

        pred_texts = [
            self.decoder.decode(log_prob[:length, :], beam_width=self.beam_size)
            for log_prob, length in zip(log_probs_batch, lengths)
        ]

        return [self.text_encoder.normalize_text(pred) for pred in pred_texts]
