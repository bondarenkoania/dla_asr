from typing import List

import numpy as np
from torch import Tensor

from src.decoder.base_decoder import BaseDecoder


class CTCBeamSearchDecoder(BaseDecoder):
    def __init__(self, beam_size: int, text_encoder):
        super().__init__(text_encoder)
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, **kwargs
    ) -> List[str]:
        log_probs_batch = log_probs.detach().cpu().numpy()
        lengths = log_probs_length.detach().cpu().numpy()

        pred_texts = [
            self.process_one_item(log_prob[:length, :])
            for log_prob, length in zip(log_probs_batch, lengths)
        ]

        return pred_texts

    def process_one_item(self, log_probs: np.ndarray) -> str:
        T = log_probs.shape[0]
        # beam[token_seq] == [log_prob_blank, log_prob_non_blank]
        beam = {(): [0, -np.inf]}

        for t in range(T):
            new_beam = self.expand_and_merge(beam, log_probs[t, :])
            beam = self.truncate_beam(new_beam)

        best = max(beam.keys(), key=lambda x: np.logaddexp(beam[x][0], beam[x][1]))
        return self.text_encoder.decode(best)

    def add_prob(self, new_beam, seq, is_blank, log_prob):
        if seq not in new_beam:
            new_beam[seq] = [-np.inf, -np.inf]

        ind = 0 if is_blank else 1

        new_beam[seq][ind] = np.logaddexp(new_beam[seq][ind], log_prob)

    def expand_and_merge(self, beam: dict, next_tok_probs: np.ndarray) -> dict:
        new_beam = {}
        blank_ind = self.text_encoder.char2ind[self.text_encoder.EMPTY_TOK]

        for seq, (log_prob_b, log_prob_nb) in beam.items():
            for t, log_p in enumerate(next_tok_probs):
                if len(seq) == 0:
                    if t == blank_ind:
                        self.add_prob(new_beam, (), True, log_p + log_prob_b)
                    else:
                        self.add_prob(new_beam, (t,), False, log_p + log_prob_b)

                else:
                    last_tok = seq[-1]
                    if t == blank_ind:
                        self.add_prob(new_beam, seq, True, log_prob_b + log_p)
                        self.add_prob(new_beam, seq, True, log_prob_nb + log_p)
                    elif t == last_tok:
                        self.add_prob(new_beam, seq, False, log_prob_nb + log_p)
                        upd_seq = *seq, t
                        self.add_prob(new_beam, upd_seq, False, log_prob_b + log_p)
                    else:
                        upd_seq = *seq, t
                        self.add_prob(new_beam, upd_seq, False, log_prob_b + log_p)
                        self.add_prob(new_beam, upd_seq, False, log_prob_nb + log_p)

        return new_beam

    def truncate_beam(self, beam: dict) -> dict:
        top_seqs = sorted(
            beam.keys(),
            key=lambda x: np.logaddexp(beam[x][0], beam[x][1]),
            reverse=True,
        )[: self.beam_size]

        return {seq: beam[seq] for seq in top_seqs}
