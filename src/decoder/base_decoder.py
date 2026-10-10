from abc import ABC, abstractmethod
from typing import List

from torch import Tensor


class BaseDecoder(ABC):
    """
    Base class for all decoders.
    Decoder is responsible for converting the output of the model (log_probs) into the predicted text.
    """

    def __init__(self, text_encoder):
        """
        Args:
            text_encoder (CTCTextEncoder)
        """
        self.text_encoder = text_encoder

    @abstractmethod
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, **kwargs
    ) -> List[str]:
        """
        Defines prediction calculation logic for a given batch.
        Returns:
           pred_texts (List[str]): list with predicted texts for a given batch.
        """
        raise NotImplementedError()
