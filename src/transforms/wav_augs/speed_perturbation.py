import torchaudio
from torch import Tensor, nn


class SpeedPerturbation(nn.Module):
    def __init__(self, sample_rate=16000, factors=(0.9, 1.1)):
        super().__init__()
        self._aug = torchaudio.transforms.SpeedPerturbation(sample_rate, factors)

    def forward(self, audio: Tensor) -> Tensor:
        audio, _ = self._aug(audio)
        return audio
