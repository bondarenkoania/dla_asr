from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF
from torch import Tensor, nn

from src.utils.io_utils import ROOT_PATH


class AddBackgroundNoise(nn.Module):
    def __init__(self, noise_dir, min_snr=10.0, max_snr=30.0, sample_rate=16000):
        super().__init__()
        self.noise_paths = list((ROOT_PATH / Path(noise_dir)).glob("*.wav"))
        if not self.noise_paths:
            raise ValueError(f"No WAV noise files found in {noise_dir}")
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sample_rate = sample_rate

    def forward(self, audio: Tensor) -> Tensor:
        noise_path = self.noise_paths[torch.randint(len(self.noise_paths), ()).item()]
        noise, sr = torchaudio.load(noise_path)
        noise = noise.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            noise = AF.resample(noise, sr, self.sample_rate)
        noise = noise.to(audio)

        length = audio.shape[-1]
        if noise.shape[-1] < length:
            noise = noise.repeat(1, (length + noise.shape[-1] - 1) // noise.shape[-1])
        start = torch.randint(noise.shape[-1] - length + 1, ()).item()
        noise = noise[..., start : start + length]

        snr = audio.new_empty(1).uniform_(self.min_snr, self.max_snr)
        mixed = AF.add_noise(audio, noise, snr)
        return mixed / mixed.abs().amax().clamp_min(1.0)
