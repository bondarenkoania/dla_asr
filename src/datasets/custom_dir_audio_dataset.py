from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, custom_dir, *args, **kwargs):
        audio_dir = Path(custom_dir) / "audio"
        transcription_dir = Path(custom_dir) / "transcriptions"
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                info = torchaudio.info(str(path))
                length = info.num_frames / info.sample_rate
                entry["audio_len"] = length
                entry["text"] = None
                if transcription_dir:
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    with transc_path.open() as f:
                        entry["text"] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
