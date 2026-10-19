import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve

ROOT_PATH = Path(__file__).resolve().parent
CHECKPOINT_URL = (
    "https://huggingface.co/bondarenkoania/conformer-ctc-asr/resolve/main/"
    "model_best_final.pth"
)
LM_URL = "https://openslr.elda.org/resources/11/3-gram.pruned.1e-7.arpa.gz"


def main():
    Path.mkdir(ROOT_PATH / "saved" / "pretrained", parents=True, exist_ok=True)

    checkpoint = ROOT_PATH / "saved" / "pretrained" / "model_best.pth"
    print(f"Downloading checkpoint to {checkpoint} ...", flush=True)
    urlretrieve(CHECKPOINT_URL, checkpoint)

    Path.mkdir(ROOT_PATH / "data" / "lm", parents=True, exist_ok=True)
    lm_path = ROOT_PATH / "data" / "lm" / "librispeech_3gram.arpa"
    archive = ROOT_PATH / "data" / "lm" / "librispeech_3gram.arpa.gz"

    print(f"Downloading language model to {archive} ...", flush=True)
    urlretrieve(LM_URL, archive)

    print(f"Extracting language model to {lm_path} ...", flush=True)
    with gzip.open(archive, "rb") as source, lm_path.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    archive.unlink()
    print("Checkpoint and language model are ready.", flush=True)


if __name__ == "__main__":
    main()
