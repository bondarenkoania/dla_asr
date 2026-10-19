# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#demo">Demo</a> •
  <a href="#report">Report</a> •
  <a href="#reproducibility">Reproducibility</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a complete speech recognition pipeline built with PyTorch for the [HSE DLA course](https://github.com/markovka17/dla) ASR homework. See the task assignment [here](https://github.com/markovka17/dla/tree/2026/hw1ASR).

The model uses the [Conformer architecture](https://arxiv.org/abs/2005.08100) with CTC loss. We also use the [OpenSLR pruned 3-gram language model](https://www.openslr.org/11/) for shallow fusion during beam search decoding.

## Results

Final test results with beam search and LM fusion.

| LibriSpeech split | CER (%) | WER (%) |
|---|---:|---:|
| test-clean | 3.95 | 10.45 |
| test-other | 11.13 | 23.60 |

## Installation

0. Create and activate a new environment with Python 3.11 using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html), `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)), or [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

   a. `conda` version:

   ```bash
   conda create -n project_env python=3.11    # create
   conda activate project_env                 # activate
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   python3.11 -m venv project_env             # create
   source project_env/bin/activate            # activate
   ```

   c. `uv` version:

   ```bash
   uv venv project_env --python 3.11 --seed   # create
   source project_env/bin/activate            # activate
   ```

1. Clone the repository:

   ```bash
   git clone https://github.com/bondarenkoania/dla_asr.git
   cd dla_asr
   ```

2. Install all required packages:

   ```bash
   pip install -r requirements.txt
   ```

## How To Use

Run the commands below from the repository root with your environment activated. Relative paths are resolved from this directory.

### Use the pretrained model

Download the pretrained Conformer Medium checkpoint and the language model for evaluation:

```bash
python download_resources.py
```

The script saves the checkpoint to `saved/pretrained/model_best.pth` and extracts the language model to `data/lm/librispeech_3gram.arpa`.

To run inference on Librispeech test-clean and test-other, run

```bash
python inference.py \
  inferencer.from_pretrained=saved/pretrained/model_best.pth \
  inferencer.save_path=null
```
It prints CER and WER for each split without saving transcriptions. To save transcriptions, set `inferencer.save_path=test_predictions`.


To use your own data, select the `custom` dataset config. `CustomDirAudioDataset` expects the following directory structure:

```text
my_dataset/
├── audio/
│   ├── utterance_1.wav
│   └── utterance_2.flac
└── transcriptions/
    ├── utterance_1.txt
    └── utterance_2.txt
```

The `transcriptions` directory is optional for inference. If provided, it must contain one TXT file for every audio file in `audio`, with the same filename stem (for example, `utterance_1.wav` and `utterance_1.txt`). Audio IDs must be unique.

Run inference with beam search and the language model. Replace `/path/to/my_dataset` with the actual path to your dataset directory (absolute or relative to the repository root):

```bash
python inference.py datasets=custom decoder=beam_lm \
  datasets.test.custom_dir=/path/to/my_dataset \
  inferencer.from_pretrained=saved/pretrained/model_best.pth \
  decoder.BeamSearchLM.path_to_lm=data/lm/librispeech_3gram.arpa \
  inferencer.save_path=custom_predictions
```

Predictions are saved as TXT files with the original utterance IDs in `data/saved/custom_predictions/test/`.

Add `dataloader.batch_size=1` if your dataset contains fewer than 10 audio files.

If reference transcriptions are available, use `calc_metrics.py` to calculate CER and WER:

```bash
python calc_metrics.py /path/to/my_dataset/transcriptions \
  data/saved/custom_predictions/test
```

### Train a model

To train the Conformer model from scratch on LibriSpeech train-clean-100, set the `COMET_API_KEY` environment variable to your Comet API key and run:

```bash
python train.py
```

You can select a different training config with `--config-name=CONFIG_NAME` or override individual settings using `key=value`.


See [Reproducibility](#reproducibility) for the stages used to train our final model.

## Demo

The [demo notebook](/demo.ipynb) shows how to install the project, download the model, run inference, and calculate metrics on your own dataset. Provide a direct download link to a ZIP archive with your data and run the cells in Google Colab.


## Report

Our Comet report presents the training logs, some experiments and their description.

[COMET REPORT](https://www.comet.com/bondarenkoania/dla-asr-ctc/reports/9K6Wq94PyTGwcslhak45ZI2hA)

## Reproducibility

To reproduce the final model, follow the installation instructions and run these training stages in order. For each stage, use the specified config and load the checkpoint from the preceding stage.

Training was performed in several stages:

- `train.yaml`: 12 full epochs on LibriSpeech train-clean-100, followed by 8 additional epochs with stronger augmentations.
- `finetune_other_inverse_sqrt.yaml`: fine-tuning on train-other-500 for 25 epochs of 1000 steps each, using warmup followed by inverse square root learning rate decay.
- `finetune_other_cosine.yaml`: 25 epochs of 1000 steps each, with cosine learning rate decay from 5e-5 to 1e-5. And then 15 epochs with a new cosine schedule from 3e-5 to 1e-5.

The exact configurations used for each run, including command-line overrides, are available in the linked Comet experiments in [report](#report).


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
