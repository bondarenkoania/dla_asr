import argparse
from pathlib import Path

from src.metrics import CERMetric, WERMetric
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def main():
    parser = argparse.ArgumentParser(description="Calculate corpus CER and WER.")
    parser.add_argument(
        "ground_truth_dir", type=Path, help="Directory with reference TXT files"
    )
    parser.add_argument(
        "predictions_dir", type=Path, help="Directory with predicted TXT files"
    )
    args = parser.parse_args()

    for directory in (args.ground_truth_dir, args.predictions_dir):
        if not directory.is_dir():
            parser.error(f"Directory does not exist: {directory}")

    targets = sorted(args.ground_truth_dir.glob("*.txt"))
    predictions = args.predictions_dir.glob("*.txt")

    if not targets or not predictions:
        parser.error("No reference or predicted TXT files found.")

    if {path.name for path in targets} != {path.name for path in predictions}:
        parser.error(
            "Reference and prediction directories must contain the same TXT filenames."
        )

    char_errors = char_count = word_errors = word_count = 0
    for target_path in targets:
        prediction_path = args.predictions_dir / target_path.name
        target = CTCTextEncoder.normalize_text(
            target_path.read_text(encoding="utf-8").strip()
        )
        prediction = CTCTextEncoder.normalize_text(
            prediction_path.read_text(encoding="utf-8").strip()
        )

        errors, count = CERMetric.calc_cer_stats(target, prediction)
        char_errors += errors
        char_count += count
        errors, count = WERMetric.calc_wer_stats(target, prediction)
        word_errors += errors
        word_count += count

    if char_count == 0 or word_count == 0:
        parser.error("References contain no characters or words after normalization.")

    print(f"CER: {char_errors / char_count * 100:.2f}%")
    print(f"WER: {word_errors / word_count * 100:.2f}%")


if __name__ == "__main__":
    main()
