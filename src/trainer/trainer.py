from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics import CERMetric, WERMetric
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        batch["pred_texts_by_decoder"] = {}

        # update metrix for each type of decoder (argmax or beam search)
        if metric_funcs:
            target_texts = [self.text_encoder.normalize_text(t) for t in batch["text"]]
            batch["target_texts"] = target_texts

            with torch.no_grad():
                for decoder_name, decoder in self.decoders.items():
                    pred_texts = decoder(**batch)
                    batch["pred_texts_by_decoder"][decoder_name] = pred_texts

                    for met in metric_funcs:
                        metric_name = f"{met.name}_({decoder_name})"

                        value = met(pred_texts=pred_texts, target_texts=target_texts)
                        metrics.update(metric_name, value)

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            if batch.get("pred_texts_by_decoder"):
                self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self,
        target_texts,
        audio_path,
        pred_texts_by_decoder,
        examples_to_log=5,
        **batch,
    ):
        rows = {}

        for i, (audio_path, target) in enumerate(zip(audio_path, target_texts)):
            if i >= examples_to_log:
                break

            row = {"target": target}

            for decoder_name, pred_texts in pred_texts_by_decoder.items():
                pred = pred_texts[i]

                wer_numer, wer_denom = WERMetric.calc_wer_stats(target, pred)
                cer_numer, cer_denom = CERMetric.calc_cer_stats(target, pred)

                row[f"prediction_({decoder_name})"] = pred
                row[f"WER_({decoder_name})"] = (
                    wer_numer / wer_denom * 100 if wer_denom != 0 else None
                )
                row[f"CER_({decoder_name})"] = (
                    cer_numer / cer_denom * 100 if cer_denom != 0 else None
                )

            rows[Path(audio_path).name] = row

        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
