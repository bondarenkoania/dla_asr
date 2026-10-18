from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        decoders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            decoders (dict): decoders for predictions.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = {} if batch_transforms is None else batch_transforms

        self.text_encoder = text_encoder
        self.decoders = decoders

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition
        self.save_path = Path(save_path) if save_path is not None else None

        # define metrics
        self.metrics = metrics

        metric_names = [
            f"{met.name}_({decoder_name})"
            for decoder_name in self.decoders
            for met in self.metrics["inference"]
        ]

        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *metric_names,
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        batch["pred_texts_by_decoder"] = {}
        with torch.no_grad():
            for decoder_name, decoder in self.decoders.items():
                pred_texts = decoder(**batch)
                batch["pred_texts_by_decoder"][decoder_name] = pred_texts

        if metrics is not None and batch["text"] is not None:
            target_texts = [self.text_encoder.normalize_text(t) for t in batch["text"]]
            batch["target_texts"] = target_texts

            for met in self.metrics["inference"]:
                for decoder_name, pred_texts in batch["pred_texts_by_decoder"].items():
                    metric_name = f"{met.name}_({decoder_name})"

                    value = met(pred_texts=pred_texts, target_texts=target_texts)
                    metrics.update(metric_name, value)

        if self.save_path is not None:
            for decoder_name, pred_texts in batch["pred_texts_by_decoder"].items():
                if len(self.decoders) > 1:
                    output_dir = Path(self.save_path) / part / decoder_name
                else:
                    output_dir = Path(self.save_path) / part

                output_dir.mkdir(parents=True, exist_ok=True)

                for pred_text, audio_path in zip(pred_texts, batch["audio_path"]):
                    utterance_id = Path(audio_path).stem
                    output_path = output_dir / f"{utterance_id}.txt"
                    output_path.write_text(pred_text, encoding="utf-8")

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics:
            self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result() if self.evaluation_metrics else None
