import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
            Each item contains "audio", "spectrogram", "text", "text_encoded" and "audio_path" fields
    Returns:
        result_batch (dict): dict, containing batched tensors,
        sequence lengths, texts, and audio paths.
    """

    sample = dataset_items[0]
    has_target = sample["text"] is not None

    result_batch = {}

    result_batch["spectrogram"] = pad_sequence(
        [item["spectrogram"].squeeze(0).transpose(0, 1) for item in dataset_items],
        batch_first=True,
    ).transpose(1, 2)
    result_batch["spectrogram_length"] = torch.tensor(
        [item["spectrogram"].shape[-1] for item in dataset_items], dtype=torch.long
    )

    result_batch["text_encoded"] = (
        pad_sequence(
            [item["text_encoded"] for item in dataset_items],
            batch_first=True,
            padding_value=0,
        )
        if has_target
        else None
    )

    result_batch["text_encoded_length"] = (
        torch.tensor(
            [len(item["text_encoded"]) for item in dataset_items], dtype=torch.long
        )
        if has_target
        else None
    )

    result_batch["audio"] = pad_sequence(
        [item["audio"].transpose(0, 1) for item in dataset_items], batch_first=True
    ).transpose(1, 2)

    result_batch["text"] = (
        [item["text"] for item in dataset_items] if has_target else None
    )
    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]

    return result_batch
