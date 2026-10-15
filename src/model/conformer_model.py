import numpy as np
import torch
from torch import nn


class ConformerModel(nn.Module):
    """
    Conformer model from https://arxiv.org/abs/2005.08100
    """

    def __init__(
        self,
        n_feats,
        n_tokens,
        d_model=512,
        num_blocks=8,
        num_heads=8,
        ff_expansion_factor=4,
        dropout=0.1,
        kernel_size=32,
        subsampling_channels=32,
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            d_model (int): number of hidden features.
            num_blocks (int): number of Conformer blocks.
            num_heads (int): number of attention heads in each block.
            ff_expansion_factor (int): expansion factor in feed-forward modules.
            dropout (float): dropout probability.
            kernel_size (int): depthwise convolution kernel size.
            subsampling_channels (int): number of channels in subsampling convolutions.
        """
        super().__init__()

        self.subsampling = ConvolutionalSubsampling(
            n_feats, d_model, subsampling_channels
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model, num_heads, ff_expansion_factor, dropout, kernel_size
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear = nn.Linear(d_model, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram. [B, n_feats, T]
            spectrogram_length (Tensor): spectrogram original lengths. [B,]
        Returns:
            output (dict): log_probs [B, T_new, n_tokens] and log_probs_length [B,].
        """
        x, log_probs_length = self.subsampling(spectrogram, spectrogram_length)
        x = self.dropout(x)
        positions = torch.arange(x.shape[1], device=x.device)
        padding_mask = (
            positions[None, :] >= log_probs_length.to(device=x.device)[:, None]
        )  # True for padding
        for block in self.blocks:
            x = block(x, padding_mask)

        output = self.linear(x)
        log_probs = nn.functional.log_softmax(output, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return self.subsampling.transform_input_lengths(input_lengths)

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class ConformerBlock(nn.Module):
    """
    Conformer block from Section 2.4 and Equation (1) of the Conformer paper.
    Input and output: [B, T, d], where d = d_model.
    """

    def __init__(
        self, d_model, num_heads, ff_expansion_factor=4, dropout=0.1, kernel_size=32
    ):
        super().__init__()

        self.ffn1 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.attention = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, dropout, kernel_size)
        self.ffn2 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask):
        """
        Args:
            x (Tensor): [B, T, d]
            padding_mask (BoolTensor): True for padded positions, [B, T]
        Returns:
            output (Tensor): [B, T, d]
        """
        x1 = x + 0.5 * self.ffn1(x)
        x2 = x1 + self.attention(x1, padding_mask)
        x3 = x2 + self.conv(x2, padding_mask)
        x4 = x3 + 0.5 * self.ffn2(x3)
        return self.norm(x4)


class ConvolutionalSubsampling(nn.Module):
    """
    Two 3x3 convolutions with stride 2, followed by a linear projection.
    Input: [B, n_feats, T]
    Output: [B, T2, d], where d = d_model.
    """

    def __init__(self, n_feats, d_model, channels=32):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )
        f_new = self.transform_input_lengths(n_feats)
        self.linear = nn.Linear(channels * f_new, d_model)

    def forward(self, spectrogram, spectrogram_length):
        """
        Args:
            spectrogram (Tensor): [B, n_feats, T]
            spectrogram_length (LongTensor): valid input lengths, [B,]
        Returns:
            output (Tensor): [B, new_T, d]
            output_lengths (LongTensor): valid output lengths, [B,]
        """

        # [B, n_feats, T] -> [B, 1, T, n_feats]
        x = spectrogram.transpose(1, 2).unsqueeze(1)
        x = self.conv(x)  # [B, 1, T, n_feats] -> [B, c, T_new, f_new]
        # [B, c, T_new, f_new] -> [B, T2_new, d]
        x = self.linear(x.transpose(1, 2).flatten(2))
        output_lengths = self.transform_input_lengths(spectrogram_length)
        return x, output_lengths

    def transform_input_lengths(self, input_lengths):
        lengths = (input_lengths - 1) // 2
        return (lengths - 1) // 2


class FeedForwardModule(nn.Module):
    """
    Feed-forward module from Section 2.3 and Figure 4 of the Conformer paper.
    Input and output: [B, T, d], where d = d_model.
    """

    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(d_model),  # [B, T, d] -> [B, T, d], normalize over d
            nn.Linear(d_model, d_model * expansion_factor),  # [B, T, d] -> [B, T, e*d]
            nn.SiLU(),  # Swish: x * sigmoid(x), [B, T, e*d]
            nn.Dropout(dropout),  # [B, T, e*d]
            nn.Linear(d_model * expansion_factor, d_model),  # [B, T, e*d] -> [B, T, d]
            nn.Dropout(dropout),  # [B, T, d]
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadSelfAttentionModule(nn.Module):
    """
    Self-attention module from Section 2.1 and Figure 3 of the Conformer paper.
    Input and output: [B, T, d], where d = d_model.
    Relative sinusoidal positional encoding scheme follows Section 3.3 of Transformer-XL https://aclanthology.org/P19-1285/.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm = nn.LayerNorm(d_model)

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.position = nn.Linear(d_model, d_model, bias=False)

        self.heads_linear = nn.Linear(d_model, d_model, bias=False)
        self.content_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.position_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        """
        Args:
            x (Tensor): [B, T, d]
            padding_mask (BoolTensor): True for padded positions, [B, T]
        Returns:
            output (Tensor): [B, T, d]
        """
        batch_size, length, _ = x.shape
        h, d_h = self.num_heads, self.head_dim
        x = self.norm(x)  # [B, T, d] -> [B, T, d]

        # [B, T, d] -> [B, h, T, d_h]
        q = self.query(x).view(batch_size, length, h, d_h).transpose(1, 2)
        k = self.key(x).view(batch_size, length, h, d_h).transpose(1, 2)
        v = self.value(x).view(batch_size, length, h, d_h).transpose(1, 2)

        positions = self._relative_position_encoding(length, x.device)  # [2*T-1, d]
        # [2*T-1, d] -> [h, 2*T-1, d_h]
        r = self.position(positions).view(2 * length - 1, h, d_h).transpose(0, 1)
        # [B, h, T, T]
        scores = (q + self.content_bias[None, :, None, :]) @ k.transpose(2, 3)
        # [B, h, T, 2*T-1]
        pos_scores = (q + self.position_bias[None, :, None, :]) @ r.transpose(1, 2)

        inds = torch.arange(length, device=x.device)  # [T]
        rel_inds = inds[:, None] - inds[None, :] + length - 1  # [T, T]
        # [B, h, T, T]
        rel_inds = rel_inds[None, None, :, :].expand(batch_size, h, -1, -1)
        pos_scores = pos_scores.gather(-1, rel_inds)  # [B, h, T, 2*T-1] -> [B, h, T, T]
        final_scores = (scores + pos_scores) / d_h**0.5  # [B, h, T, T]

        mask = padding_mask[:, None, None, :]  # [B, 1, 1, T]
        final_scores = final_scores.masked_fill(mask, -np.inf)

        weights = final_scores.softmax(dim=-1)  # [B, h, T, T]
        result = weights @ v  # [B, h, T, d_h]
        result = result.transpose(1, 2).reshape(
            batch_size, length, self.d_model
        )  # [B, h, T, d_h] -> [B, T, d]
        output = self.dropout(self.heads_linear(result))  # [B, T, d] -> [B, T, d]
        return output

    def _relative_position_encoding(self, length, device):
        offsets = torch.arange(1 - length, length, device=device, dtype=torch.float32)
        frequencies = 10000.0 ** (
            -torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            / self.d_model
        )
        angles = offsets[:, None] * frequencies[None, :]
        encoding = torch.zeros(
            2 * length - 1, self.d_model, device=device, dtype=torch.float32
        )
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, : self.d_model // 2].cos()
        return encoding.to(dtype=self.position.weight.dtype)


class ConvolutionModule(nn.Module):
    """
    Convolution module from Section 2.2 and Figure 2 of the Conformer paper.
    Input and output: [B, T, d], where d = d_model.
    """

    def __init__(self, d_model, dropout=0.1, kernel_size=32):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size, groups=d_model, padding="same"
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        """
        Args:
            x (Tensor): [B, T, d]
            padding_mask (BoolTensor): True for padded positions, [B, T]
        Returns:
            output (Tensor): [B, T, d]
        """
        x = self.norm(x)  # [B, T, d]
        x = self.pointwise_conv1(x.transpose(1, 2))  # [B, 2*d, T]
        x = self.glu(x)  # [B, 2*d, T] -> [B, d, T]

        mask = padding_mask[:, None, :]  # [B, T] -> [B, 1, T]
        x = x.masked_fill(
            mask, 0.0
        )  # [B, d, T], make zeros before mixing time positions
        x = self.depthwise_conv(x)  # [B, d, T] -> [B, d, T]
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)
