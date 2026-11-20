"""NormalFloat4 (NF4) Quantization from Scratch.

Reference: Tim Dettmers et al. (2023) - QLoRA: Efficient Finetuning of Quantized LLMs.

Core Insight:
    Pre-trained neural network weights follow a zero-mean Gaussian distribution N(0, sigma^2).
    Standard uniform INT4 (evenly spaced grid) places most buckets in empty tails and clumps
    the high-density bell curve into 2-3 bins, destroying precision.
    
    NF4 defines 16 information-theoretically optimal quantiles such that each bin has
    EQUAL probability mass under N(0, 1), minimizing quantization error.
"""

from typing import Tuple
import torch
import torch.nn as nn

# The exact 16 quantile values for NF4 spanning [-1.0, 1.0]
# Computed from the inverse CDF (quantile function) of standard normal distribution:
# Q(i) = 0.5 * ( q_{N(0,1)}(2i / 32) + q_{N(0,1)}((2i+1)/32) )
NF4_QUANTILES = torch.tensor(
    [
        -1.00000000,
        -0.69619280,
        -0.52507305,
        -0.39491665,
        -0.28444138,
        -0.18477343,
        -0.09105004,
        0.00000000,
        0.07958030,
        0.16093020,
        0.24611230,
        0.33791524,
        0.44070983,
        0.56261700,
        0.72295684,
        1.00000000,
    ],
    dtype=torch.float32,
)


def quantize_nf4_blockwise(
    weight: torch.Tensor,
    block_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantizes a 2D weight tensor into 4-bit NF4 indices using block-wise scaling.

    Args:
        weight: Float tensor of shape (out_features, in_features)
        block_size: Number of elements per independent quantization block (default: 64)

    Returns:
        indices: 4-bit indices (0-15) stored in torch.uint8
        scales: Absolute maximum scales per block for reconstruction
    """
    orig_shape = weight.shape
    flat_weight = weight.contiguous().view(-1)
    num_elements = flat_weight.numel()

    # Pad if not evenly divisible by block_size
    pad_len = (block_size - (num_elements % block_size)) % block_size
    if pad_len > 0:
        flat_weight = torch.cat([flat_weight, torch.zeros(pad_len, device=weight.device)])

    # Reshape into blocks: (num_blocks, block_size)
    blocks = flat_weight.view(-1, block_size)

    # Absolute max scale per block (with epsilon guard)
    scales = blocks.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)

    # Normalize blocks into [-1.0, 1.0]
    normalized_blocks = blocks / scales  # Shape: (num_blocks, block_size)

    # Map each value to nearest NF4 quantile
    # Compute squared difference to all 16 quantiles: (num_blocks, block_size, 16)
    quantiles = NF4_QUANTILES.to(weight.device)
    diffs = (normalized_blocks.unsqueeze(-1) - quantiles).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)

    return indices, scales.squeeze(-1)


def dequantize_nf4_blockwise(
    indices: torch.Tensor,
    scales: torch.Tensor,
    orig_shape: torch.Size,
    block_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantizes NF4 indices back to high-precision float tensor.

    Formula:
        W_reconstructed = NF4_QUANTILES[indices] * scale
    """
    quantiles = NF4_QUANTILES.to(device=indices.device, dtype=dtype)
    num_blocks = indices.shape[0]

    # Look up float values from quantiles table
    normalized_values = quantiles[indices.long()]  # (num_blocks, block_size)

    # Multiply by scale per block
    reconstructed = normalized_values * scales.unsqueeze(-1).to(dtype)

    # Flatten and truncate back to original element count
    total_elements = orig_shape.numel()
    flat_out = reconstructed.view(-1)[:total_elements]

    return flat_out.view(orig_shape)


class QLoRALinear(nn.Module):
    """Linear layer with 4-bit NF4 quantized base weights and FP16 LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        block_size: int = 64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = lora_alpha / rank
        self.block_size = block_size

        # Create dummy float weights to initialize NF4 quantization
        dummy_weight = torch.randn(out_features, in_features) * (1.0 / (in_features**0.5))
        indices, scales = quantize_nf4_blockwise(dummy_weight, block_size=block_size)

        # Store quantized representations (non-trainable)
        self.register_buffer("nf4_indices", indices)
        self.register_buffer("nf4_scales", scales)

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """On-the-fly dequantization in GPU SRAM during forward pass."""
        # 1. Dequantize frozen base weight to matching dtype
        weight_dequant = dequantize_nf4_blockwise(
            self.nf4_indices,
            self.nf4_scales,
            orig_shape=torch.Size((self.out_features, self.in_features)),
            block_size=self.block_size,
            dtype=x.dtype,
        )

        # 2. Base model computation: x @ W_0.T
        base_out = torch.matmul(x, weight_dequant.t())

        # 3. LoRA adapter computation: (x @ A.T) @ B.T * scaling
        lora_out = torch.matmul(x, self.lora_A.t())
        lora_out = torch.matmul(lora_out, self.lora_B.t()) * self.scaling

        return base_out + lora_out
