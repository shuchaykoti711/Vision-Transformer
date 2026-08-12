from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Split an image into patches and project each patch to an embedding vector."""

    def __init__(self, color_channels: int, patch_size: int, embedding_dimension: int) -> None:
        super().__init__()
        self.patcher = nn.Conv2d(
            in_channels=color_channels,
            out_channels=embedding_dimension,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)


class MultiheadSelfAttention(nn.Module):
    """Layer-normed multi-head self-attention with a residual connection."""

    def __init__(self, embedding_dimension: int, num_heads: int, attention_dropout: float) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dimension)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.layer_norm(x)
        attention_output, _ = self.attention(
            query=normalized, key=normalized, value=normalized, need_weights=False,
        )
        return x + attention_output


class MLPBlock(nn.Module):
    """Layer-normed feed-forward block with a residual connection."""

    def __init__(self, embedding_dimension: int, mlp_size: int, mlp_dropout: float) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dimension)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dimension, mlp_size),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(mlp_size, embedding_dimension),
            nn.Dropout(p=mlp_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.layer_norm(x))


class TransformerEncoderBlock(nn.Module):
    """A single transformer encoder block: self-attention followed by an MLP."""

    def __init__(self, embedding_dimension: int, num_heads: int,
                 attention_dropout: float, mlp_size: int, mlp_dropout: float) -> None:
        super().__init__()
        self.attention = MultiheadSelfAttention(embedding_dimension, num_heads, attention_dropout)
        self.mlp = MLPBlock(embedding_dimension, mlp_size, mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.attention(x))


class ViT(nn.Module):
    """Vision Transformer for image classification.

    Args mirror the fields in :class:`src.config.ModelConfig`.
    """

    def __init__(self,
                 color_channels: int = 3,
                 patch_size: int = 4,
                 embedding_dimension: int = 384,
                 embedding_dropout: float = 0.1,
                 num_patches: int = 64,
                 num_heads: int = 6,
                 num_layers: int = 7,
                 mlp_size: int = 1536,
                 mlp_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 num_classes: int = 100) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(color_channels, patch_size, embedding_dimension)
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dimension))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dimension))
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(
                embedding_dimension=embedding_dimension,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                mlp_size=mlp_size,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dimension),
            nn.Linear(embedding_dimension, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.encoder(x)
        return self.classifier(x[:, 0])


def build_model(model_config, num_patches: int) -> ViT:
    """Construct a :class:`ViT` from a :class:`src.config.ModelConfig`."""
    return ViT(
        color_channels=model_config.color_channels,
        patch_size=model_config.patch_size,
        embedding_dimension=model_config.embedding_dimension,
        embedding_dropout=model_config.embedding_dropout,
        num_patches=num_patches,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        mlp_size=model_config.mlp_size,
        mlp_dropout=model_config.mlp_dropout,
        attention_dropout=model_config.attention_dropout,
        num_classes=model_config.num_classes,
    )
