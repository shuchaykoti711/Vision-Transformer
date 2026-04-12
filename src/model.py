import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self,
                 color_channels,
                 patch_size,
                 embedding_dimension):
        super().__init__()
        self.patcher = nn.Conv2d(
            in_channels=color_channels,
            stride=patch_size,
            kernel_size=patch_size,
            out_channels=embedding_dimension,
            padding=0,
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)


class MSA(nn.Module):
    def __init__(self,
                 embedding_dimension,
                 number_heads,
                 attention_dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dimension)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=number_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

    def forward(self, x):
        normalized_x = self.layer_norm(x)
        attention_output, _ = self.multihead_attention(
            query=normalized_x,
            key=normalized_x,
            value=normalized_x,
            need_weights=False,
        )
        return x + attention_output


class MLP(nn.Module):
    def __init__(self,
                 embedding_dimension,
                 mlp_size,
                 mlp_dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dimension)
        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=embedding_dimension,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dimension),
            nn.Dropout(p=mlp_dropout),
        )

    def forward(self, x):
        normalized_x = self.layer_norm(x)
        return x + self.mlp_block(normalized_x)


class Encoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 numb_of_heads,
                 attn_dropout,
                 mlpsize,
                 mlp_drop):
        super().__init__()
        self.msa = MSA(
            embedding_dimension=embedding_dim,
            number_heads=numb_of_heads,
            attention_dropout=attn_dropout
        )
        self.mlp = MLP(
            embedding_dimension=embedding_dim,
            mlp_size=mlpsize,
            mlp_dropout=mlp_drop,
        )

    def forward(self, x):
        x = self.msa(x)
        x = self.mlp(x)
        return x


class ViT(nn.Module):
    def __init__(self,
                 colorchannels,
                 patchsize,
                 embeddingdimension,
                 embeddingdropout,
                 num_patches,
                 num_head,
                 num_transformation_layers,
                 multilpsize,
                 multilpdrop,
                 att_drop,
                 num_classes):
        super().__init__()
        self.patch_embedding = Embedding(
            color_channels=colorchannels,
            patch_size=patchsize,
            embedding_dimension=embeddingdimension,
        )
        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, embeddingdimension),
            requires_grad=True,
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embeddingdimension),
            requires_grad=True,
        )
        self.embedding_dropout = nn.Dropout(p=embeddingdropout)
        self.transformer_encoder = nn.Sequential(*[
            Encoder(embedding_dim=embeddingdimension,
                    numb_of_heads=num_head,
                    attn_dropout=att_drop,
                    mlpsize=multilpsize,
                    mlp_drop=multilpdrop) for _ in range(num_transformation_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embeddingdimension),
            nn.Linear(in_features=embeddingdimension,
                      out_features=num_classes),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x
