from base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.module import Attention, PreNorm, FeedForward


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CustomResNet18(ResNet):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], 1000)
        weights = ResNet18_Weights.DEFAULT
        self.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
        num_ftrs = self.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x, debug=False):

        x = self.conv1(x)
        if debug:
            print(x.shape)

        x = self.bn1(x)
        if debug:
            print(x.shape)

        x = self.relu(x)
        if debug:
            print(x.shape)

        x = self.maxpool(x)
        if debug:
            print(x.shape)

        x = self.layer1(x)
        if debug:
            print(x.shape)

        x = self.layer2(x)
        if debug:
            print(x.shape)

        x = self.layer3(x)
        if debug:
            print(x.shape)

        x = self.layer4(x)
        if debug:
            print(x.shape)

        x = self.avgpool(x)
        if debug:
            print(x.shape)

        x = torch.flatten(x, 1)
        if debug:
            print(x.shape)

        x = self.fc(x)
        if debug:
            print(x.shape)

        return x


class CustomResNet50(ResNet):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], 1000)
        weights = ResNet50_Weights.DEFAULT
        self.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
        num_ftrs = self.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x, debug=False):

        x = self.conv1(x)
        if debug:
            print(x.shape)

        x = self.bn1(x)
        if debug:
            print(x.shape)

        x = self.relu(x)
        if debug:
            print(x.shape)

        x = self.maxpool(x)
        if debug:
            print(x.shape)

        x = self.layer1(x)
        if debug:
            print(x.shape)

        x = self.layer2(x)
        if debug:
            print(x.shape)

        x = self.layer3(x)
        if debug:
            print(x.shape)

        x = self.layer4(x)
        if debug:
            print(x.shape)

        x = self.avgpool(x)
        if debug:
            print(x.shape)

        x = torch.flatten(x, 1)
        if debug:
            print(x.shape)

        x = self.fc(x)
        if debug:
            print(x.shape)

        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        num_frames,
        dim=192,
        depth=4,
        heads=3,
        pool="cls",
        in_channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        scale_dim=4,
    ):
        super().__init__()

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size**2
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, num_patches + 1, dim)
        )
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(
            dim, depth, heads, dim_head, dim * scale_dim, dropout
        )

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(
            dim, depth, heads, dim_head, dim * scale_dim, dropout
        )

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, "() n d -> b t n d", b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, : (n + 1)]
        x = self.dropout(x)

        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], "(b t) ... -> b t ...", b=b)

        cls_temporal_tokens = repeat(self.temporal_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        return self.mlp_head(x)
