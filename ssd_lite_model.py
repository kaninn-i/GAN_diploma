"""
Облегчённый StyleGAN-подобный генератор для сравнения с ssd_model.py:
слабее per-layer noise, меньше mapping, InstanceNorm в D, меньше Dropout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

LITE_STYLE_DIM = 128
LITE_MAP_DEPTH = 2
NOISE_SCALE = 0.25


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=128, style_dim=LITE_STYLE_DIM, depth=LITE_MAP_DEPTH):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = style_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim=LITE_STYLE_DIM, demodulate=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.modulation = spectral_norm(nn.Linear(style_dim, in_channels))

        nn.init.kaiming_normal_(self.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x, style):
        batch, in_ch, height, width = x.shape
        s = self.modulation(style).view(batch, 1, in_ch, 1, 1)
        weight = self.weight.unsqueeze(0) * s
        if self.demodulate:
            d = torch.rsqrt(weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8)
            weight = weight * d.view(batch, self.out_channels, 1, 1, 1)
        x_reshaped = x.view(1, batch * in_ch, height, width)
        weight_reshaped = weight.view(
            batch * self.out_channels, in_ch, self.kernel_size, self.kernel_size
        )
        out = F.conv2d(
            x_reshaped, weight_reshaped, padding=self.kernel_size // 2, groups=batch
        )
        _, _, h, w = out.shape
        return out.view(batch, self.out_channels, h, w)


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        return x + self.scale * noise * NOISE_SCALE


class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim=LITE_STYLE_DIM, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv = ModulatedConv2d(in_channels, out_channels, 3, style_dim)
        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, style, noise=None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x, style)
        x = self.noise(x, noise)
        x = x + self.bias
        x = self.activation(x)
        return x


class ToRGB(nn.Module):
    def __init__(self, in_channels, style_dim=LITE_STYLE_DIM):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style):
        x = self.conv(x, style)
        return x + self.bias


class Generator(nn.Module):
    def __init__(self, latent_dim=128, style_dim=LITE_STYLE_DIM, img_size=64):
        super().__init__()
        self.style_dim = style_dim
        self.mapping = MappingNetwork(latent_dim, style_dim, depth=LITE_MAP_DEPTH)
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4) * 0.1)

        blocks = []
        in_ch = 512
        out_chs = [512, 256, 128, 64, 32, 16]
        current_size = 4
        idx = 0
        while current_size < img_size:
            out_ch = out_chs[idx] if idx < len(out_chs) else 16
            blocks.append(StyleBlock(in_ch, out_ch, style_dim, upsample=True))
            in_ch = out_ch
            current_size *= 2
            idx += 1
        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = ToRGB(in_ch, style_dim)

    def forward(self, z):
        if z.dim() == 4:
            z = z.squeeze(-1).squeeze(-1)
        style = self.mapping(z)
        x = self.const.repeat(z.size(0), 1, 1, 1)
        for block in self.blocks:
            x = block(x, style)
        img = self.to_rgb(x, style)
        return torch.tanh(img)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=64):
        super().__init__()
        drop = 0.1
        layers = [
            spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
        ]
        if img_size >= 128:
            layers.extend(
                [
                    spectral_norm(nn.Conv2d(512, 512, 4, 2, 1, bias=False)),
                    nn.InstanceNorm2d(512, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(drop),
                ]
            )

        layers.append(spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False)))
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img).view(-1)


def weights_init(m):
    pass
