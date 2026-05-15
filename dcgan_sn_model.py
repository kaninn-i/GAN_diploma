"""
DCGAN-вариант с spectral norm в дискриминаторе (без BatchNorm в D)
и без BatchNorm перед Tanh в генераторе — для сравнения с gan_model.py.
"""
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()

        def sn_conv(in_c, out_c, stride):
            return spectral_norm(
                nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False)
            )

        self.main = nn.Sequential(
            sn_conv(img_channels, 64, 2),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(64, 128, 2),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(128, 256, 2),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(256, 512, 2),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(512, 1, 1),
        )

    def forward(self, img):
        return self.main(img).view(-1)


def weights_init_generator(m):
    classname = m.__class__.__name__
    if classname.find("ConvTranspose") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
