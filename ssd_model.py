# ssd_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ModulatedConv2d(nn.Module):
    """
    Слой модулированной свёртки, как в StyleGAN2, но упрощённый.
    Веса масштабируются в зависимости от вектора стиля.
    """
    def __init__(self, in_channels, out_channels, kernel_size, style_dim=100, demodulate=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate

        # Основные веса свёртки
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # Линейный слой для преобразования стиля в коэффициенты модуляции
        self.style_mod = spectral_norm(nn.Linear(style_dim, in_channels))

        # Инициализация весов
        nn.init.kaiming_normal_(self.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, style):
        batch, in_channel, height, width = x.shape

        # Получаем коэффициенты модуляции из стиля: [batch, in_channel]
        style = self.style_mod(style).view(batch, 1, in_channel, 1, 1)

        # Масштабируем веса: [out_ch, in_ch, k, k] -> [batch, out_ch, in_ch, k, k]
        weight = self.weight.unsqueeze(0) * style

        # Демодуляция (нормализация для стабильности)
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        # Преобразуем вход и веса для групповой свёртки
        x = x.view(1, batch * in_channel, height, width)
        weight = weight.view(batch * self.out_channels, in_channel, self.kernel_size, self.kernel_size)

        # Групповая свёртка с группами = batch (каждый пример обрабатывается своими весами)
        out = F.conv2d(x, weight, padding=self.kernel_size // 2, groups=batch)

        # Восстанавливаем форму
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channels, height, width)
        return out


class StyleBlock(nn.Module):
    """
    Блок генератора: модулированная свёртка + шум + bias + активация.
    """
    def __init__(self, in_channels, out_channels, style_dim=100, upsample=False):
        super().__init__()
        self.upsample = upsample

        self.conv = ModulatedConv2d(in_channels, out_channels, 3, style_dim)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.noise_weight = nn.Parameter(torch.zeros(1))
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, style, noise=None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv(x, style)

        # Добавление шума (опционально)
        if noise is not None:
            x = x + self.noise_weight * noise

        x = x + self.bias
        x = self.activate(x)
        return x


class ToRGB(nn.Module):
    """Преобразование признаков в RGB-изображение."""
    def __init__(self, in_channels, style_dim=100):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style):
        x = self.conv(x, style)
        x = x + self.bias
        return x


class Generator(nn.Module):
    """
    Генератор Spectral Style-DCGAN для изображений 64x64.
    Использует начальную константу 4x4, затем несколько StyleBlock с повышением разрешения.
    """
    def __init__(self, latent_dim=100, img_channels=3):
        super().__init__()
        self.latent_dim = latent_dim

        # Начальная константа
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4))

        # Блоки генератора
        self.block1 = StyleBlock(512, 512, latent_dim)          # 4x4 -> 4x4
        self.block2 = StyleBlock(512, 256, latent_dim, upsample=True)  # 8x8
        self.block3 = StyleBlock(256, 128, latent_dim, upsample=True)  # 16x16
        self.block4 = StyleBlock(128, 64, latent_dim, upsample=True)   # 32x32
        self.block5 = StyleBlock(64, 32, latent_dim, upsample=True)    # 64x64

        self.to_rgb = ToRGB(32, latent_dim)

        # Спектральная нормализация для всех обычных слоёв (уже применена в ModulatedConv2d через spectral_norm)
        # Но для остальных слоёв (нет) - здесь их нет.

    def forward(self, z):
        batch = z.shape[0]
        # Приводим z к двумерному виду (batch, latent_dim) независимо от исходной формы
        z = z.view(batch, -1)
        style = z

        # Повторяем константу для батча
        x = self.const.repeat(batch, 1, 1, 1)

        # Генерируем шум для каждого блока (можно не использовать)
        noise = None  # или torch.randn(...)

        x = self.block1(x, style, noise)
        x = self.block2(x, style, noise)
        x = self.block3(x, style, noise)
        x = self.block4(x, style, noise)
        x = self.block5(x, style, noise)

        img = self.to_rgb(x, style)
        return torch.tanh(img)


class Discriminator(nn.Module):
    """
    Дискриминатор с спектральной нормализацией для стабильности.
    Архитектура аналогична DCGAN, но с добавлением spectral_norm.
    """
    def __init__(self, img_channels=3):
        super().__init__()

        def conv_block(in_ch, out_ch, kernel, stride, padding):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            # 64x64 -> 32x32
            spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            conv_block(64, 128, 4, 2, 1),

            # 16x16 -> 8x8
            conv_block(128, 256, 4, 2, 1),

            # 8x8 -> 4x4
            conv_block(256, 512, 4, 2, 1),

            # 4x4 -> 1
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(-1)


def weights_init(m):
    """
    Инициализация весов для обычных слоёв (не используется в SSD, 
    так как спектральная нормализация и modulated conv имеют свою инициализацию).
    Оставлена для совместимости с gan_training.py.
    """
    pass  # ничего не делаем, инициализация уже встроена в слои