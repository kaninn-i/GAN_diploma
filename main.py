import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Параметры
batch_size = 64
image_size = 64   # будем ресайзить до 64x64, хотя CIFAR-10 32x32, но для генератора лучше 64
latent_dim = 100   # размер случайного шума
n_epochs = 50
lr = 0.0002
beta1 = 0.5        # для Adam
sample_interval = 400  # через сколько итераций сохранять примеры

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Создаём папку для сохранения результатов
os.makedirs("gan_output", exist_ok=True)
os.makedirs("gan_output/images", exist_ok=True)
os.makedirs("gan_output/models", exist_ok=True)

# Трансформации: ресайз до 64x64, тензор, нормализация в диапазон [-1, 1] (так лучше для tanh на выходе генератора)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # для 3 каналов
])

class Generator(nn.Module):
    def __init__(self, latent_dim, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # вход: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # состояние: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # состояние: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # состояние: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # состояние: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # выход: (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # вход: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # состояние: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # состояние: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # состояние: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # состояние: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

criterion = nn.BCELoss()

if __name__ == "__main__":
    # Загружаем CIFAR-10
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Оставляем только один класс, например, класс 1 (автомобиль)
    class_to_keep = 1  # автомобиль
    indices = [i for i, (_, label) in enumerate(full_dataset) if label == class_to_keep]
    subset = torch.utils.data.Subset(full_dataset, indices)

    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    print(f"Количество изображений автомобилей: {len(subset)}")
    
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Создаём метки для реальных и поддельных изображений (будем заполнять позже)
    real_label = 1.0
    fake_label = 0.0

    # Оптимизаторы
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)  # фиксированный шум для визуализации прогресса

    for epoch in range(n_epochs):
        for i, (real_imgs, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # === Обучаем дискриминатор ===
            optimizer_D.zero_grad()

            # Реальные изображения
            label_real = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output_real = discriminator(real_imgs).view(-1)
            loss_D_real = criterion(output_real, label_real)
            loss_D_real.backward()

            # Поддельные изображения (генерируем)
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = generator(noise)
            label_fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
            output_fake = discriminator(fake_imgs.detach()).view(-1)  # detach чтобы градиенты не шли в генератор
            loss_D_fake = criterion(output_fake, label_fake)
            loss_D_fake.backward()

            loss_D = loss_D_real + loss_D_fake
            optimizer_D.step()

            # === Обучаем генератор ===
            optimizer_G.zero_grad()
            # Генерируем новые поддельные изображения (те же noise, но теперь мы хотим обмануть D)
            output_fake = discriminator(fake_imgs).view(-1)
            # Для генератора цель — чтобы D думал, что fake — реальные
            label_real_for_G = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            loss_G = criterion(output_fake, label_real_for_G)
            loss_G.backward()
            optimizer_G.step()

            # === Визуализация и логирование ===
            if i % sample_interval == 0:
                print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
                    f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

                # Сохраняем изображения, сгенерированные из фиксированного шума
                with torch.no_grad():
                    fake_fixed = generator(fixed_noise).detach().cpu()
                # Денормализуем (из [-1,1] в [0,1])
                fake_fixed = (fake_fixed + 1) / 2.0
                save_image(fake_fixed, f"gan_output/images/epoch_{epoch}_batch_{i}.png", nrow=8, normalize=False)

        # Сохраняем модели после каждой эпохи
        torch.save(generator.state_dict(), f"gan_output/models/generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"gan_output/models/discriminator_epoch_{epoch}.pth")