import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.cuda.amp import GradScaler, autocast

from gan.generator import Generator
from gan.discriminator import Discriminator
from gan.dataset import CropDataset


def train_gan(
    data_path,
    save_path="gan_weights",
    epochs=50,
    batch_size=64,
    noise_dim=100,
    lr=0.0002,
    img_size=64,
    device="cuda"
):
    os.makedirs(save_path, exist_ok=True)

    dataset = CropDataset(data_path, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    num_classes = len(dataset.class_to_idx)
    print(f"Найдено классов: {num_classes}")
    print(f"Всего объектов: {len(dataset)}")

    generator = Generator(noise_dim=noise_dim, num_classes=num_classes).to(device)
    discriminator = Discriminator(num_classes=num_classes).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    scaler_G = GradScaler()
    scaler_D = GradScaler()

    fixed_noise = torch.randn(16, noise_dim, 1, 1, device=device)
    fixed_labels = torch.randint(0, num_classes, (16,), device=device)

    for epoch in range(epochs):
        for i, (real_imgs, labels) in enumerate(dataloader):

            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            batch_size_curr = real_imgs.size(0)

            real_targets = torch.ones(batch_size_curr, device=device)
            fake_targets = torch.zeros(batch_size_curr, device=device)

            # =======================
            #  Train Discriminator
            # =======================

            optimizer_D.zero_grad()

            noise = torch.randn(batch_size_curr, noise_dim, 1, 1, device=device)

            with autocast():
                fake_imgs = generator(noise, labels)

                real_loss = criterion(discriminator(real_imgs, labels), real_targets)
                fake_loss = criterion(discriminator(fake_imgs.detach(), labels), fake_targets)

                d_loss = real_loss + fake_loss

            scaler_D.scale(d_loss).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            # =======================
            #  Train Generator
            # =======================

            optimizer_G.zero_grad()

            noise = torch.randn(batch_size_curr, noise_dim, 1, 1, device=device)

            with autocast():
                fake_imgs = generator(noise, labels)
                g_loss = criterion(discriminator(fake_imgs, labels), real_targets)

            scaler_G.scale(g_loss).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            if i % 50 == 0:
                print(
                    f"[Epoch {epoch+1}/{epochs}] "
                    f"[Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] "
                    f"[G loss: {g_loss.item():.4f}]"
                )

        # сохраняем примеры генерации
        with torch.no_grad():
            fake_samples = generator(fixed_noise, fixed_labels)
            fake_samples = (fake_samples + 1) / 2  # обратно в [0,1]
            save_image(fake_samples, os.path.join(save_path, f"epoch_{epoch+1}.png"), nrow=4)

        # сохраняем веса
        torch.save(generator.state_dict(), os.path.join(save_path, "generator.pth"))
        torch.save(discriminator.state_dict(), os.path.join(save_path, "discriminator.pth"))

    print("Обучение завершено.")