import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# from gan.generator import Generator
# from gan.discriminator import Discriminator
from gan.dataset import CropDataset

from gan.new_generator import Generator
from gan.new_discriminator import Discriminator


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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    num_classes = len(dataset.class_to_idx)
    print(f"Найдено классов: {num_classes}")
    print(f"Всего объектов: {len(dataset)}")

    # generator = Generator(noise_dim=noise_dim, num_classes=num_classes).to(device)
    generator = Generator().to(device)
    # discriminator = Discriminator(num_classes=num_classes).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCEWithLogitsLoss()
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, noise_dim, 1, 1, device=device)
    fixed_labels = torch.randint(0, num_classes, (16,), device=device)

    for epoch in range(epochs):
        
        epoch_d = 0
        epoch_g = 0
        
        for i, (real_imgs, labels) in enumerate(dataloader):

            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            
            real_imgs = real_imgs + 0.05 * torch.randn_like(real_imgs)
            real_imgs = torch.clamp(real_imgs, -1, 1)

            batch_size_curr = real_imgs.size(0)

            real_targets = torch.full((batch_size,), 0.9, device=device)
            fake_targets = torch.full((batch_size,), 0.1, device=device)

            # =======================
            #  Train Discriminator
            # =======================

            optimizer_D.zero_grad()

            # noise = torch.randn(batch_size_curr, noise_dim, 1, 1, device=device)
            z = torch.randn(batch_size, 100, 1, 1).to(device)

            # fake_imgs = generator(noise, labels)
            fake_imgs = generator(z)

            real_loss = criterion(discriminator(real_imgs, labels), real_targets)
            fake_loss = criterion(discriminator(fake_imgs.detach(), labels), fake_targets)

            d_loss = real_loss + fake_loss

            d_loss.backward()
            optimizer_G.step()
            # =======================
            #  Train Generator
            # =======================

            optimizer_G.zero_grad()

            noise = torch.randn(batch_size_curr, noise_dim, 1, 1, device=device)

            fake_imgs = generator(noise, labels)
            g_loss = criterion(discriminator(fake_imgs, labels), real_targets)

            g_loss.backward()
            optimizer_G.step()
            
            epoch_d += d_loss.item()
            epoch_g += g_loss.item()
            
            print(
                f"[Epoch {epoch+1}/{epochs}] "
                f"[Batch {i}/{len(dataloader)}] "
                f"[D loss: {d_loss.item():.4f}] "
                f"[G loss: {g_loss.item():.4f}]"
            )
            
        print(f"Epoch {epoch} avg D: {epoch_d/len(dataloader):.3f} avg G: {epoch_g/len(dataloader):.3f}")

        # сохраняем примеры генерации
        with torch.no_grad():
            fake_samples = generator(fixed_noise, fixed_labels)
            fake_samples = (fake_samples + 1) / 2  # обратно в [0,1]
            save_image(fake_samples, os.path.join(save_path, f"epoch_{epoch+1}.png"), nrow=4)

        # сохраняем веса
        torch.save(generator.state_dict(), os.path.join(save_path, "generator.pth"))
        torch.save(discriminator.state_dict(), os.path.join(save_path, "discriminator.pth"))

    print("Обучение завершено.")