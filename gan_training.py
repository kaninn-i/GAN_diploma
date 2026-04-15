import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from gan_model import weights_init
from dataset_utils import CropDataset


def train_gan(
    data_path,
    save_path="gan_weights",
    epochs=50,
    batch_size=64,
    noise_dim=100,
    lr=0.0002,
    img_size=64,
    device="cuda",
    model_type="dcgan"   # ← ДОБАВИЛИ
):

    os.makedirs(save_path, exist_ok=True)

    # ==============================
    # выбор модели
    # ==============================

    if model_type.lower() == "ssd":
        from ssd_model import Generator, Discriminator
        print("Using SSD GAN")
    else:
        from gan_model import Generator, Discriminator
        print("Using DCGAN")

    # ==============================
    # dataset
    # ==============================

    dataset = CropDataset(data_path, img_size=img_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    
    if len(dataloader) == 0:
        print("Недостаточно данных для обучения GAN")
        return

    print(f"Всего объектов: {len(dataset)}")

    # ==============================
    # models
    # ==============================

    generator = Generator(latent_dim=noise_dim).to(device)
    discriminator = Discriminator().to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # ==============================
    # optimizers
    # ==============================

    criterion = nn.BCELoss()

    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )

    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )

    fixed_noise = torch.randn(16, noise_dim, 1, 1, device=device)

    # ==============================
    # training
    # ==============================

    for epoch in range(epochs):

        epoch_d = 0
        epoch_g = 0

        for i, (real_imgs, _) in enumerate(dataloader):

            real_imgs = real_imgs.to(device)
            batch_size_curr = real_imgs.size(0)

            real_targets = torch.full(
                (batch_size_curr,),
                0.9,
                device=device
            )

            fake_targets = torch.full(
                (batch_size_curr,),
                0.1,
                device=device
            )

            # ======================
            # Train Discriminator
            # ======================

            optimizer_D.zero_grad()

            z = torch.randn(
                batch_size_curr,
                noise_dim,
                1,
                1,
                device=device
            )

            fake_imgs = generator(z)

            real_loss = criterion(
                discriminator(real_imgs),
                real_targets
            )

            fake_loss = criterion(
                discriminator(fake_imgs.detach()),
                fake_targets
            )

            d_loss = real_loss + fake_loss

            d_loss.backward()
            optimizer_D.step()

            # ======================
            # Train Generator
            # ======================

            optimizer_G.zero_grad()

            z = torch.randn(
                batch_size_curr,
                noise_dim,
                1,
                1,
                device=device
            )

            fake_imgs = generator(z)

            g_loss = criterion(
                discriminator(fake_imgs),
                real_targets
            )

            g_loss.backward()
            optimizer_G.step()

            epoch_d += d_loss.item()
            epoch_g += g_loss.item()

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"D: {epoch_d/max(len(dataloader), 1):.4f} "
            f"G: {epoch_g/len(dataloader):.4f}"
        )

        # preview
        with torch.no_grad():

            fake_samples = generator(fixed_noise)
            fake_samples = (fake_samples + 1) / 2

            save_image(
                fake_samples,
                os.path.join(save_path, f"epoch_{epoch+1}.png"),
                nrow=4
            )

        # save weights
        torch.save(
            generator.state_dict(),
            os.path.join(save_path, "generator.pth")
        )

        torch.save(
            discriminator.state_dict(),
            os.path.join(save_path, "discriminator.pth")
        )

    print("Обучение завершено.")
    return {
        "g_loss": epoch_g / len(dataloader),
        "d_loss": epoch_d / len(dataloader)
    }