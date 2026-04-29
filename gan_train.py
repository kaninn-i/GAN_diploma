import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from gan_model import Generator, Discriminator, weights_init
from ssd_model import Generator as SSDGenerator, Discriminator as SSDDiscriminator
import cv2
import numpy as np


class CropImageDataset(torch.utils.data.Dataset):
    def __init__(self, class_dir, img_size=64, augment=True):
        super().__init__()
        self.samples = []
        for f in os.listdir(class_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.samples.append(os.path.join(class_dir, f))
        self.img_size = img_size
        self.augment = augment
        self.basic_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = cv2.imread(self.samples[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.basic_transform(img)
        if self.aug_transform:
            img = self.aug_transform(img)
        return img, 0


def diff_augment(x, policy='', p=0.5):
    # ... без изменений (как раньше)
    if not policy:
        return x
    if 'color' in policy and torch.rand(1) < p:
        x = x * torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
        x = x + torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(-0.1, 0.1)
    if 'translation' in policy and torch.rand(1) < p:
        shift_x = torch.randint(-4, 5, (1,)).item()
        shift_y = torch.randint(-4, 5, (1,)).item()
        x = torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))
    if 'cutout' in policy and torch.rand(1) < p:
        mask_size = x.size(2) // 4
        mask = torch.ones_like(x)
        y1 = torch.randint(0, x.size(2) - mask_size, (1,)).item()
        x1 = torch.randint(0, x.size(3) - mask_size, (1,)).item()
        mask[:, :, y1:y1+mask_size, x1:x1+mask_size] = 0
        x = x * mask
    return x


def train_gan(class_dir, save_dir, epochs=200, batch_size=64, latent_dim=128,
              lr_G=0.0002, lr_D=0.0004, device="cuda", model_type="ssd",
              img_size=64, use_ema=True, ema_decay=0.999,
              augment_policy='color,translation,cutout',
              pretrained_generator_path=None):
    
    os.makedirs(save_dir, exist_ok=True)
    dataset = CropImageDataset(class_dir, img_size=img_size, augment=True)
    if len(dataset) < 5:
        print(f"[WARN] {class_dir} has less than 5 images, skipping training.")
        return None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    if len(dataloader) == 0:
        print(f"[WARN] DataLoader empty (batch={batch_size}). Not enough data.")
        return None

    if model_type.lower() == "ssd":
        generator = SSDGenerator(latent_dim, img_size=img_size).to(device)
        discriminator = SSDDiscriminator(img_channels=3, img_size=img_size).to(device)
    else:
        generator = Generator(latent_dim).to(device)
        discriminator = Discriminator().to(device)
        
    if pretrained_generator_path and os.path.exists(pretrained_generator_path):
        print(f"Loading pretrained generator from {pretrained_generator_path}")
        state_dict = torch.load(pretrained_generator_path, map_location=device)
        generator.load_state_dict(state_dict, strict=False)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    ema_generator = None
    if use_ema:
        ema_generator = type(generator)(latent_dim, img_size=img_size).to(device) if model_type == "ssd" else type(generator)(latent_dim).to(device)
        ema_generator.load_state_dict(generator.state_dict())
        ema_generator.eval()

    opt_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0.0, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.0, 0.999))

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=epochs, eta_min=lr_G*0.1)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=epochs, eta_min=lr_D*0.1)

    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

    # Hinge loss for logits (no Sigmoid)
    def d_hinge_loss(real_logits, fake_logits):
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()

    def g_hinge_loss(fake_logits):
        return -fake_logits.mean()

    for epoch in range(epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(device)
            batch_curr = real_imgs.size(0)

            # Discriminator
            opt_D.zero_grad()
            z = torch.randn(batch_curr, latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)

            real_aug = diff_augment(real_imgs, augment_policy)
            fake_aug = diff_augment(fake_imgs.detach(), augment_policy)

            real_logits = discriminator(real_aug)
            fake_logits = discriminator(fake_aug)
            d_loss = d_hinge_loss(real_logits, fake_logits)
            d_loss.backward()
            opt_D.step()

            # Generator
            opt_G.zero_grad()
            z = torch.randn(batch_curr, latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)
            fake_aug = diff_augment(fake_imgs, augment_policy)
            fake_logits = discriminator(fake_aug)
            g_loss = g_hinge_loss(fake_logits)
            g_loss.backward()
            opt_G.step()

            if use_ema and ema_generator is not None:
                with torch.no_grad():
                    for ema_param, param in zip(ema_generator.parameters(), generator.parameters()):
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        scheduler_G.step()
        scheduler_D.step()

        avg_d = epoch_d_loss / len(dataloader)
        avg_g = epoch_g_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | D loss: {avg_d:.4f} | G loss: {avg_g:.4f}")

        if (epoch+1) % 10 == 0 or epoch == 0:
            viz_gen = ema_generator if use_ema else generator
            with torch.no_grad():
                samples = viz_gen(fixed_noise)
                samples = (samples + 1) / 2
                save_image(samples, os.path.join(save_dir, f"epoch_{epoch+1}.png"), nrow=4)

        torch.save(generator.state_dict(), os.path.join(save_dir, "generator.pth"))
        torch.save(discriminator.state_dict(), os.path.join(save_dir, "discriminator.pth"))
        if use_ema and ema_generator is not None:
            torch.save(ema_generator.state_dict(), os.path.join(save_dir, "generator_ema.pth"))

    return {"g_loss": avg_g, "d_loss": avg_d}