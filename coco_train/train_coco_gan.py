# train_coco_gan.py
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from ssd_model import Generator, Discriminator

# Конфигурация по умолчанию
IMG_SIZE = 128
BATCH_SIZE = 64
LATENT_DIM = 128
EPOCHS = 300
LR_G = 0.0002
LR_D = 0.0004
EMA_DECAY = 0.999
CHECKPOINT_DIR = "coco_checkpoints"
OUTPUT_WEIGHTS = "coco_pretrained_128.pth"

# Аугментация (как в gan_train.py)
def diff_augment(x, policy='color,translation,cutout', p=0.5):
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
        y1 = torch.randint(0, x.size(2)-mask_size, (1,)).item()
        x1 = torch.randint(0, x.size(3)-mask_size, (1,)).item()
        mask[:, :, y1:y1+mask_size, x1:x1+mask_size] = 0
        x = x * mask
    return x

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, img_size=128):
        self.samples = []
        for f in os.listdir(root_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.samples.append(os.path.join(root_dir, f))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = cv2.imread(self.samples[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)

def save_checkpoint(epoch, generator, discriminator, opt_G, opt_D, sched_G, sched_D, path):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'opt_D_state_dict': opt_D.state_dict(),
        'sched_G_state_dict': sched_G.state_dict(),
        'sched_D_state_dict': sched_D.state_dict(),
    }, path)

def load_checkpoint(path, generator, discriminator, opt_G, opt_D, sched_G, sched_D, device):
    checkpoint = torch.load(path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
    opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
    sched_G.load_state_dict(checkpoint['sched_G_state_dict'])
    sched_D.load_state_dict(checkpoint['sched_D_state_dict'])
    return checkpoint['epoch']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='coco_objects_128/train')
    parser.add_argument('--resume', type=str, default=None, help='Путь к checkpoint для возобновления')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = COCODataset(args.data, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    print(f"Размер датасета: {len(dataset)} изображений")

    generator = Generator(latent_dim=LATENT_DIM, img_size=IMG_SIZE).to(device)
    discriminator = Discriminator(img_channels=3, img_size=IMG_SIZE).to(device)

    opt_G = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(0.0, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.0, 0.999))
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=EPOCHS, eta_min=LR_G*0.1)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=EPOCHS, eta_min=LR_D*0.1)

    start_epoch = 0

    if args.resume:
        print(f"Возобновление из {args.resume}")
        start_epoch = load_checkpoint(args.resume, generator, discriminator, opt_G, opt_D, sched_G, sched_D, device)
        start_epoch += 1

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # EMA
    ema_generator = Generator(latent_dim=LATENT_DIM, img_size=IMG_SIZE).to(device)
    ema_generator.load_state_dict(generator.state_dict())
    ema_generator.eval()

    criterion = nn.BCEWithLogitsLoss()  # не используется с hinge loss

    fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)

    for epoch in range(start_epoch, EPOCHS):
        generator.train()
        discriminator.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for real_imgs in pbar:
            real_imgs = real_imgs.to(device)
            batch_curr = real_imgs.size(0)

            # Discriminator
            opt_D.zero_grad()
            z = torch.randn(batch_curr, LATENT_DIM, 1, 1, device=device)
            fake_imgs = generator(z)

            real_aug = diff_augment(real_imgs)
            fake_aug = diff_augment(fake_imgs.detach())

            real_logits = discriminator(real_aug)
            fake_logits = discriminator(fake_aug)
            d_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
            d_loss.backward()
            opt_D.step()

            # Generator
            opt_G.zero_grad()
            z = torch.randn(batch_curr, LATENT_DIM, 1, 1, device=device)
            fake_imgs = generator(z)
            fake_aug = diff_augment(fake_imgs)
            fake_logits = discriminator(fake_aug)
            g_loss = -fake_logits.mean()
            g_loss.backward()
            opt_G.step()

            # EMA
            with torch.no_grad():
                for ema_p, p in zip(ema_generator.parameters(), generator.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(p.data, alpha=1 - EMA_DECAY)

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            pbar.set_postfix(D=f"{d_loss.item():.4f}", G=f"{g_loss.item():.4f}")

        sched_G.step()
        sched_D.step()

        avg_d = epoch_d_loss / len(dataloader)
        avg_g = epoch_g_loss / len(dataloader)
        print(f"Epoch {epoch+1} | D_loss: {avg_d:.4f}, G_loss: {avg_g:.4f}")

        # Сохраняем чекпойнт каждые 10 эпох
        if (epoch+1) % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(epoch, generator, discriminator, opt_G, opt_D, sched_G, sched_D, checkpoint_path)
            # Сохраняем также итоговые веса
            torch.save(ema_generator.state_dict(), OUTPUT_WEIGHTS)
            # Визуализация
            with torch.no_grad():
                samples = ema_generator(fixed_noise)
                samples = (samples + 1) / 2
                save_image(samples, f"sample_epoch_{epoch+1}.png", nrow=4)

    # Финальное сохранение
    torch.save(ema_generator.state_dict(), OUTPUT_WEIGHTS)
    print(f"Обучение завершено. Веса сохранены в {OUTPUT_WEIGHTS}")

if __name__ == "__main__":
    main()