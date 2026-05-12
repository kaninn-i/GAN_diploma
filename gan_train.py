import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from gan_model import Generator, Discriminator, weights_init
from ssd_model import Generator as SSDGenerator, Discriminator as SSDDiscriminator
from dcgan_sn_model import (
    Generator as DCGANSNGenerator,
    Discriminator as DCGANSNDiscriminator,
    weights_init_generator as dcgan_sn_weights_init_g,
)
from ssd_lite_model import Generator as SSDLiteGenerator, Discriminator as SSDLiteDiscriminator
import cv2


def _build_aug_transforms(augment, n_samples):
    """При малом N слабее аугментация (без ColorJitter), чтобы не уводить распределение."""
    if not augment:
        return None
    flip = transforms.RandomHorizontalFlip(p=0.5 if n_samples >= 50 else 0.3)
    if n_samples < 200:
        return transforms.Compose([flip])
    return transforms.Compose(
        [
            flip,
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05
            ),
        ]
    )


def _diff_augment_policy_for_n(n):
    """Меньше агрессивных diff_augment при малом датасете."""
    if n < 100:
        return "color", 0.35
    if n < 200:
        return "color,translation", 0.45
    return "color,translation,cutout", 0.5


class CropImageDataset(torch.utils.data.Dataset):
    def __init__(self, class_dir, img_size=64, augment=True):
        super().__init__()
        self.samples = []
        for f in os.listdir(class_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                self.samples.append(os.path.join(class_dir, f))
        self.img_size = img_size
        self.augment = augment
        n = len(self.samples)
        self.resize_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
            ]
        )
        self.aug_transform = _build_aug_transforms(augment, n)
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = cv2.imread(self.samples[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.resize_transform(img)
        if self.aug_transform:
            img = self.aug_transform(img)
        img = self.to_tensor(img)
        return img, 0


def diff_augment(x, policy="", p=0.5):
    if not policy:
        return x
    if "color" in policy and torch.rand(1, device=x.device) < p:
        x = x * torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
        x = x + torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(-0.1, 0.1)
    if "translation" in policy and torch.rand(1, device=x.device) < p:
        shift_x = torch.randint(-4, 5, (1,), device=x.device).item()
        shift_y = torch.randint(-4, 5, (1,), device=x.device).item()
        x = torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))
    if "cutout" in policy and torch.rand(1, device=x.device) < p:
        mask_size = max(1, x.size(2) // 4)
        mask = torch.ones_like(x)
        y1 = torch.randint(0, max(1, x.size(2) - mask_size), (1,), device=x.device).item()
        x1 = torch.randint(0, max(1, x.size(3) - mask_size), (1,), device=x.device).item()
        mask[:, :, y1 : y1 + mask_size, x1 : x1 + mask_size] = 0
        x = x * mask
    return x


def _r1_penalty(real_img, discriminator):
    real_img = real_img.detach().requires_grad_(True)
    logits = discriminator(real_img)
    grad = torch.autograd.grad(
        outputs=logits.sum(),
        inputs=real_img,
        create_graph=True,
        only_inputs=True,
    )[0]
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


def _actual_generator_output_size(img_size: int) -> int:
    """
    Возвращает фактический размер выхода генератора SSD/SSD-lite.
    Генератор удваивает от 4 до тех пор, пока размер < img_size,
    поэтому реальный выход — ближайшая степень 2, >= img_size.
    """
    s = 4
    while s < img_size:
        s *= 2
    return s


def train_gan(
    class_dir,
    save_dir,
    epochs=200,
    batch_size=64,
    latent_dim=128,
    lr_G=0.0002,
    lr_D=0.0004,
    device="cuda",
    model_type="ssd",
    img_size=64,
    use_ema=True,
    ema_decay=0.999,
    augment_policy=None,
    augment_policy_p=None,
    pretrained_generator_path=None,
    progress_callback=None,
    n_critic=1,
    r1_gamma=10.0,
    adam_betas=(0.5, 0.999),
    save_best=True,
):
    mt = model_type.lower()

    # DCGAN зафиксирован на 64
    if mt in ("dcgan", "dcgan_sn"):
        img_size = 64

    # Для SSD-моделей приводим img_size к ближайшей степени 2
    # чтобы размер датасета совпадал с выходом генератора
    if mt in ("ssd", "ssd_lite"):
        img_size = _actual_generator_output_size(img_size)

    os.makedirs(save_dir, exist_ok=True)
    dataset = CropImageDataset(class_dir, img_size=img_size, augment=True)
    n_samples = len(dataset)
    batch_size = min(batch_size, n_samples)

    if augment_policy is None:
        augment_policy, augment_policy_p = _diff_augment_policy_for_n(n_samples)
    if augment_policy_p is None:
        _, augment_policy_p = _diff_augment_policy_for_n(n_samples)

    # Смягчаем R1 для малых датасетов, чтобы не убивать градиенты
    if n_samples < 100 and r1_gamma > 5.0:
        r1_gamma = 5.0
    if n_samples < 50 and r1_gamma > 1.0:
        r1_gamma = 1.0

    if n_samples < 25:
        print(f"[WARN] {class_dir} has less than 25 images, skipping training.")
        return None

    drop_last = n_samples > batch_size * 2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=drop_last,
    )
    if len(dataloader) == 0:
        print(f"[WARN] DataLoader empty (batch={batch_size}). Not enough data.")
        return None

    if mt == "ssd":
        generator = SSDGenerator(latent_dim, img_size=img_size).to(device)
        discriminator = SSDDiscriminator(img_channels=3, img_size=img_size).to(device)
    elif mt == "ssd_lite":
        generator = SSDLiteGenerator(latent_dim, img_size=img_size).to(device)
        discriminator = SSDLiteDiscriminator(img_channels=3, img_size=img_size).to(device)
    elif mt == "dcgan_sn":
        generator = DCGANSNGenerator(latent_dim).to(device)
        discriminator = DCGANSNDiscriminator().to(device)
    else:
        generator = Generator(latent_dim).to(device)
        discriminator = Discriminator().to(device)

    if mt == "dcgan_sn":
        generator.apply(dcgan_sn_weights_init_g)
    else:
        generator.apply(weights_init)
    if mt != "dcgan_sn":
        discriminator.apply(weights_init)

    if pretrained_generator_path and os.path.exists(pretrained_generator_path):
        print(f"Loading pretrained generator from {pretrained_generator_path}")
        state_dict = torch.load(pretrained_generator_path, map_location=device)
        generator.load_state_dict(state_dict, strict=False)

    ema_generator = None
    if use_ema:
        if mt in ("ssd", "ssd_lite"):
            ema_generator = type(generator)(latent_dim, img_size=img_size).to(device)
        else:
            ema_generator = type(generator)(latent_dim).to(device)
        ema_generator.load_state_dict(generator.state_dict())
        ema_generator.eval()

    opt_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=adam_betas)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=adam_betas)

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=epochs, eta_min=lr_G * 0.1
    )
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_D, T_max=epochs, eta_min=lr_D * 0.1
    )

    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)
    n_critic = max(1, int(n_critic))

    def d_hinge_loss(real_logits, fake_logits):
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()

    def g_hinge_loss(fake_logits):
        return -fake_logits.mean()

    best_g = float("inf")
    best_epoch = -1

    # ── история лоссов по эпохам ──
    epoch_history: list[dict] = []

    for epoch in range(epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        n_steps = 0

        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(device)
            batch_curr = real_imgs.size(0)

            for critic_idx in range(n_critic):
                opt_D.zero_grad(set_to_none=True)
                z = torch.randn(batch_curr, latent_dim, 1, 1, device=device)
                with torch.no_grad():
                    fake_imgs = generator(z)

                real_aug = diff_augment(real_imgs, augment_policy, augment_policy_p)
                fake_aug = diff_augment(fake_imgs.detach(), augment_policy, augment_policy_p)

                real_logits = discriminator(real_aug)
                fake_logits = discriminator(fake_aug)
                d_loss = d_hinge_loss(real_logits, fake_logits)

                if r1_gamma and r1_gamma > 0.0 and critic_idx == n_critic - 1:
                    d_loss = d_loss + r1_gamma * _r1_penalty(real_imgs, discriminator)

                d_loss.backward()
                opt_D.step()

                epoch_d_loss += d_loss.item()
                n_steps += 1

            opt_G.zero_grad(set_to_none=True)
            z = torch.randn(batch_curr, latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)
            fake_aug = diff_augment(fake_imgs, augment_policy, augment_policy_p)
            fake_logits = discriminator(fake_aug)
            g_loss = g_hinge_loss(fake_logits)
            g_loss.backward()
            opt_G.step()

            if use_ema and ema_generator is not None:
                with torch.no_grad():
                    for ema_param, param in zip(
                        ema_generator.parameters(), generator.parameters()
                    ):
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

            epoch_g_loss += g_loss.item()

        scheduler_G.step()
        scheduler_D.step()

        denom_d = max(1, n_steps)
        denom_g = max(1, len(dataloader))
        avg_d = epoch_d_loss / denom_d
        avg_g = epoch_g_loss / denom_g
        print(f"Epoch {epoch+1}/{epochs} | D loss: {avg_d:.4f} | G loss: {avg_g:.4f}")

        # ── фиксируем историю ──
        epoch_history.append(
            {"epoch": epoch + 1, "g_loss": round(avg_g, 5), "d_loss": round(avg_d, 5)}
        )

        if save_best and avg_g < best_g - 1e-6:
            best_g = avg_g
            best_epoch = epoch + 1
            torch.save(generator.state_dict(), os.path.join(save_dir, "generator_best.pth"))
            torch.save(
                discriminator.state_dict(), os.path.join(save_dir, "discriminator_best.pth")
            )
            if use_ema and ema_generator is not None:
                torch.save(
                    ema_generator.state_dict(),
                    os.path.join(save_dir, "generator_ema_best.pth"),
                )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            viz_gen = (
                ema_generator if use_ema and ema_generator is not None else generator
            )
            with torch.no_grad():
                samples = viz_gen(fixed_noise)
                samples = (samples + 1) / 2
            preview_path = os.path.join(save_dir, f"epoch_{epoch + 1}.png")
            save_image(samples, preview_path, nrow=4)

        if progress_callback:
            progress_value = float((epoch + 1) / epochs)
            progress_callback(
                progress_value,
                f"Обучение эпоха {epoch + 1}/{epochs} | G: {avg_g:.4f} | D: {avg_d:.4f}",
            )

        torch.save(generator.state_dict(), os.path.join(save_dir, "generator.pth"))
        torch.save(discriminator.state_dict(), os.path.join(save_dir, "discriminator.pth"))
        if use_ema and ema_generator is not None:
            torch.save(ema_generator.state_dict(), os.path.join(save_dir, "generator_ema.pth"))

    if save_best and best_epoch > 0:
        for src, dst in [
            ("generator_best.pth", "generator.pth"),
            ("discriminator_best.pth", "discriminator.pth"),
        ]:
            src_path = os.path.join(save_dir, src)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, os.path.join(save_dir, dst))
        best_ema = os.path.join(save_dir, "generator_ema_best.pth")
        if use_ema and os.path.isfile(best_ema):
            shutil.copy2(best_ema, os.path.join(save_dir, "generator_ema.pth"))

    return {
        "g_loss": avg_g,
        "d_loss": avg_d,
        "best_g_loss": best_g,
        "best_epoch": best_epoch,
        "epoch_history": epoch_history,   # ← новое поле
        "img_size_used": img_size,         # ← для диагностики
    }
