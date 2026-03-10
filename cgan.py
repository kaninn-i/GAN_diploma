import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from tqdm import tqdm

if __name__ == '__main__':
    # Параметры
    batch_size = 64
    image_size = 64
    latent_dim = 100
    n_classes = 10
    n_epochs = 200
    lr = 0.0002
    beta1 = 0.5
    epoch_interval = 5  # сохраняем картинку каждые 5 эпох

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Создаём папки для результатов
    os.makedirs("cgan_output", exist_ok=True)
    os.makedirs("cgan_output/images", exist_ok=True)
    os.makedirs("cgan_output/models", exist_ok=True)

    # Трансформации для изображений
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Загружаем CIFAR-10 (все классы)
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Определение моделей
    class Generator(nn.Module):
        def __init__(self, latent_dim, n_classes, ngf=64, nc=3, embedding_dim=50):
            super(Generator, self).__init__()
            self.label_emb = nn.Embedding(n_classes, embedding_dim)
            self.init_size = image_size // 16  # для 64x64 это 4
            self.fc = nn.Linear(latent_dim + embedding_dim, ngf * 8 * self.init_size * self.init_size)
            self.main = nn.Sequential(
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, z, labels):
            label_embedding = self.label_emb(labels)
            gen_input = torch.cat([z, label_embedding], dim=1)
            out = self.fc(gen_input)
            out = out.view(out.size(0), -1, self.init_size, self.init_size)
            img = self.main(out)
            return img

    class Discriminator(nn.Module):
        def __init__(self, n_classes, ndf=64, nc=3, img_size=64):
            super(Discriminator, self).__init__()
            self.label_embedding = nn.Embedding(n_classes, img_size * img_size)
            self.conv = nn.Sequential(
                nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(ndf * 8 * 4 * 4, 1),
                nn.Sigmoid()
            )

        def forward(self, img, labels):
            label_map = self.label_embedding(labels).view(img.size(0), 1, img.size(2), img.size(3))
            d_in = torch.cat([img, label_map], dim=1)
            features = self.conv(d_in)
            validity = self.fc(features)
            return validity

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Инициализация моделей
    generator = Generator(latent_dim, n_classes).to(device)
    discriminator = Discriminator(n_classes).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Функция потерь и оптимизаторы
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Фиксированный шум для визуализации прогресса (по одному на класс)
    fixed_noise = torch.randn(n_classes, latent_dim, device=device)
    fixed_labels = torch.arange(0, n_classes, device=device)

    # Цикл обучения
    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")):
            batch_size_current = imgs.size(0)
            imgs = imgs.to(device)
            labels = labels.to(device)

            valid = torch.ones(batch_size_current, 1, device=device)
            fake = torch.zeros(batch_size_current, 1, device=device)

            # --- Обучаем дискриминатор ---
            optimizer_D.zero_grad()
            validity_real = discriminator(imgs, labels)
            loss_D_real = criterion(validity_real, valid)
            loss_D_real.backward()

            noise = torch.randn(batch_size_current, latent_dim, device=device)
            gen_labels = torch.randint(0, n_classes, (batch_size_current,), device=device)
            gen_imgs = generator(noise, gen_labels)
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            loss_D_fake = criterion(validity_fake, fake)
            loss_D_fake.backward()
            optimizer_D.step()

            # --- Обучаем генератор ---
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size_current, latent_dim, device=device)
            gen_labels = torch.randint(0, n_classes, (batch_size_current,), device=device)
            gen_imgs = generator(noise, gen_labels)
            validity = discriminator(gen_imgs, gen_labels)
            loss_G = criterion(validity, valid)
            loss_G.backward()
            optimizer_G.step()

        # ===== СОХРАНЕНИЕ КАРТИНКИ ПОСЛЕ ЭПОХИ =====
        # Проверяем, кратна ли текущая эпоха (epoch+1) заданному интервалу
        if (epoch + 1) % epoch_interval == 0:
            with torch.no_grad():
                gen_imgs_fixed = generator(fixed_noise, fixed_labels).detach().cpu()
            # Денормализация из [-1,1] в [0,1]
            gen_imgs_fixed = (gen_imgs_fixed + 1) / 2.0
            save_image(
                gen_imgs_fixed,
                f"cgan_output/images/epoch_{epoch+1}.png",
                nrow=n_classes,
                normalize=False
            )
            print(f"Сохранена картинка для эпохи {epoch+1}")

        # Сохраняем модели после каждой эпохи
        torch.save(generator.state_dict(), f"cgan_output/models/generator_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"cgan_output/models/discriminator_epoch_{epoch+1}.pth")