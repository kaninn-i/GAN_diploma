import os
import torch
from torchvision.utils import save_image
from gan.generator import Generator


def generate_objects(
    weights_path,
    output_path="generated_objects",
    num_classes=2,
    num_images_per_class=100,
    noise_dim=100,
    device="cuda"
):
    os.makedirs(output_path, exist_ok=True)

    generator = Generator(noise_dim=noise_dim, num_classes=num_classes).to(device)
    generator.load_state_dict(torch.load(weights_path, map_location=device))
    generator.eval()

    for cls in range(num_classes):
        class_dir = os.path.join(output_path, f"class_{cls}")
        os.makedirs(class_dir, exist_ok=True)

        for i in range(num_images_per_class):
            noise = torch.randn(1, noise_dim, 1, 1, device=device)
            label = torch.tensor([cls], device=device)

            with torch.no_grad():
                fake_img = generator(noise, label)
                fake_img = (fake_img + 1) / 2  # в диапазон [0,1]

            save_path = os.path.join(class_dir, f"{cls}_{i}.png")
            save_image(fake_img, save_path)

    print("Генерация завершена.")