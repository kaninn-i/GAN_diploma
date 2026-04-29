import os
import torch
from torchvision.utils import save_image
from ssd_model import Generator as SSDGenerator
from gan_model import Generator as DCGANGenerator


def generate_objects(generator_path, output_dir, num_images, latent_dim=128,
                     device="cuda", batch_size=64, model_type="dcgan", img_size=64,
                     ema_weights_path=None):
    if model_type.lower() == "ssd":
        generator = SSDGenerator(latent_dim, img_size=img_size).to(device)
    else:
        generator = DCGANGenerator(latent_dim).to(device)

    weights_file = ema_weights_path if ema_weights_path else generator_path
    generator.load_state_dict(torch.load(weights_file, map_location=device))
    generator.eval()

    os.makedirs(output_dir, exist_ok=True)
    generated = 0
    img_id = 0
    while generated < num_images:
        current_batch = min(batch_size, num_images - generated)
        z = torch.randn(current_batch, latent_dim, 1, 1, device=device)
        with torch.no_grad():
            fake = generator(z)
        fake = (fake + 1) / 2
        for i in range(current_batch):
            save_image(fake[i], os.path.join(output_dir, f"synth_{img_id:06d}.png"))
            img_id += 1
        generated += current_batch
    return img_id