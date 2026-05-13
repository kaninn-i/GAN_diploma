import os
import torch
from torchvision.utils import save_image
from ssd_model import Generator as SSDGenerator
from ssd_lite_model import Generator as SSDLiteGenerator
from gan_model import Generator as DCGANGenerator
from dcgan_sn_model import Generator as DCGANSNGenerator


def generate_objects(
    generator_path,
    output_dir,
    num_images,
    latent_dim=128,
    device="cuda",
    batch_size=64,
    model_type="dcgan",
    img_size=64,
    ema_weights_path=None,
    # ── StyleGAN2-ADA ────────────────────────────────────────────────────────
    sg2ada_repo=None,
    truncation_psi=0.7,
    progress_callback=None,
):
    """
    Генерация синтетических объектов.

    model_type:
        "dcgan"        – оригинальный DCGAN
        "dcgan_sn"     – DCGAN со spectral norm
        "ssd"          – SSD-генератор
        "ssd_lite"     – облегчённый SSD
        "stylegan2_ada"– StyleGAN2-ADA (.pkl чекпоинт; generator_path = путь к .pkl)
    """
    mt = model_type.lower()

    # ── StyleGAN2-ADA ─────────────────────────────────────────────────────────
    if mt == "stylegan2_ada":
        from stylegan2_ada_generator import generate_objects_sg2ada
        return generate_objects_sg2ada(
            generator_path=generator_path,
            output_dir=output_dir,
            num_images=num_images,
            device=device,
            batch_size=min(16, batch_size),   # SG2 тяжелее, батч меньше
            truncation_psi=truncation_psi,
            sg2ada_repo=sg2ada_repo,
            progress_callback=progress_callback,
        )

    # ── GAN (без изменений) ───────────────────────────────────────────────────
    if mt == "ssd":
        generator = SSDGenerator(latent_dim, img_size=img_size).to(device)
    elif mt == "ssd_lite":
        generator = SSDLiteGenerator(latent_dim, img_size=img_size).to(device)
    elif mt == "dcgan_sn":
        generator = DCGANSNGenerator(latent_dim).to(device)
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
