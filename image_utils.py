import os
import torch
from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    TensorDataset,
    Dataset,
)
import torch.nn as nn
from PIL import Image
import math
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as NNF
import random


def normalize_transform():
    return transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class Random90Rotation:
    """Custom transform to rotate images into one of the 4 cardinal directions (0, 90, 180, 270 degrees)."""

    def __call__(self, img):
        # Randomly choose to rotate 0, 1, 2, or 3 times by 90 degrees
        k = random.choice([0, 1, 2, 3])
        return transforms.functional.rotate(img, k * 90)


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice = nn.Sequential(*list(vgg[:16])).eval()

        for p in self.slice.parameters():
            p.requires_grad = False

        # ImageNet normalization (required!)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, sr, hr):
        sr = self.normalize(sr)
        hr = self.normalize(hr)

        # Extract VGG features
        sr_f = self.slice(sr)
        hr_f = self.slice(hr)

        # L1 feature loss
        return NNF.l1_loss(sr_f, hr_f)


class SRLoss(nn.Module):
    def __init__(self, perceptual_weight=0.1):
        super().__init__()
        self.pixel_loss = nn.L1Loss()
        self.percep_loss = VGGPerceptualLoss()
        self.percep_loss.to(DEVICE)
        self.weight = perceptual_weight

    def forward(self, sr, hr):
        pixel = self.pixel_loss(sr, hr)
        percep = self.percep_loss(sr, hr)
        return pixel + self.weight * percep


def load_images_cropped(
    directory_path,
    crop_size,
    max_num_patches_per_image,
    keep_first_full_scale,
    grayscale=False,
):
    image_paths = [
        os.path.join(directory_path, fname)
        for fname in os.listdir(directory_path)
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    cropped_images = []

    for index, path in enumerate(image_paths):
        if index % 200 == 0:
            print(f"Processing image {index + 1} of {len(image_paths)}...")
        try:
            # Image is automatically closed when leaving this block
            with Image.open(path) as img:
                # Use grayscale ("L") or RGB
                img = img.convert("L" if grayscale else "RGB")
                w, h = img.size

                num_channels = 1 if grayscale else 3

                seen = set()
                attempts = 0
                max_attempts = max_num_patches_per_image * 2

                while len(seen) < max_num_patches_per_image and attempts < max_attempts:
                    attempts += 1

                    min_side = min(w, h)

                    if min_side > crop_size:
                        if keep_first_full_scale and len(seen) == 0:
                            scale = 1.0
                        else:
                            scale = (
                                torch.empty(1)
                                .uniform_(crop_size / min_side, 1.0)
                                .item()
                            )

                        new_w = int(w * scale)
                        new_h = int(h * scale)

                        resized = img.resize(
                            (new_w, new_h),
                            Image.Resampling.BICUBIC,
                        )
                    else:
                        resized = img
                        new_w, new_h = w, h

                    if new_w < crop_size or new_h < crop_size:
                        if resized is not img:
                            resized.close()
                        continue

                    top = torch.randint(0, new_h - crop_size + 1, (1,)).item()

                    left = torch.randint(0, new_w - crop_size + 1, (1,)).item()

                    patch_id = (left, top, new_w, new_h)

                    if patch_id in seen:
                        if resized is not img:
                            resized.close()
                        continue

                    seen.add(patch_id)

                    # Crop directly from PIL
                    crop = resized.crop(
                        (
                            left,
                            top,
                            left + crop_size,
                            top + crop_size,
                        )
                    )

                    # Convert ONLY the crop to torch tensor
                    crop_tensor = torch.ByteTensor(
                        torch.ByteStorage.from_buffer(crop.tobytes())
                    )

                    crop_tensor = (
                        crop_tensor.view(crop_size, crop_size, num_channels)
                        .permute(2, 0, 1)
                        .to(torch.float32)
                        / 255.0
                    )

                    cropped_images.append(crop_tensor)

                    crop.close()

                    if resized is not img:
                        resized.close()

                    del crop_tensor

                del seen

        except Exception as e:
            print(f"Failed to process image {path}: {e}")

    if cropped_images:
        return torch.stack(cropped_images)
    else:
        return torch.empty(
            (0, 1 if grayscale else 3, crop_size, crop_size),
            dtype=torch.float32,
        )


def cropped_dataset(
    image_dir,
    crop_size,
    max_num_patches_per_image=1,
    transform=None,
    keep_first_full_scale=False,
    grayscale=False,
):
    # Load the clean [N, 3, H, W] tensor using the existing function
    cropped_tensor = load_images_cropped(
        image_dir,
        crop_size,
        max_num_patches_per_image,
        keep_first_full_scale,
        grayscale,
    )
    if transform:
        cropped_tensor = transform(cropped_tensor)

    return TensorDataset(cropped_tensor)


def cifar_dataset(dataset_name="CIFAR100", root="./data"):
    def cifar_transform():
        return transforms.Compose([transforms.ToTensor(), normalize_transform()])

    dataset_classes = {
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
    }

    if dataset_name not in dataset_classes:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset_classes[dataset_name](
        root=root, download=True, transform=cifar_transform()
    )


def cifar100_dataset(root="./data"):
    return cifar_dataset("CIFAR100", root)


def cifar10_dataset(root="./data"):
    return cifar_dataset("CIFAR10", root)


def mixed_dataloader(datasets, batch_size):
    mixed_dataset = ConcatDataset(datasets)
    return DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def show_image_eval(image_type, images, loss, image_size=32):
    fsize = max(1.5, image_size / 32)

    plt.figure(figsize=(fsize, fsize), dpi=100)

    loss = loss.detach().item()

    if image_type == "real":
        p = math.exp(-loss)
    elif image_type == "fake":
        p = 1 - math.exp(-loss)
    else:
        p = 0.0

    img = images[0]

    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = img.permute(1, 2, 0)

    # Scale [-1, 1] -> [0, 1]
    img = (img + 1) / 2

    if img.shape[-1] == 1:
        img = img.squeeze(-1)
        cmap = "gray"
    else:
        cmap = None

    img = img.detach().cpu().numpy()

    plt.imshow(img, cmap=cmap)
    plt.title(f"p={p:.2f}: loss={loss:.3f}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def display_images(generated_images, image_size=32, dpi=100):
    fsize = max(2, image_size / 32)

    rows, cols = 2, 2
    num_images = min(len(generated_images), rows * cols)

    plt.figure(
        figsize=(fsize, fsize),
        dpi=dpi,
    )

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)

        img = generated_images[i].detach().cpu().permute(1, 2, 0)
        img = ((img + 1) / 2).clamp(0, 1).numpy()

        if img.shape[-1] == 1:
            plt.imshow(img.squeeze(-1), cmap="gray")
        else:
            plt.imshow(img)

        plt.axis("off")

    plt.tight_layout()
    plt.show()


def normalize(image):
    """
    Normalizes a tensor image from range [0, 1] to [-1, 1]
    """
    return (image - 0.5) / 0.5


def denormalize(tensor):
    """
    Denormalizes a tensor image from range [-1, 1] to [0, 1]
    for proper display with matplotlib.
    """
    # Inverse operation of Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1)  # Clamp values just in case
    return tensor


class ImageManager:
    """
    Manages image and latent conversions, caching, and visualization
    for the model, optimizing GPU memory by offloading the VAE.
    """

    def __init__(self, logger, images_dir):
        self.logger = logger
        self.images_dir = images_dir
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                # Mild color change (5%)
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.003
                ),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    class VAEManager:
        def __init__(self, model_name):
            self.model_name = model_name
            self.vae = None

        def create_vae(self):
            """Create a VAE and keep it in GPU memory until program ends."""
            if self.vae is None:
                self.vae = AutoencoderKL.from_pretrained(
                    self.model_name, torch_dtype=torch.float16
                ).to(common.DEVICE)
                self.vae.eval()
            return self.vae

        def __enter__(self):
            """Context manager entry — create VAE if not already created."""
            return self.create_vae()

        def __exit__(self, exc_type, exc_value, traceback):
            """Context manager exit — delete VAE to free GPU."""
            if self.vae is not None:
                del self.vae
                self.vae = None
                torch.cuda.empty_cache()

    def cache_latents_from_image_dataset(
        self, latent_dir, raw_crop_size, vae_model_name, latent_scale
    ):
        if os.path.exists(latent_dir):
            self.logger.info(
                f"{latent_dir} directory already exists. Skip to generate latent files"
            )
        else:
            self.logger.info("Creating latents from dataset...")
            dataset = cropped_dataset(
                self.images_dir,
                crop_size=raw_crop_size,
                max_num_patches_per_image=1,
                transform=self.transform,
            )
            os.makedirs(latent_dir)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
            counter = 0
            with ImageManager.VAEManager(vae_model_name) as vae:
                for data in dataloader:
                    images = data[0] if isinstance(data, (tuple, list)) else data
                    images = images.to(common.DEVICE, dtype=torch.float16)
                    with torch.inference_mode():
                        latents = vae.encode(images).latent_dist.mode() * latent_scale
                    # Save each latent separately
                    for latent in latents:
                        torch.save(latent.cpu(), f"{latent_dir}/latent_{counter}.pt")
                        counter += 1

            self.logger.info(f"Saved latents to {latent_dir}")

    def cache_images_from_image_dataset(
        self, cropped_images_path, image_size, grayscale=False
    ):
        cache_file = cropped_images_path
        if os.path.exists(cache_file):
            self.logger.info(
                f"{cache_file} already exists. " "Skip to generate cached images"
            )
            return

        self.logger.info("Creating cached images from dataset...")

        dataset = cropped_dataset(
            self.images_dir,
            crop_size=image_size,
            max_num_patches_per_image=1,
            grayscale=grayscale,
        )

        images = []

        for i in range(len(dataset)):
            data = dataset[i]
            image = data[0] if isinstance(data, (tuple, list)) else data

            if not torch.is_tensor(image):
                image = T.ToTensor()(image)

            images.append(image)

        images = torch.stack(images)
        torch.save(images, cache_file)
        self.logger.info(f"Saved {len(images)} images to {cache_file}")

    class LatentPatchDataset(Dataset):
        def __init__(self, latent_dir, crop_size=None):
            self.files = [
                os.path.join(latent_dir, f)
                for f in os.listdir(latent_dir)
                if f.endswith(".pt")
            ]
            self.crop_size = crop_size

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            latent = torch.load(self.files[idx])

            if self.crop_size:
                C, H, W = latent.shape
                top = random.randint(0, H - self.crop_size)
                left = random.randint(0, W - self.crop_size)
                latent = latent[
                    :, top : top + self.crop_size, left : left + self.crop_size
                ]

            return latent

    class CachedImageDataset(Dataset):
        def __init__(self, images, transform=None):
            self.images = images
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            if self.transform is not None:
                image = self.transform(image)
            return image

    @staticmethod
    def _to_pil(img_tensor):
        """Convert a [C,H,W] tensor to a PIL Image."""

        img = img_tensor.detach().cpu().float()
        img = img.clamp(0, 1)

        # Grayscale: [1, H, W] -> [H, W]
        if img.shape[0] == 1:
            img = img.squeeze(0).numpy()
            img = (img * 255).astype("uint8")
            return Image.fromarray(img, mode="L")

        # RGB: [3, H, W] -> [H, W, 3]
        elif img.shape[0] == 3:
            img = img.permute(1, 2, 0).numpy()
            img = (img * 255).astype("uint8")
            return Image.fromarray(img, mode="RGB")

        else:
            raise ValueError(f"Expected 1 or 3 channels, got shape {tuple(img.shape)}")

    @staticmethod
    def latent_to_image(latent_scale, vae, latent):
        latent = latent.unsqueeze(0).to(vae.device, dtype=torch.float16)
        with torch.no_grad():
            img = vae.decode(latent / latent_scale).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return ImageManager._to_pil(img[0])

    @staticmethod
    def tensor_to_image(img_tensor):
        return ImageManager._to_pil(img_tensor)
