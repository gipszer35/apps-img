import os
import torch
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import torch.nn as nn
from PIL import Image
import math
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as NNF


def normalize_transform():
    return transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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
):
    image_paths = [
        os.path.join(directory_path, fname)
        for fname in os.listdir(directory_path)
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    cropped_images = []

    for path in image_paths:
        try:
            # Image is automatically closed when leaving this block
            with Image.open(path) as img:
                img = img.convert("RGB")
                w, h = img.size

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

                    # Crop directly from PIL.
                    # This avoids converting the entire resized image
                    # to a large torch tensor.
                    crop = resized.crop(
                        (
                            left,
                            top,
                            left + crop_size,
                            top + crop_size,
                        )
                    )

                    # Convert ONLY the crop to torch tensor [C, H, W]
                    crop_tensor = torch.ByteTensor(
                        torch.ByteStorage.from_buffer(crop.tobytes())
                    )
                    crop_tensor = (
                        crop_tensor.view(crop_size, crop_size, 3)
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
        return torch.empty((0, 3, crop_size, crop_size), dtype=torch.float32)


def cropped_dataset(
    image_dir,
    crop_size,
    max_num_patches_per_image=1,
    transform=None,
    keep_first_full_scale=False,
):
    # Load the clean [N, 3, H, W] tensor using the existing function
    cropped_tensor = load_images_cropped(
        image_dir,
        crop_size,
        max_num_patches_per_image,
        keep_first_full_scale,
    )
    if transform:
        cropped_tensor = transform(cropped_tensor)

    return TensorDataset(cropped_tensor)


def cifar100_dataset(root="./data"):
    cifar_dataset = datasets.CIFAR100(
        root=root, download=True, transform=normalize_transform()
    )
    return cifar_dataset


def cifar10_dataset(root="./data"):
    cifar_dataset = datasets.CIFAR10(
        root=root, download=True, transform=normalize_transform()
    )
    return cifar_dataset


def mixed_dataloader(datasets, batch_size):
    mixed_dataset = ConcatDataset(datasets)
    return DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def show_image_eval(image_type, images, loss):
    plt.figure(figsize=(1.5, 1.5))

    loss = loss.detach()
    if image_type == "real":
        p = math.exp(-loss)
    elif image_type == "fake":
        p = 1 - math.exp(-loss)
    img = images[0]

    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = img.permute(1, 2, 0)  # -> (H, W, C)

    # Scale [-1,1] → [0,1]
    img = (img + 1) / 2

    if img.shape[-1] == 1:
        img = img.squeeze(-1)
        cmap = "gray"
    else:
        cmap = None

    img = img.detach().numpy()
    plt.imshow(img, cmap=cmap)
    plt.title(f"p={p:.2f}: loss={loss:.3f}")
    plt.axis("off")

    plt.show()


def display_images(generated_images, dpi=100):
    dim = (2, 2)
    num_images = min(len(generated_images), dim[0] * dim[1])

    plt.figure(figsize=(2, 2), dpi=dpi)

    for i in range(num_images):
        plt.subplot(dim[0], dim[1], i + 1)

        img = generated_images[i].permute(1, 2, 0).detach().cpu()
        img = (img + 1) / 2
        img = img.clamp(0, 1).numpy()

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
