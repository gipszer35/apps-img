import os
import sys
import datetime
import random
import dataclasses
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from PIL import Image
import matplotlib.pyplot as plt


class Environment:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.content_drive = "/content/drive"
        self.is_colab = "google.colab" in sys.modules

        if self.is_colab:
            self._mount_colab()

            self.root_dir = os.path.join(self.content_drive, "MyDrive")
            self.apps_img = os.path.join(self.root_dir, "apps-img")
            self.images_dir = os.path.join(self.root_dir, "images_few")
        else:
            self.root_dir = os.path.expanduser("~/work")
            self.apps_img = os.path.join(self.root_dir, "apps-img")
            self.images_dir = os.path.join(self.apps_img, "images", "my-images")

        self.work_dir = os.path.join(self.apps_img, "autoencoder")

        os.makedirs(self.work_dir, exist_ok=True)

        for path in (
            self.root_dir,
            self.apps_img,
            self.work_dir,
        ):
            if path not in sys.path:
                sys.path.append(path)

        import common
        import image_utils

        globals()["common"] = common
        globals()["image_utils"] = image_utils

    def _mount_colab(self):
        if not os.path.ismount(self.content_drive):
            from google.colab import drive

            drive.mount(self.content_drive)


# Initialize the environment before using common/image_utils.
Environment()


@dataclasses.dataclass
class AutoencoderConfig:
    # Automatically populated environment paths.
    work_dir: str
    images_dir: str
    content_drive: str

    # Training configuration.
    batch_size: int
    epoch: int
    lr: float
    beta1: float
    num_workers: int

    # Image configuration.
    num_channel: int
    image_size: int
    lr_image_size: int
    crop_image_size: int
    num_patches_per_image: int

    # Checkpoint configuration.
    checkpoint_name: str

    # Loss configuration.
    perceptual_weight: float = 0.1

    # Logging and visualization.
    log_interval: int = 50
    show_images: bool = True

    # Optimizer configuration.
    beta2: float = 0.999

    @classmethod
    def create(cls, **kwargs) -> "AutoencoderConfig":
        env = Environment()

        kwargs.update(
            {
                "work_dir": env.work_dir,
                "images_dir": env.images_dir,
                "content_drive": env.content_drive,
            }
        )

        # Environment-dependent defaults.
        kwargs.setdefault(
            "batch_size",
            64 if env.is_colab else 4,
        )
        kwargs.setdefault(
            "num_workers",
            2,
        )
        kwargs.setdefault(
            "num_patches_per_image",
            4 if env.is_colab else 5,
        )

        # Apply dataclass defaults for optional fields.
        for field in dataclasses.fields(cls):
            if field.name not in kwargs and field.default is not dataclasses.MISSING:
                kwargs[field.name] = field.default

        return cls(**kwargs)

    @property
    def out_dir(self):
        return os.path.join(self.work_dir, "out_dir")

    @property
    def checkpoint_path(self):
        return os.path.join(self.out_dir, self.checkpoint_name)

    @property
    def cropped_images_path(self):
        return os.path.join(
            self.out_dir,
            f"cropped_{self.image_size}.pt",
        )


class ResBlock(nn.Module):
    def __init__(self, channels, p=0.7):
        super().__init__()

        self.p = p

        self.sequential = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return x * (1 - self.p) + self.sequential(x) * self.p


def upshuffle(in_channels, out_channels, scale=2):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels * (scale**2),
            kernel_size=3,
            padding=1,
        ),
        nn.ReLU(inplace=True),
        nn.PixelShuffle(scale),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def downsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SuperResolutionAutoEncoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()

        self.config = config

        # Preserve the original architecture's channel layout.
        divisor = 2

        channel256x256 = 256 // divisor
        channel128x128 = 128 // divisor
        channel64x64 = 128 // divisor
        channel32x32 = 256 // divisor
        channel16x16 = 512 // divisor
        channel8x8 = 256 // divisor

        # Encoder: 32x32 -> 16x16 -> 8x8.
        self.down_conv1 = nn.Conv2d(
            config.num_channel,
            channel32x32,
            kernel_size=3,
            padding=1,
        )

        self.down_sample1 = downsample(
            channel32x32,
            channel16x16,
        )

        self.down_conv2 = ResBlock(
            channel16x16,
        )

        self.down_sample2 = downsample(
            channel16x16,
            channel8x8,
        )

        self.bottleneck = ResBlock(
            channel8x8,
        )

        # Decoder: 8x8 -> 16x16 -> 32x32.
        self.up_sample1 = upshuffle(
            channel8x8,
            channel16x16,
        )

        self.up_conv1 = ResBlock(
            channel16x16 * 2,
        )

        self.up_sample2 = upshuffle(
            channel16x16 * 2,
            channel32x32,
        )

        self.up_conv2 = ResBlock(
            channel32x32 * 2,
        )

        # Decoder: 32x32 -> 64x64 -> 128x128 -> 256x256.
        self.up_sample3 = upshuffle(
            channel32x32 * 2,
            channel64x64,
        )

        self.up_conv3 = ResBlock(
            channel64x64,
        )

        self.up_sample4 = upshuffle(
            channel64x64,
            channel128x128,
        )

        self.up_conv4 = ResBlock(
            channel128x128,
        )

        self.up_sample5 = upshuffle(
            channel128x128,
            channel256x256,
        )

        self.out = nn.Conv2d(
            channel256x256,
            config.num_channel,
            kernel_size=5,
            padding=2,
        )

    def forward(self, x):
        # Encoder.
        down_conv1 = self.down_conv1(x)
        down_sample1 = self.down_sample1(down_conv1)

        down_conv2 = self.down_conv2(down_sample1)
        down_sample2 = self.down_sample2(down_conv2)

        bottleneck = self.bottleneck(down_sample2)

        # Decoder with skip connections.
        up_sample1 = self.up_sample1(bottleneck)

        cat1 = torch.cat(
            [up_sample1, down_conv2],
            dim=1,
        )

        up_conv1 = self.up_conv1(cat1)
        up_sample2 = self.up_sample2(up_conv1)

        cat2 = torch.cat(
            [up_sample2, down_conv1],
            dim=1,
        )

        up_conv2 = self.up_conv2(cat2)
        up_sample3 = self.up_sample3(up_conv2)

        up_conv3 = self.up_conv3(up_sample3)
        up_sample4 = self.up_sample4(up_conv3)

        up_conv4 = self.up_conv4(up_sample4)
        up_sample5 = self.up_sample5(up_conv4)

        out = self.out(up_sample5)

        return torch.tanh(out)


class SuperResolutionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        config: AutoencoderConfig,
    ):
        self.base_dataset = base_dataset
        self.config = config

        self.hr_transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: F.rotate(
                        img,
                        random.choice([0, 90, 180, 270]),
                    )
                ),
                transforms.RandomCrop(
                    (
                        config.image_size,
                        config.image_size,
                    )
                ),
                transforms.ColorJitter(
                    brightness=0.05,
                    contrast=0.05,
                    saturation=0.05,
                    hue=0.05,
                ),
            ]
        )

        self.lr_transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        config.lr_image_size,
                        config.lr_image_size,
                    ),
                    interpolation=(transforms.InterpolationMode.BICUBIC),
                )
            ]
        )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]

        if isinstance(image, torch.Tensor):
            image = common.denormalize(image)
            image = F.to_pil_image(image)

        # Create the high-resolution target.
        image_hr = self.hr_transform(image)

        # Downsample the same image to create the input.
        image_lr = self.lr_transform(image_hr)

        image_hr = common.normalize(self.to_tensor(image_hr))

        image_lr = common.normalize(self.to_tensor(image_lr))

        return image_lr, image_hr


class AutoencoderTrainer:
    def __init__(
        self,
        logger,
        config: AutoencoderConfig,
    ):
        self.logger = logger
        self.config = config

        os.makedirs(
            self.config.out_dir,
            exist_ok=True,
        )

        self.model = None
        self.optimizer = None
        self.dataloader = None
        self.sr_dataset = None

    @classmethod
    def from_yaml(
        cls,
        profile_name: str,
        logger,
        config_path: str = "config.yaml",
    ):
        if not os.path.isabs(config_path):
            env = Environment()

            config_path = os.path.join(
                env.work_dir,
                config_path,
            )

        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        profile = full_config["trainers"][profile_name]

        config = AutoencoderConfig.create(**profile)

        return cls(
            logger=logger,
            config=config,
        )

    def prepare_dataloader(self):
        self.logger.info("Preparing super-resolution dataset")

        config = self.config

        # Keep using the project's image-patch dataset.
        base_dataset = image_utils.cropped_dataset(
            config.images_dir,
            crop_size=config.crop_image_size,
            num_patches_per_image=(config.num_patches_per_image),
        )

        self.sr_dataset = SuperResolutionDataset(
            base_dataset,
            config,
        )

        self.dataloader = DataLoader(
            self.sr_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

        self.logger.info(f"Dataset size: {len(self.sr_dataset)}")

        self.logger.info(
            f"Batches: {len(self.dataloader)} | "
            f"Batch size: {config.batch_size} | "
            f"LR size: {config.lr_image_size} | "
            f"HR size: {config.image_size}"
        )

        return self.dataloader

    @staticmethod
    def save_checkpoint(
        model,
        optimizer,
        path,
        epoch,
    ):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            path,
        )

    def load_checkpoint_if_exists(self):
        path = self.config.checkpoint_path

        if not os.path.isfile(path):
            self.logger.info(f"No checkpoint found at {path}")
            return 0

        checkpoint = torch.load(
            path,
            map_location=common.DEVICE,
        )

        self.model.load_state_dict(checkpoint["model"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint.get("epoch", 0)

        self.logger.info(
            f"Loaded checkpoint from {path}; " f"resuming at epoch {start_epoch + 1}"
        )

        return start_epoch

    def initialize_model(self):
        self.model = SuperResolutionAutoEncoder(self.config).to(common.DEVICE)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            betas=(
                self.config.beta1,
                self.config.beta2,
            ),
        )

        self.logger.info("Super-resolution autoencoder")

        common.print_parameter_summary(self.model)

        start_epoch = self.load_checkpoint_if_exists()

        return start_epoch

    @staticmethod
    def show_tensor_image(img, title):
        img = img.detach().cpu()

        img = common.denormalize(img)
        img = img.clamp(0, 1)

        img = img.permute(1, 2, 0).numpy()

        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    def show_resized_image(self):
        if self.sr_dataset is None or len(self.sr_dataset) == 0:
            return

        self.model.eval()

        with torch.no_grad():
            idx = random.randrange(len(self.sr_dataset))

            test_lr, test_hr = self.sr_dataset[idx]

            test_lr = test_lr.unsqueeze(0).to(common.DEVICE)

            output_hr = self.model(test_lr).squeeze(0)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        self.show_tensor_image(
            test_lr.squeeze(0),
            f"Input {self.config.lr_image_size}x" f"{self.config.lr_image_size}",
        )

        plt.subplot(1, 3, 2)
        self.show_tensor_image(
            test_hr,
            "Target",
        )

        plt.subplot(1, 3, 3)
        self.show_tensor_image(
            output_hr,
            "Generated",
        )

        plt.tight_layout()
        plt.show()

    def upscale_and_show_one_image(self, image_path):
        self.model.eval()

        image = Image.open(image_path).convert("RGB")

        image = image.resize(
            (
                self.config.lr_image_size,
                self.config.lr_image_size,
            ),
            Image.Resampling.BICUBIC,
        )

        input_tensor = transforms.ToTensor()(image).unsqueeze(0)

        input_tensor = common.normalize(input_tensor).to(common.DEVICE)

        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        output_tensor = common.denormalize(output_tensor.squeeze(0).cpu()).clamp(0, 1)

        output_image = transforms.ToPILImage()(output_tensor)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Low-resolution input")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(output_image)
        plt.title(f"Generated {self.config.image_size}x" f"{self.config.image_size}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def train(self):
        config = self.config

        self.logger.info("Starting super-resolution training")

        self.prepare_dataloader()
        start_epoch = self.initialize_model()

        criterion = common.SRLoss(perceptual_weight=config.perceptual_weight)

        global_step = 0

        for epoch in range(
            start_epoch,
            config.epoch,
        ):
            self.model.train()

            total_loss = 0.0

            for batch_idx, (lr_imgs, hr_imgs) in enumerate(self.dataloader):
                lr_imgs = lr_imgs.to(common.DEVICE)

                hr_imgs = hr_imgs.to(common.DEVICE)

                # Forward pass.
                outputs = self.model(lr_imgs)
                loss = criterion(outputs, hr_imgs)

                # Backward pass.
                self.optimizer.zero_grad(set_to_none=True)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                global_step += 1

                if global_step % config.log_interval == 0:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    self.logger.info(
                        f"Time: {now} | "
                        f"Epoch [{epoch + 1}/{config.epoch}] | "
                        f"Step [{batch_idx + 1}/"
                        f"{len(self.dataloader)}] | "
                        f"Loss: {loss.item():.4f}"
                    )

                    self.save_checkpoint(
                        self.model,
                        self.optimizer,
                        config.checkpoint_path,
                        epoch + 1,
                    )

                    if config.show_images:
                        self.show_resized_image()

            avg_loss = (
                total_loss / len(self.dataloader) if len(self.dataloader) > 0 else 0.0
            )

            self.logger.info(
                f"Epoch [{epoch + 1}/{config.epoch}] "
                f"completed | Average loss: {avg_loss:.4f}"
            )

            # Save at the end of every epoch, too.
            self.save_checkpoint(
                self.model,
                self.optimizer,
                config.checkpoint_path,
                epoch + 1,
            )

        self.logger.info("Super-resolution training finished.")


if __name__ == "__main__":
    logger = common.create_logger()

    trainer = AutoencoderTrainer.from_yaml(
        profile_name="autoencoder_small",
        logger=logger,
    )

    trainer.train()
