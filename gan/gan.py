import torch
import torch.optim as optim
import torch.nn as nn
import os
import datetime
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
import dataclasses
import yaml
from torch.utils.data import DataLoader


class Environment:
    _instance = None  # Stores the single global instance

    def __new__(cls):
        """Ensures only one instance of Environment is ever created (Singleton)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Runs only once during the very first instantiation."""
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
        self.work_dir = os.path.join(self.apps_img, "gan")

        # Update sys.path so Python knows where to look for imports
        sys.path.extend([self.root_dir, self.apps_img, self.work_dir])

        # Import the modules inside and inject them into the global namespace
        import common
        import image_utils

        globals()["common"] = common
        globals()["image_utils"] = image_utils

    def _mount_colab(self):
        if not os.path.ismount(self.content_drive):
            from google.colab import drive

            drive.mount(self.content_drive)


# Triggers initialization, mounts drive, and injects global imports immediately.
# Custom library imports and directory accesses will fail if this is moved or removed.
Environment()


@dataclasses.dataclass(frozen=False)
class GanConfig:
    # Environment paths
    work_dir: str
    images_dir: str
    content_drive: str
    batch_size: int
    num_channel: int
    beta: float
    epoch: int
    checkpoint_name: str

    # Image and latent dimensions
    image_size: int = 32
    input_size: int = 32
    latent_vector_size: int = 128
    generator_base: int = 256
    discriminator_base: int = 256

    # Training mode
    super_resolution: bool = False

    # Loss weights
    reconstruction_weight: float = 10.0

    # Learning rates: change these directly in the source
    lr_generator: float = 1e-4
    lr_discriminator: float = 1e-4

    @classmethod
    def create(cls, **kwargs) -> "GanConfig":
        env = Environment()

        kwargs.update(
            {
                "work_dir": env.work_dir,
                "images_dir": env.images_dir,
                "content_drive": env.content_drive,
            }
        )

        kwargs.setdefault("batch_size", 64 if env.is_colab else 2)

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


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base = config.generator_base

        if config.image_size < 4 or config.image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4 and at least 4.")

        if config.super_resolution:
            if config.input_size < 4:
                raise ValueError("input_size must be at least 4.")

            if config.image_size < config.input_size:
                raise ValueError(
                    "image_size must be >= input_size for super-resolution."
                )

            if config.image_size % config.input_size != 0:
                raise ValueError("image_size must be divisible by input_size.")

            scale = config.image_size // config.input_size

            if scale & (scale - 1):
                raise ValueError(
                    "The super-resolution scale factor must be a power of 2."
                )

            # Encode the low-resolution image.
            self.input_conv = nn.Sequential(
                nn.Conv2d(
                    config.num_channel,
                    self.base // 4,
                    kernel_size=3,
                    padding=1,
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    self.base // 4,
                    self.base // 2,
                    kernel_size=3,
                    padding=1,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )

            # Optional additional latent vector.
            if config.latent_vector_size > 0:
                self.latent_fc = nn.Linear(
                    config.latent_vector_size,
                    self.base // 2,
                )

            # Project encoded features to the decoder's channels.
            self.input_projection = nn.Conv2d(
                self.base // 2,
                self.base,
                kernel_size=3,
                padding=1,
            )

            # Build the super-resolution decoder during initialization.
            # This ensures all parameters are registered before the
            # optimizer is created.
            decoder_layers = []
            channels = self.base
            current_size = config.input_size

            while current_size < config.image_size:
                next_channels = max(
                    self.base // 4,
                    channels // 2,
                )

                decoder_layers.extend(
                    [
                        nn.Upsample(
                            scale_factor=2,
                            mode="nearest",
                        ),
                        nn.Conv2d(
                            channels,
                            next_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.GroupNorm(
                            min(32, next_channels),
                            next_channels,
                        ),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )

                channels = next_channels
                current_size *= 2

            decoder_layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        config.num_channel,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.Tanh(),
                ]
            )

            self.sr_decoder = nn.Sequential(*decoder_layers)

        else:
            # Unconditional GAN architecture.
            self.fc = nn.Linear(
                config.latent_vector_size,
                self.base * 4 * 4,
            )

            layers = []
            channels = self.base
            current_size = 4

            while current_size < config.image_size:
                next_channels = max(
                    self.base // 4,
                    channels // 2,
                )

                layers.extend(
                    [
                        nn.Upsample(
                            scale_factor=2,
                            mode="nearest",
                        ),
                        nn.Conv2d(
                            channels,
                            next_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.GroupNorm(
                            min(32, next_channels),
                            next_channels,
                        ),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )

                channels = next_channels
                current_size *= 2

            layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        config.num_channel,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.Tanh(),
                ]
            )

            self.main = nn.Sequential(*layers)

    def forward(self, x, z=None):
        if self.config.super_resolution:
            # x: (B, C, input_size, input_size)
            x = self.input_conv(x)

            # Optionally inject additional latent information.
            if self.config.latent_vector_size > 0:
                if z is None:
                    z = torch.randn(
                        x.size(0),
                        self.config.latent_vector_size,
                        device=x.device,
                        dtype=x.dtype,
                    )

                latent = self.latent_fc(z)
                latent = latent.unsqueeze(-1).unsqueeze(-1)

                # Broadcast latent information over spatial dimensions.
                x = x + latent

            x = self.input_projection(x)

            # input_size -> image_size
            return self.sr_decoder(x)

        # Original unconditional GAN mode.
        x = self.fc(x)
        x = x.view(x.size(0), self.base, 4, 4)

        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base = config.discriminator_base

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(
                    in_c,
                    out_c,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                common.GaussianNoise(0.1),
                nn.GroupNorm(
                    min(32, out_c),
                    out_c,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )

        channels = [
            max(self.base // 4, 32),
            max(self.base // 2, 32),
            self.base,
            self.base * 2,
        ]

        layers = []
        in_channels = config.num_channel

        current_size = config.image_size

        for out_channels in channels:
            if current_size < 2:
                break

            layers.append(block(in_channels, out_channels))

            in_channels = out_channels
            current_size //= 2

        self.blocks = nn.Sequential(*layers)

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.final(x).view(-1)


class GanTrainer:
    def __init__(self, logger, config: GanConfig):
        self.logger = logger
        self.config = config
        self.multi_loss_tracker = common.MultiLossTracker()
        self.image_manager = image_utils.ImageManager(logger, config.images_dir)

        if not os.path.exists(self.config.out_dir):
            self.logger.info(f"Creating output directory: {self.config.out_dir}")
            os.makedirs(self.config.out_dir, exist_ok=True)

    @classmethod
    def from_yaml(cls, profile_name: str, logger, config_path: str = "config.yaml"):
        """Factory method to instantiate the trainer using a YAML profile."""

        # If the path is relative, look for it inside the environment's work directory
        if not os.path.isabs(config_path):
            env = Environment()
            config_path = os.path.join(env.work_dir, config_path)

        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        profile_dict = full_config["trainers"][profile_name]
        config = GanConfig.create(**profile_dict)
        for key, value in vars(config).items():
            logger.info("Config: %s = %s", key, value)
        return cls(logger=logger, config=config)

    @staticmethod
    def save_gan_checkpoint(model_g, opt_g, model_d, opt_d, path):
        checkpoint = {
            "G": model_g.state_dict(),
            "opt_G": opt_g.state_dict(),
            "D": model_d.state_dict(),
            "opt_D": opt_d.state_dict(),
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load_gan_checkpoint_if_exists(model_g, opt_g, model_d, opt_d, path):
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            model_g.load_state_dict(checkpoint["G"])
            opt_g.load_state_dict(checkpoint["opt_G"])
            model_d.load_state_dict(checkpoint["D"])
            opt_d.load_state_dict(checkpoint["opt_D"])
            print(f"Loaded checkpoint from {path}")
        else:
            print(f"No checkpoint found at {path}, skipping load.")

    def resize_to_input(self, images):
        return nn.functional.interpolate(
            images,
            size=(
                self.config.input_size,
                self.config.input_size,
            ),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

    def prepare_cropped_image_dataloader(self):
        self.logger.info("Creating dataset and preparing dataloader...")

        self.image_manager.cache_images_from_image_dataset(
            self.config.cropped_images_path, self.config.image_size
        )
        cache_file = self.config.cropped_images_path
        images = torch.load(
            cache_file,
            map_location="cpu",
        )
        transform = transforms.Compose(
            [
                image_utils.Random90Rotation(),
                transforms.Normalize(
                    mean=[0.5] * self.config.num_channel,
                    std=[0.5] * self.config.num_channel,
                ),
            ]
        )

        dataset = image_utils.ImageManager.CachedImageDataset(
            images,
            transform=transform,
        )

        self.logger.info(
            f"Dataloader successfully initialized with {len(dataset)} samples."
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

    def prepare_cifar_dataloader(self):
        self.logger.info("Preparing datasets and dataloader")

        cifar_cache_path = os.path.join(self.config.out_dir, "cifar100_dataset.pt")

        if os.path.exists(cifar_cache_path):
            self.logger.info(f"Loading CIFAR100 dataset from cache: {cifar_cache_path}")
            cifar_dataset = torch.load(cifar_cache_path, weights_only=False)
        else:
            self.logger.info("CIFAR100 dataset not found in cache, creating new...")
            cifar_dataset = image_utils.cifar100_dataset()

            torch.save(cifar_dataset, cifar_cache_path)
            self.logger.info(f"Saved CIFAR100 dataset to: {cifar_cache_path}")

        datasets = [
            cifar_dataset,
            # image_utils.cropped_dataset(self.config.images_dir, 32),
        ]

        dataloader = image_utils.mixed_dataloader(
            datasets,
            self.config.batch_size,
        )
        self.logger.info(
            f"Dataloader preparation finished: "
            f"{len(dataloader)} batches, "
            f"batch_size={self.config.batch_size}"
        )
        return dataloader

    def print_losses(self, loss_d, loss_real, loss_fake, loss_g):

        avg_loss_d = self.multi_loss_tracker.calculate_loss("loss_d", loss_d)
        avg_loss_real = self.multi_loss_tracker.calculate_loss("loss_real", loss_real)
        avg_loss_fake = self.multi_loss_tracker.calculate_loss("loss_fake", loss_fake)
        avg_loss_g = self.multi_loss_tracker.calculate_loss("loss_g", loss_g)

        self.logger.info(
            f"D Loss: {loss_d.item():.4f} (avg: {avg_loss_d:.4f}) | "
            f"Real: {loss_real.item():.4f} (avg: {avg_loss_real:.4f}) | "
            f"Fake: {loss_fake.item():.4f} (avg: {avg_loss_fake:.4f})"
        )

        self.logger.info(f"G Loss: {loss_g.item():.4f} (avg: {avg_loss_g:.4f})")

    def train(self):
        generator = Generator(self.config).to(common.DEVICE)
        discriminator = Discriminator(self.config).to(common.DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        reconstruction_criterion = nn.L1Loss()

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self.config.lr_discriminator,
            betas=(self.config.beta, 0.999),
        )

        optimizerG = optim.Adam(
            generator.parameters(),
            lr=self.config.lr_generator,
            betas=(self.config.beta, 0.999),
        )

        dataloader = self.prepare_cropped_image_dataloader()

        GanTrainer.load_gan_checkpoint_if_exists(
            generator,
            optimizerG,
            discriminator,
            optimizerD,
            self.config.checkpoint_path,
        )

        self.logger.info("Generator")
        common.print_parameter_summary(generator)

        self.logger.info("Discriminator")
        common.print_parameter_summary(discriminator)

        i=-1
        for epoch in range(self.config.epoch):
            for batch in dataloader:
                i+=1
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                real = images.to(common.DEVICE)
                batch_size = real.size(0)

                if self.config.super_resolution:
                    real_lr = self.resize_to_input(real)

                    z = None

                    if self.config.latent_vector_size > 0:
                        z = torch.randn(
                            batch_size,
                            self.config.latent_vector_size,
                            device=common.DEVICE,
                        )

                    fake = generator(real_lr, z)

                else:
                    real_lr = None

                    noise = torch.randn(
                        batch_size,
                        self.config.latent_vector_size,
                        device=common.DEVICE,
                    )

                    fake = generator(noise)

                optimizerD.zero_grad(set_to_none=True)

                out_real = discriminator(real)
                out_fake = discriminator(fake.detach())

                loss_real = criterion(
                    out_real,
                    torch.ones_like(out_real),
                )

                loss_fake = criterion(
                    out_fake,
                    torch.zeros_like(out_fake),
                )

                loss_d = 0.5 * (loss_real + loss_fake)

                loss_d.backward()
                optimizerD.step()

                optimizerG.zero_grad(set_to_none=True)

                out = discriminator(fake)

                loss_adv = criterion(
                    out,
                    torch.ones_like(out),
                )

                if self.config.super_resolution:
                    fake_lr = self.resize_to_input(fake)

                    loss_reconstruction = reconstruction_criterion(
                        fake_lr,
                        real_lr,
                    )

                    loss_g = (
                        loss_adv
                        + self.config.reconstruction_weight * loss_reconstruction
                    )

                else:
                    loss_reconstruction = torch.zeros(
                        (),
                        device=common.DEVICE,
                    )

                    loss_g = loss_adv

                loss_g.backward()
                optimizerG.step()

                if i % 500 == 0:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    self.logger.info(
                        f"Time: {now} | "
                        f"Epoch [{epoch + 1}/{self.config.epoch}] | "
                        f"Step [{i}/{len(dataloader)}]"
                    )

                    self.print_losses(
                        loss_d,
                        loss_real,
                        loss_fake,
                        loss_g,
                    )

                    if self.config.super_resolution:
                        avg_reconstruction = self.multi_loss_tracker.calculate_loss(
                            "loss_reconstruction",
                            loss_reconstruction,
                        )

                        self.logger.info(
                            f"Reconstruction Loss: "
                            f"{loss_reconstruction.item():.4f} "
                            f"(avg: {avg_reconstruction:.4f})"
                        )

                    GanTrainer.save_gan_checkpoint(
                        generator,
                        optimizerG,
                        discriminator,
                        optimizerD,
                        self.config.checkpoint_path,
                    )

                    image_utils.show_image_eval(
                        "real", real, loss_real, image_size=self.config.image_size
                    )

                    image_utils.show_image_eval(
                        "fake",
                        fake.detach(),
                        loss_fake,
                        image_size=self.config.image_size,
                    )

                    image_utils.display_images(
                        fake.detach(), image_size=self.config.image_size
                    )

        self.logger.info("Training GAN done.")


if __name__ == "__main__":
    logger = common.create_logger()

    ganTrainer = GanTrainer.from_yaml(
        profile_name="srgan",
        logger=logger,
    )

    ganTrainer.train()
