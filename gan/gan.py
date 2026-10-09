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
    # Environment paths generated automatically from the Environment Singleton
    work_dir: str
    images_dir: str
    content_drive: str
    batch_size: int
    num_channel: int
    beta: float
    epoch: int
    checkpoint_name: str

    # Explicit architecture and training hyperparameters
    image_size: int
    latent_vector_size: int

    # Static constants
    lr_generator: float = 1e-4
    lr_discriminator: float = 1e-4

    @classmethod
    def create(cls, **kwargs) -> "GanConfig":
        """Factory method that dynamically resolves defaults for ALL dataclass fields."""

        env = Environment()

        # Inject the mandatory environment paths
        kwargs.update(
            {
                "work_dir": env.work_dir,
                "images_dir": env.images_dir,
                "content_drive": env.content_drive,
            }
        )

        kwargs.setdefault("batch_size", 2048 if env.is_colab else 2)

        # Dynamically loop through all defined fields in the class
        for field in dataclasses.fields(cls):
            if field.name not in kwargs and field.default is not dataclasses.MISSING:
                kwargs[field.name] = field.default

        return cls(**kwargs)

    @property
    def out_dir(self):
        return os.path.join(self.work_dir, "out_dir")

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.out_dir, self.checkpoint_name)

    @property
    def cropped_images_path(self) -> str:
        return os.path.join(self.out_dir, f"cropped_{self.image_size}.pt")


class Generator(nn.Module):
    def __init__(
        self,
        config,
        base=256,
    ):
        super().__init__()
        self.config = config
        # project latent vector to 4×4
        self.fc = nn.Sequential(
            nn.Linear(config.latent_vector_size, base * 4 * 4),
        )

        # We will build: 4→8→16→32 (upsampling)
        self.main = nn.Sequential(
            # 4×4 → 8×8
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base, base, kernel_size=3, padding=1, bias=False),
            common.GaussianNoise(0.1),
            nn.GroupNorm(32, base),
            nn.LeakyReLU(0.2, inplace=True),
            # 8×8 → 16×16
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base, base // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, base // 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16×16 → 32×32
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base // 2, base // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, base // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base // 4, config.num_channel, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), -1, 4, 4)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, config, base=256):
        super().__init__()

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
                nn.GroupNorm(32, out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.blocks = nn.Sequential(
            block(config.num_channel, base // 4),  # 32 -> 16
            block(base // 4, base // 2),  # 16 -> 8
            block(base // 2, base),  # 8 -> 4
            block(base, base * 2),  # 4 -> 2
        )

        self.final = nn.Conv2d(base * 2, 1, kernel_size=2)  # 2 -> 1

    def forward(self, x):
        x = self.blocks(x)
        x = self.final(x)
        return x.view(x.size(0))


class GanTrainer:
    def __init__(self, logger, config: GanConfig):
        self.logger = logger
        self.config = config
        self.multi_loss_tracker = common.MultiLossTracker()

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

    def prepare_dataloader(self):
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

        dataloader = self.prepare_dataloader()

        GanTrainer.load_gan_checkpoint_if_exists(
            generator,
            optimizerG,
            discriminator,
            optimizerD,
            self.config.checkpoint_path,
        )
        logger.info("Generator")
        common.print_parameter_summary(generator)
        logger.info("Discriminator")
        common.print_parameter_summary(discriminator)

        # Training loop
        for epoch in range(self.config.epoch):
            for i, (images, _) in enumerate(dataloader):
                real = images.to(common.DEVICE)
                batch_size = real.size(0)

                # labelsb_size
                real_labels = torch.ones(batch_size, device=common.DEVICE)
                fake_labels = torch.zeros(batch_size, device=common.DEVICE)

                # Train Discriminator
                optimizerD.zero_grad(set_to_none=True)

                out_real = discriminator(real)
                loss_real = criterion(out_real, real_labels)

                noise = torch.randn(
                    batch_size,
                    self.config.latent_vector_size,
                    device=common.DEVICE,
                )
                fake = generator(noise)

                out_fake = discriminator(fake.detach())
                loss_fake = criterion(out_fake, fake_labels)

                loss_d = loss_real + loss_fake
                loss_d.backward()
                optimizerD.step()

                # Train Generator
                optimizerG.zero_grad(set_to_none=True)

                out = discriminator(fake)
                loss_g = criterion(out, real_labels)
                loss_g.backward()
                optimizerG.step()

                if i % 500 == 0:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    self.logger.info(
                        f"Time: {now} | "
                        f"Epoch [{epoch + 1}/{self.config.epoch}] | "
                        f"Step [{i}/{len(dataloader)}]"
                    )

                    self.print_losses(loss_d, loss_real, loss_fake, loss_g)

                    GanTrainer.save_gan_checkpoint(
                        generator,
                        optimizerG,
                        discriminator,
                        optimizerD,
                        self.config.checkpoint_path,
                    )

                    image_utils.show_image_eval("real", real, loss_real)
                    image_utils.show_image_eval("fake", fake.detach(), loss_fake)
                    image_utils.display_images(fake.detach())

        self.logger.info("Training GAN done.")


if __name__ == "__main__":
    logger = common.create_logger()
    ganTrainer = GanTrainer.from_yaml(profile_name="gan_small", logger=logger)
    ganTrainer.train()
