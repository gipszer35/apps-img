import os
import sys
import datetime
import dataclasses
import random
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    random_split,
)

from torchvision import transforms
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

            self.root_dir = os.path.join(
                self.content_drive,
                "MyDrive",
            )

            self.apps_img = os.path.join(
                self.root_dir,
                "apps-img",
            )

            self.images_dir = os.path.join(
                self.root_dir,
                "images_few",
            )

        else:
            self.root_dir = os.path.expanduser("~/work")

            self.apps_img = os.path.join(
                self.root_dir,
                "apps-img",
            )

            self.images_dir = os.path.join(
                self.apps_img,
                "images",
                "my-images",
            )

        self.work_dir = os.path.join(
            self.apps_img,
            "vae",
        )

        os.makedirs(
            self.work_dir,
            exist_ok=True,
        )

        for path in (
            self.root_dir,
            self.apps_img,
            self.work_dir,
        ):
            if path not in sys.path:
                sys.path.append(path)

        # Keep the same project environment as the GAN.
        import common
        import image_utils

        globals()["common"] = common
        globals()["image_utils"] = image_utils

    def _mount_colab(self):
        if not os.path.ismount(self.content_drive):
            from google.colab import drive

            drive.mount(self.content_drive)


# Initialize environment before importing project modules.
Environment()


@dataclasses.dataclass
class VaeConfig:
    # Automatically injected environment paths.
    work_dir: str
    content_drive: str

    # Dataset configuration.
    dataset_dirs: list
    train_ratio: float
    seed: int

    # Training configuration.
    batch_size: int
    epoch: int
    lr: float
    num_workers: int

    # Model configuration.
    num_channel: int
    latent_dim: int
    image_height: int
    image_width: int

    # Checkpoint configuration.
    checkpoint_name: str

    # Logging and visualization.
    log_interval: int = 20
    num_visualize: int = 5
    show_images: bool = True

    @classmethod
    def create(cls, **kwargs) -> "VaeConfig":
        env = Environment()

        kwargs.update(
            {
                "work_dir": env.work_dir,
                "content_drive": env.content_drive,
            }
        )

        # Apply dataclass defaults.
        for field in dataclasses.fields(cls):
            if field.name not in kwargs and field.default is not dataclasses.MISSING:
                kwargs[field.name] = field.default

        return cls(**kwargs)

    @property
    def out_dir(self):
        return os.path.join(
            self.work_dir,
            "out_dir",
        )

    @property
    def checkpoint_dir(self):
        return os.path.join(
            self.out_dir,
            "checkpoints",
        )

    @property
    def checkpoint_path(self):
        return os.path.join(
            self.checkpoint_dir,
            self.checkpoint_name,
        )

    @property
    def input_shape(self):
        return (
            self.num_channel,
            self.image_height,
            self.image_width,
        )


class FlatImageFolder(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        extensions=(".jpg", ".png", ".jpeg"),
    ):
        self.root = root
        self.transform = transform
        self.extensions = tuple(ext.lower() for ext in extensions)

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        self.image_paths = sorted(
            os.path.join(root, filename)
            for filename in os.listdir(root)
            if filename.lower().endswith(self.extensions)
        )

        if not self.image_paths:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

        return image


class VAE(nn.Module):
    def __init__(
        self,
        config: VaeConfig,
    ):
        super().__init__()

        self.config = config
        self.latent_dim = config.latent_dim

        channels = config.num_channel

        # Ensure the decoder's output size matches the input.
        if config.image_height % 8 != 0 or config.image_width % 8 != 0:
            raise ValueError(
                "image_height and image_width must " "both be divisible by 8."
            )

        self.encoder = nn.Sequential(
            nn.Conv2d(
                channels,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # Calculate the encoder output shape dynamically.
        with torch.no_grad():
            dummy_input = torch.zeros(
                1,
                *config.input_shape,
            )

            conv_out = self.encoder(dummy_input)

            self.conv_output_shape = tuple(conv_out.shape[1:])

            self.flattened_size = conv_out.flatten(start_dim=1).shape[1]

        self.fc_mu = nn.Linear(
            self.flattened_size,
            self.latent_dim,
        )

        self.fc_logvar = nn.Linear(
            self.flattened_size,
            self.latent_dim,
        )

        self.decoder_input = nn.Linear(
            self.latent_dim,
            self.flattened_size,
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                32,
                channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)

        x = x.reshape(
            z.size(0),
            *self.conv_output_shape,
        )

        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar,
        )

        recon = self.decode(z)

        return recon, mu, logvar


def vae_loss(
    recon_x,
    x,
    mu,
    logvar,
):
    # Reconstruction error per image.
    recon_loss = F.mse_loss(
        recon_x,
        x,
        reduction="sum",
    )

    # KL divergence.
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Normalize by batch size to make the loss scale
    # independent of the number of images in the batch.
    batch_size = x.size(0)

    loss = (recon_loss + kld) / batch_size

    return loss


class VaeTrainer:
    def __init__(
        self,
        logger,
        config: VaeConfig,
    ):
        self.logger = logger
        self.config = config

        os.makedirs(
            self.config.out_dir,
            exist_ok=True,
        )

        os.makedirs(
            self.config.checkpoint_dir,
            exist_ok=True,
        )

        self.model = None
        self.optimizer = None

        self.train_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.test_loader = None

        self.start_epoch = 0
        self.global_step = 0

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

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as f:
            full_config = yaml.safe_load(f)

        profile = full_config["trainers"][profile_name]

        config = VaeConfig.create(**profile)

        return cls(
            logger=logger,
            config=config,
        )

    def prepare_dataloader(self):
        config = self.config

        self.logger.info("Preparing VAE datasets")

        # Normalize all images to a consistent size.
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        config.image_height,
                        config.image_width,
                    )
                ),
                transforms.ToTensor(),
            ]
        )

        datasets_list = []

        for folder in config.dataset_dirs:
            self.logger.info(f"Loading images from: {folder}")

            dataset = FlatImageFolder(
                root=folder,
                transform=transform,
            )

            self.logger.info(f"Found {len(dataset)} images")

            datasets_list.append(dataset)

        if not datasets_list:
            raise RuntimeError("No dataset directories configured.")

        full_dataset = ConcatDataset(datasets_list)

        train_size = int(config.train_ratio * len(full_dataset))

        test_size = len(full_dataset) - train_size

        if train_size == 0:
            raise ValueError(
                "Training dataset is empty. " "Increase train_ratio or add more images."
            )

        generator = torch.Generator().manual_seed(config.seed)

        self.train_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=generator,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=(torch.cuda.is_available()),
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(torch.cuda.is_available()),
        )

        self.logger.info(f"Total images: {len(full_dataset)}")

        self.logger.info(f"Training images: {len(self.train_dataset)}")

        self.logger.info(f"Test images: {len(self.test_dataset)}")

        self.logger.info(f"Training batches: {len(self.train_loader)}")

        return self.train_loader, self.test_loader

    def initialize_model(self):
        self.model = VAE(self.config).to(common.DEVICE)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
        )

        self.logger.info(f"Using device: {common.DEVICE}")

        self.logger.info("VAE model")

        common.print_parameter_summary(self.model)

        self.load_checkpoint_if_exists()

    def save_checkpoint(self, epoch):
        path = self.config.checkpoint_path

        torch.save(
            {
                "model_state_dict": (self.model.state_dict()),
                "optimizer_state_dict": (self.optimizer.state_dict()),
                "epoch": epoch,
                "global_step": self.global_step,
                "config": dataclasses.asdict(self.config),
            },
            path,
        )

        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint_if_exists(self):
        path = self.config.checkpoint_path

        if not os.path.isfile(path):
            self.logger.info(f"No checkpoint found at {path}")

            return

        checkpoint = torch.load(
            path,
            map_location=common.DEVICE,
        )

        # Support the checkpoint format used in the
        # original VAE implementation.
        model_state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("model"),
        )

        if model_state is None:
            raise KeyError("Checkpoint contains no model state.")

        self.model.load_state_dict(model_state)

        optimizer_state = checkpoint.get(
            "optimizer_state_dict",
            checkpoint.get("optimizer"),
        )

        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)

            except (ValueError, KeyError) as exc:
                self.logger.warning("Could not restore optimizer state: " f"{exc}")

        self.start_epoch = checkpoint.get(
            "epoch",
            0,
        )

        self.global_step = checkpoint.get(
            "global_step",
            0,
        )

        self.logger.info(f"Loaded checkpoint: {path}")

        self.logger.info(f"Resuming at epoch {self.start_epoch + 1}")

    def show_images(self):
        if self.train_loader is None:
            return

        was_training = self.model.training

        self.model.eval()

        try:
            # Prefer the test set for visualization.
            loader = (
                self.test_loader
                if self.test_loader is not None and len(self.test_dataset) > 0
                else self.train_loader
            )

            batch = next(iter(loader))

            batch = batch.to(common.DEVICE)

            with torch.no_grad():
                recon, _, _ = self.model(batch)

            num_images = min(
                self.config.num_visualize,
                batch.size(0),
            )

            fig, axes = plt.subplots(
                2,
                num_images,
                figsize=(
                    num_images * 2.5,
                    5,
                ),
                squeeze=False,
            )

            for i in range(num_images):
                original = batch[i].detach().cpu().permute(1, 2, 0).numpy()

                reconstructed = (
                    recon[i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                )

                axes[0, i].imshow(original)

                axes[0, i].axis("off")

                axes[1, i].imshow(reconstructed)

                axes[1, i].axis("off")

                if i == 0:
                    axes[0, i].set_ylabel(
                        "Original",
                        fontsize=11,
                    )

                    axes[1, i].set_ylabel(
                        "Reconstructed",
                        fontsize=11,
                    )

            plt.tight_layout()
            plt.show()
            plt.close(fig)

        finally:
            self.model.train(was_training)

    def train(self):
        config = self.config

        self.logger.info("Starting VAE training")

        self.prepare_dataloader()
        self.initialize_model()

        for epoch in range(
            self.start_epoch,
            config.epoch,
        ):
            self.model.train()

            total_loss = 0.0
            num_samples = 0

            for batch_idx, batch in enumerate(self.train_loader):
                batch = batch.to(
                    common.DEVICE,
                    non_blocking=True,
                )

                self.optimizer.zero_grad(set_to_none=True)

                recon, mu, logvar = self.model(batch)

                loss = vae_loss(
                    recon,
                    batch,
                    mu,
                    logvar,
                )

                loss.backward()

                self.optimizer.step()

                batch_size = batch.size(0)

                total_loss += loss.item() * batch_size

                num_samples += batch_size

                self.global_step += 1

                if self.global_step % config.log_interval == 0:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    self.logger.info(
                        f"Time: {now} | "
                        f"Epoch [{epoch + 1}/{config.epoch}] | "
                        f"Step [{batch_idx + 1}/"
                        f"{len(self.train_loader)}] | "
                        f"Loss: {loss.item():.4f}"
                    )

                    self.save_checkpoint(epoch + 1)

                    if config.show_images:
                        self.show_images()

            avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

            self.logger.info(
                f"Epoch [{epoch + 1}/{config.epoch}] "
                f"completed | "
                f"Average loss per image: {avg_loss:.4f}"
            )

            # Save at the end of every epoch.
            self.save_checkpoint(epoch + 1)

        self.logger.info("VAE training finished.")


if __name__ == "__main__":
    logger = common.create_logger()

    trainer = VaeTrainer.from_yaml(
        profile_name="vae_small",
        logger=logger,
    )

    trainer.train()
