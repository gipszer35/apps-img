import os
import torch
import logging
import torch.nn as nn
import math
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def save_checkpoint(model, optimizer, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from: {path}")

    if optimizer and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state.")
        except Exception as e:
            print(f"Optimizer state not loaded: {e}")


def load_checkpoint_if_exists(model, optimizer, path):

    if os.path.exists(path):
        print("Load checkpoint")
        load_checkpoint(model, optimizer, path)
    else:
        print("No checkpoint found — initialized new model and optimizer.")
    return model, optimizer


def print_parameter_summary(model, detailed=True):

    def print_basic_summary():
        # Prints the structural overview: names, shapes, and counts
        print(f"{'Parameter':40s} {'Shape':20s} {'# Params'}")
        print("-" * 70)
        total = 0
        for name, p in model.named_parameters():
            n = p.numel()
            total += n
            print(f"{name:40s} {str(list(p.shape)):20s} {n}")
        print("-" * 70)
        print("Total parameters:", total)
        print("\n" + "=" * 80 + "\n")

    def print_detailed_stats():
        header = f"{'Parameter':40s} {'Type':12s} {'Req Grad':10s} {'Mean':10s} {'Std':10s} {'Min':10s} {'Max':10s}"
        print(header)
        print("-" * len(header))

        total_params = 0
        trainable_params = 0
        total_memory_bytes = 0

        with torch.no_grad():
            for name, p in model.named_parameters():
                n = p.numel()
                total_params += n
                if p.requires_grad:
                    trainable_params += n

                total_memory_bytes += n * p.element_size()

                dtype_str = str(p.dtype).split(".")[-1]
                req_grad_str = "True" if p.requires_grad else "False"

                # Compute statistics safely
                p_flat = p.detach().cpu().float()
                p_mean = p_flat.mean().item()
                p_std = p_flat.std().item()  # Standard Deviation
                p_min = p_flat.min().item()
                p_max = p_flat.max().item()

                print(
                    f"{name:40s} {dtype_str:12s} {req_grad_str:10s} {p_mean:<10.4f} {p_std:<10.4f} {p_min:<10.4f} {p_max:<10.4f}"
                )

        print("-" * len(header))
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters:    {total_params - trainable_params:,}")
        print(f"Model Memory Size:    {total_memory_bytes / (1024 ** 2):.2f} MB")

    print_basic_summary()
    if detailed:
        print_detailed_stats()


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma > 0:
            noise = torch.randn_like(x)
            return x * (1 - self.sigma) + noise * self.sigma
        return x


class MultiLossTracker:
    def __init__(self, maxlen=1000):
        self.maxlen = maxlen
        self.queues = {}  # stores name -> deque

    def calculate_loss(self, loss_name, loss_value):
        if loss_name not in self.queues:
            self.queues[loss_name] = deque(maxlen=self.maxlen)

        self.queues[loss_name].append(loss_value)

        valid_losses = [x for x in self.queues[loss_name] if x is not None]
        if not valid_losses:
            return 0.0
        return sum(valid_losses) / len(valid_losses)


def create_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # always reset in notebooks (Colab/IPython safe)
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False  # prevents duplicate root logs

    return logger


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb
