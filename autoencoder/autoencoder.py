import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import datetime
import sys
import random
from torchvision.transforms import functional as F


# BEGIN BASIC SETUP
def is_colab():
    return "COLAB_GPU" in os.environ


if is_colab():
    if not os.path.ismount("/content/drive"):
        from google.colab import drive

        drive.mount("/content/drive")
    ROOT_DIR = "/content/drive/MyDrive/"
    IMAGE_GENERATOR_DIR = ROOT_DIR + "ImageGenerator/"
    NUM_PATCHES_PER_IMAGE = 4
    BATCH_SIZE = 64
else:
    ROOT_DIR = "./"
    IMAGE_GENERATOR_DIR = ROOT_DIR
    NUM_PATCHES_PER_IMAGE = 5
    BATCH_SIZE = 4


sys.path.append(ROOT_DIR)
import my_common as my

my.test_function()
# END BASIC SETUP

# IMAGES_DIR = ROOT_DIR + "images"
IMAGES_DIR = ROOT_DIR + "pattern_images"

AUTOENCODER_CHECKPOINT_PATH = ROOT_DIR + "/autoencoder.pt"
IMAGE_SIZE = 256  # IMAGE_SIZE x IMAGE_SIZE
CROP_IMAGE_SIZE = IMAGE_SIZE + 20


def show_resized_image(model, sr_dataset):
    model.eval()
    with torch.no_grad():
        idx = random.randint(0, len(sr_dataset) - 1)
        test_lr, test_hr = sr_dataset[idx]

        test_lr = test_lr.unsqueeze(0).to(my.DEVICE)
        output_hr = model(test_lr).cpu().squeeze(0)

        test_lr_denorm = my.denormalize(test_lr.cpu().squeeze(0))
        test_hr_denorm = my.denormalize(test_hr.cpu().squeeze(0))
        output_hr_denorm = my.denormalize(output_hr)

        # Image display helper function
        def imshow(img, title):
            # Convert from Tensor (C, H, W) to Numpy (H, W, C)
            img = img.numpy().transpose((1, 2, 0))
            plt.imshow(img)
            plt.title(title)
            plt.axis("off")

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        imshow(test_lr_denorm, "Input 32x32")
        plt.subplot(1, 3, 2)
        imshow(test_hr_denorm, "Target (Ground Truth)")
        plt.subplot(1, 3, 3)
        imshow(output_hr_denorm, "Generated Image")
        plt.show()


class ResBlock(nn.Module):
    def __init__(self, channels, *, p=0.7):
        super().__init__()
        self.p = p
        self.sequential = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return x * (1 - self.p) + self.sequential(x) * self.p


def upshuffle(in_channels, out_channels, scale=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels * (scale**2), 3, padding=1),
        nn.ReLU(inplace=True),
        nn.PixelShuffle(scale),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def downsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class SuperResolutionAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        divisor = 2
        channel256x256 = 256 // divisor
        channel128x128 = 128 // divisor
        channel64x64 = 128 // divisor
        channel32x32 = 256 // divisor
        channel16x16 = 512 // divisor
        channel8x8 = 256 // divisor

        self.down_conv1 = nn.Conv2d(3, channel32x32, 3, padding=1)  # 32x32
        self.down_sample1 = downsample(channel32x32, channel16x16)
        self.down_conv2 = ResBlock(channel16x16)  # 16x16
        self.down_sample2 = downsample(channel16x16, channel8x8)
        self.bottleneck = ResBlock(channel8x8)  # 8x8
        self.up_sample1 = upshuffle(channel8x8, channel16x16)
        self.up_conv1 = ResBlock(channel16x16 * 2)  # 16x16
        self.up_sample2 = upshuffle(channel16x16 * 2, channel32x32)
        self.up_conv2 = ResBlock(channel32x32 * 2)  # 32x32
        self.up_sample3 = upshuffle(channel32x32 * 2, channel64x64)
        self.up_conv3 = ResBlock(channel64x64)  # 64x64
        self.up_sample4 = upshuffle(channel64x64, channel128x128)
        self.up_conv4 = ResBlock(channel128x128)  # 128x128
        self.up_sample5 = upshuffle(channel128x128, channel256x256)
        self.out = nn.Conv2d(channel256x256, 3, 5, padding=2)  # 256x256

    def forward(self, x):
        # Encoder

        down_conv1 = self.down_conv1(x)  # 32×32
        down_sample1 = self.down_sample1(down_conv1)
        down_conv2 = self.down_conv2(down_sample1)  # 16x16
        down_sample2 = self.down_sample2(down_conv2)
        bottleneck = self.bottleneck(down_sample2)  # 8x8
        up_sample1 = self.up_sample1(bottleneck)
        cat1 = torch.cat([up_sample1, down_conv2], dim=1)
        up_conv1 = self.up_conv1(cat1)  # 16x16
        up_sample2 = self.up_sample2(up_conv1)
        cat2 = torch.cat([up_sample2, down_conv1], dim=1)
        up_conv2 = self.up_conv2(cat2)  # 32x32
        up_sample3 = self.up_sample3(up_conv2)
        up_conv3 = self.up_conv3(up_sample3)  # 64x64
        up_sample4 = self.up_sample4(up_conv3)
        up_conv4 = self.up_conv4(up_sample4)  # 128x128
        up_sample5 = self.up_sample5(up_conv4)
        out = self.out(up_sample5)  # 256x256

        # print("out:", out[0][1][12][1:3])
        tanh_out = torch.tanh(out)
        return tanh_out


class SuperResolutionDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, lr_size=32):
        self.base_dataset = base_dataset
        self.lr_size = lr_size

        self.hr_transform = transforms.Compose([
            transforms.Lambda(
                lambda img: F.rotate(img, random.choice([0, 90, 180, 270]))
            ),
            transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            ]
        )


        self.lr_transform = transforms.Compose(
            [
                transforms.Resize(
                    (lr_size, lr_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            ]
        )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        if isinstance(image, torch.Tensor):
            image = my.denormalize(image)
            image = F.to_pil_image(image)

        image_hr = self.hr_transform(image)
        image_lr = self.lr_transform(image_hr)
        image_hr = my.normalize(self.to_tensor(image_hr))
        image_lr = my.normalize(self.to_tensor(image_lr))

        return image_lr, image_hr

def init_model():
    model = SuperResolutionAutoEncoder().to(my.DEVICE)
    my.print_parameter_summary(model)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    my.load_checkpoint_if_exists(model, optimizer, AUTOENCODER_CHECKPOINT_PATH)
    return model,optimizer

def init():
    dataset = my.cropped_dataset(
        IMAGES_DIR,
        crop_size=CROP_IMAGE_SIZE,
        num_patches_per_image=NUM_PATCHES_PER_IMAGE,
    )
    sr_dataset = SuperResolutionDataset(dataset)
    dataloader = DataLoader(
        sr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    model, optimizer = init_model()
    return model, dataloader, optimizer, sr_dataset

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def upscale_and_show_one_image(image_path, model):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img = img.resize((32, 32), Image.Resampling.BICUBIC)

    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        output_tensor = model(my.normalize(input_tensor))
    output_tensor = my.denormalize(output_tensor)

    output_img = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Input 32×32")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Upscaled 256×256")
    plt.imshow(output_img)
    plt.axis("off")

    plt.show()


def train(model, dataloader, optimizer, sr_dataset):
    x=0
    criterion = my.SRLoss(perceptual_weight=0.1)
    num_epochs = 10000
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            model.train()
            lr_imgs = lr_imgs.to(my.DEVICE)
            hr_imgs = hr_imgs.to(my.DEVICE)

            # Forward pass
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            mod=50
            if x % mod == 0:
                now = datetime.datetime.now()
                print("Current date and time::", now.strftime("%Y-%m-%d %H:%M:%S"))
                print(f"Epoch [{epoch+1}/{num_epochs}], Temp loss: {loss.item():.4f}")
                my.save_checkpoint(model, optimizer, AUTOENCODER_CHECKPOINT_PATH)
                show_resized_image(model, sr_dataset)
            x+=1
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(dataloader):.4f}")
    print("Training finished.")


if __name__ == "__main__":
    model, dataloader, optimizer, sr_dataset = init()
    train(model, dataloader, optimizer, sr_dataset)

