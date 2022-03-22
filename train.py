#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 PyTorch and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions.

""" Training a Deep Convolutional Generative Adversarial Network (DCGAN) leveraging the 🤗 ecosystem.

Paper: https://arxiv.org/abs/1511.06434.

Based on PyTorch's official tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html.

"""

import argparse
import os

from datasets import load_dataset

from modeling_dcgan import Generator, Discriminator

from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import torch.nn as nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="mnist", help="Dataset to load from the HuggingFace hub."
)
parser.add_argument(
    "--num_workers", type=int, default=2, help="Number of workers when loading data"
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="Batch size to use during training"
)
parser.add_argument(
    "--image_size",
    type=int,
    default=64,
    help="Spatial size to use when resizing images for training.",
)
parser.add_argument(
    "--num_channels",
    type=int,
    default=3,
    help="Number of channels in the training images. For color images this is 3.",
)
parser.add_argument(
    "--latent_dim", type=int, default=100, help="Dimensionality of the latent space."
)
parser.add_argument(
    "--generator_hidden_size",
    type=int,
    default=64,
    help="Hidden size of the generator's feature maps.",
)
parser.add_argument(
    "--discriminator_hidden_size",
    type=int,
    default=64,
    help="Hidden size of the discriminator's feature maps.",
)
parser.add_argument(
    "--num_epochs", type=int, default=5, help="number of epochs of training"
)
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument(
    "--beta1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
args = parser.parse_args()
print(args)

# Make directory for saving generated images
os.makedirs("images", exist_ok=True)

# Custom weights initialization called on Generator and Discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Loss function
criterion = nn.BCELoss()

# Initialize generator and discriminator
netG = Generator(
    num_channels=args.num_channels,
    latent_dim=args.latent_dim,
    hidden_size=args.generator_hidden_size,
)
netD = Discriminator(
    num_channels=args.num_channels, hidden_size=args.discriminator_hidden_size
)

device = "cuda" if torch.cuda.is_available() else "cpu"
netG.to(device)
netD.to(device)
criterion.to(device)

# Initialize weights
netG.apply(weights_init)
netD.apply(weights_init)

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.0
fake_label = 0.0

# Setup Adam optimizers for both G and D
optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

# Configure data loader
dataset = load_dataset(args.dataset)

transform = Compose(
    [
        Resize(args.image_size),
        CenterCrop(args.image_size),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def transforms(examples):
    examples["pixel_values"] = [
        transform(image.convert("RGB")) for image in examples["image"]
    ]

    del examples["image"]

    return examples


transformed_dataset = dataset.with_transform(transforms)

dataloader = DataLoader(
    transformed_dataset["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
)

# ----------
#  Training
# ----------

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(args.num_epochs):
    # For each batch in the dataloader
    for i, batch in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = batch["pixel_values"].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print(
                "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                % (
                    epoch,
                    args.num_epochs,
                    i,
                    len(dataloader),
                    errD.item(),
                    errG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                )
            )

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or (
            (epoch == args.num_epochs - 1) and (i == len(dataloader) - 1)
        ):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            #img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            save_image(fake.data[:25], "images/%d.png" % i, nrow=5, normalize=True)

        iters += 1
