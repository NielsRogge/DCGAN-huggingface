# DCGAN-huggingface

An implementation of DCGAN, leveraging the HuggingFace ecosystem for getting data and pushing to the hub.

To train the model with the default parameters, simply do:

```
python train.py
```

# Citation

This repo is entirely based on PyTorch's official [DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
The only differences being that:
* HuggingFace Datasets is leveraged for getting a dataset from the [hub](https://huggingface.co/) and perform data augmentation on-the-fly using `set_transform` (or `with_transform`). This means that image transformations are only performed when the images are loaded into RAM.
* `PyTorchModelHubMixin` of the huggingface_hub Python client is leveraged to push and load back the trained generator using `push_to_hub` and `from_pretrained` respectively.

This means that after training, generating a new image can be done as follows:

```
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class Generator(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_channels=3, latent_dim=100, hidden_size=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, hidden_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.ReLU(True),
            # state size. (hidden_size*8) x 4 x 4
            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            # state size. (hidden_size*4) x 8 x 8
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            # state size. (hidden_size*2) x 16 x 16
            nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            # state size. (hidden_size) x 32 x 32
            nn.ConvTranspose2d(hidden_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, noise):
        pixel_values = self.model(noise)

        return pixel_values

model = Generator.from_pretrained("huggan/dcgan-mnist")

device = "cuda" if torch.cuda.is_available() else "cpu
model.to(device)
 
with torch.no_grad():
    z = torch.randn(1, 100, 1, 1, device=device)
    outputs = model(z)