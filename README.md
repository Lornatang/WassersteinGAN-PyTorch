# WassersteinGAN-PyTorch

### Update (Feb 21, 2020)

The mnist and fmnist models are now available. Their usage is identical to the other models: 
```python
from wgan_pytorch import Generator
model = Generator.from_pretrained('g-mnist') 
```

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Wasserstein GAN](http://xxx.itp.ac.cn/pdf/1701.07875).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

At the moment, you can easily:  
 * Load pretrained Generate models 
 * Use Generate models for extended dataset

_Upcoming features_: In the next few days, you will be able to:
 * Quickly finetune an Generate on your own dataset
 * Export Generate models for production

### Table of contents
1. [About Wasserstein GAN](#about-wasserstein-gan)
2. [Model Description](#model-description)
3. [Installation](#installation)
4. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Extended dataset](#example-extended-dataset)
    * [Example: Visual](#example-visual)
5. [Contributing](#contributing) 

### About Wasserstein GAN

If you're new to Wasserstein GAN, here's an abstract straight from the paper:

We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.
### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

Install from pypi:
```bash
pip install wgan_pytorch
```

Install from source:
```bash
git clone https://github.com/Lornatang/WassersteinGAN-PyTorch.git
cd WassersteinGAN-PyTorch
pip install -e .
``` 

### Usage

#### Loading pretrained models

Load an Wasserstein GAN:
```python
from wgan_pytorch import Generator
model = Generator.from_name("g-mnist")
```

Load a pretrained Wasserstein GAN:
```python
from wgan_pytorch import Generator
model = Generator.from_pretrained("g-mnist")
```

#### Example: Extended dataset

As mentioned in the example, if you load the pre-trained weights of the MNIST dataset, it will create a new `imgs` directory and generate 64 random images in the `imgs` directory.

```python
import os
import torch
import torchvision.utils as vutils
from wgan_pytorch import Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Generator.from_pretrained("g-mnist")
model.to(device)
# switch to evaluate mode
model.eval()

try:
    os.makedirs("./imgs")
except OSError:
    pass

with torch.no_grad():
    for i in range(64):
        noise = torch.randn(64, 100, device=device)
        fake = model(noise)
        vutils.save_image(fake.detach(), f"./imgs/fake_{i:04d}.png", normalize=True)
    print("The fake image has been generated!")
```

#### Example: Visual

```text
cd $REPO$/framework
sh start.sh
```

Then open the browser and type in the browser address [http://127.0.0.1:10002/](http://127.0.0.1:10002/).
Enjoy it.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 