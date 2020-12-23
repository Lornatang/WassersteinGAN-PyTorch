# GAN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation
of [Wasserstein GAN](http://xxx.itp.ac.cn/pdf/1701.07875v3).

### Table of contents

1. [About Wasserstein GAN](#about-wasserstein-gan)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights-eg-mnist)
4. [Test](#test)
5. [Train](#train-eg-mnist)
6. [Contributing](#contributing)
7. [Credit](#credit)

### About Wasserstein GAN

If you're new to WGANs, here's an abstract straight from the paper:

We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we
can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves
useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is
sound, and provide extensive theoretical work highlighting the deep connections to other distances between
distributions.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives
a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that
discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that
x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/WassersteinGAN-PyTorch.git
$ cd WassersteinGAN-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights (e.g. mnist)

```bash
$ cd weights/
$ python3 download_weights.py
```

### Test

Using pre training model to generate pictures.

```text
usage: test.py [-h] [-a ARCH] [-n NUM_IMAGES] [--outf PATH] [--device DEVICE]

Research and application of GAN based super resolution technology for
pathological microscopic images.

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: mnist | fashion_mnist |cifar10 |
                        (default: mnist)
  -n NUM_IMAGES, --num-images NUM_IMAGES
                        How many samples are generated at one time. (default:
                        64).
  --outf PATH           The location of the image in the evaluation process.
                        (default: ``test``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``cpu``).

# Example (e.g. MNIST)
$ python3 test.py -a mnist
```

<span align="center"><img src="assets/mnist.gif" alt="">
</span>

### Train (e.g. MNIST)

```text
usage: train.py [-h] --dataset DATASET [--dataroot DATAROOT] [-j N]
                [--manualSeed MANUALSEED] [--device DEVICE] [-p N] [-a ARCH]
                [--pretrained] [--netD PATH] [--netG PATH] [--start-epoch N]
                [--iters N] [-b N] [--image-size IMAGE_SIZE]
                [--channels CHANNELS] [--lr LR] [--n_critic N_CRITIC]
                [--clip_value CLIP_VALUE]

Research and application of GAN based super resolution technology for
pathological microscopic images.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     mnist | fashion-mnist | cifar10 |.
  --dataroot DATAROOT   Path to dataset. (default: ``data``).
  -j N, --workers N     Number of data loading workers. (default:4)
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:1111)
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default: ````).
  -p N, --save-freq N   Save frequency. (default: 50).
  -a ARCH, --arch ARCH  model architecture: mnist | fashion_mnist |cifar10 |
                        (default: mnist)
  --pretrained          Use pre-trained model.
  --netD PATH           Path to latest discriminator checkpoint. (default:
                        ````).
  --netG PATH           Path to latest generator checkpoint. (default: ````).
  --start-epoch N       manual epoch number (useful on restarts)
  --iters N             The number of iterations is needed in the training of
                        PSNR model. (default: 1e5)
  -b N, --batch-size N  mini-batch size (default: 64), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --image-size IMAGE_SIZE
                        The height / width of the input image to network.
                        (default: 28).
  --channels CHANNELS   The number of channels of the image. (default: 1).
  --lr LR               Learning rate. (default:0.00005)
  --n_critic N_CRITIC   Number of training steps for discriminator per iter.
                        (Default: 5).
  --clip_value CLIP_VALUE
                        Lower and upper clip value for disc. weights.
                        (Default: 0.01).

# Example (e.g. MNIST)
$ python3 train.py -a mnist --dataset mnist --image-size 28 --channels 1 --pretrained
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py -a mnist \
                   --dataset mnist \
                   --image-size 28 \
                   --channels 1 \
                   --start-epoch 18 \
                   --netG weights/netG_epoch_18.pth \
                   --netD weights/netD_epoch_18.pth
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Wasserstein GAN

*Arjovsky, Martin; Chintala, Soumith; Bottou, Léon*

**Abstract**

We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we
can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves
useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is
sound, and provide extensive theoretical work highlighting the deep connections to other distances between
distributions.

[[Paper]](http://xxx.itp.ac.cn/pdf/1701.07875)

```
@article{adversarial,
  title={Wasserstein GAN},
  author={Arjovsky, Martin; Chintala, Soumith; Bottou, Léon},
  journal={cvpr},
  year={2017}
}
```