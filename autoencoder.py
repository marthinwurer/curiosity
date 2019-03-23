import glob
import math
import random

import matplotlib.pyplot as plt
import numpy
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F


# Ignore warnings
import warnings

from torchvision.transforms import ToTensor, transforms

from utilites import conv2d_factory, unflatten

warnings.filterwarnings("ignore")




VG_PATH = "/mnt/nas/datasets/visualgenome/VG_100K/"


class VGImagesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        if transform is None:
            transform = numpy.array
        self.transform = transform
        self.images = glob.glob(root_dir + "*.jpg")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        print(filename)
        image = Image.open(filename)

        sample = self.transform(image)

        return sample


class Encoder256(nn.Module):
    def __init__(self, input_shape, max_final_width=8, activation=F.relu):
        """

        Args:
            input_shape: a 3-element tuple with the number of channels, the height, and the width. (channels first)
            max_final_width: the maximum desired final width of the network.
            activation: The activation function used by each layer
        """
        super().__init__()

        # from rgb to ours 3 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512
        #                       256  128   64    32    16     8      4

        next_shape = input_shape
        self.from_rgb, next_shape, layer_params = conv2d_factory(next_shape, kernel_size=1,
                                                            padding=1, out_channels=8)
        self.conv11, next_shape, layer_params = conv2d_factory(next_shape, padding=1)

        self.conv21, next_shape, layer_params = conv2d_factory(next_shape / 2, out_channels=16, padding=1)
        self.conv31, next_shape, layer_params = conv2d_factory(next_shape / 2, out_channels=32, padding=1)
        self.conv41, next_shape, layer_params = conv2d_factory(next_shape / 2, out_channels=64, padding=1)
        self.conv51, next_shape, layer_params = conv2d_factory(next_shape / 2, out_channels=128, padding=1)
        self.conv61, next_shape, layer_params = conv2d_factory(next_shape / 2, out_channels=256, padding=1)
        self.conv71, next_shape, layer_params = conv2d_factory(next_shape / 2, out_channels=512, padding=1)

        self.activation = activation
        self.output_shape = next_shape


    def forward(self, x):

        for layer in self.conv_layers:
            x = self.activation(layer(x))
        return x


class LayerFactory(object):
    def __init__(self, clazz, compute_next=None, other_args=None):
        self.clazz = clazz
        self.compute_next = compute_next
        self.other_args = other_args

    def output_shape(self, input_shape):
        if self.compute_next:
            return self.compute_next(input_shape)
        else:
            return self.clazz.output_shape(input_shape)


class ProGANAutoencoder(nn.Module):
    def __init__(self, latent_size, input_power, output_power, start_filters=16, max_filters=512, activation=nn.ReLU):
        super().__init__()
        self.encoder = ProGANEncoder(input_power, output_power, start_filters, max_filters, activation)

        self.latent_size = latent_size

        self.decoder = ProGANDecoder(latent_size, output_power, input_power, start_filters, max_filters, activation)

    def forward(self, input):
        pass



class ProGANEncoder(nn.Module):
    """
    ProGAN discriminator layers go like this:

    image
    1x1 -> 16 (n=256)
    [
        3x3 -> 1x
        3x3 -> 2x
        downsample (n -> n//2)
    ]
    3x3 -> 1x (n=4)
    fully connected

    so what I really need this part to do is to do that inner loop, then tack
    on an extra conv layer or two at the end for that part. Also the start and
    end are really simple. I just go until the next size will be our 4x4

    """
    def __init__(self, input_power, output_power, start_filters=16, max_filters=512, activation=nn.ReLU):
        """

        Args:
            input_power: the power of two that the side length of the input will be
            output_power: the power of two that the side length of the output will be
            activation: The activation function used by each layer
        """
        super().__init__()

        layers = nn.ModuleList([])

        num_layers = input_power - output_power
        in_channels = start_filters
        out_channels = in_channels

        from_rgb = nn.Conv2d(3, start_filters, 1, 1)
        layers.append(from_rgb)
        layers.append(activation())

        for i in range(num_layers):
            starter = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)  # need padding to not shrink size
            out_channels = min(in_channels * 2, max_filters)
            grow = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
            in_channels = out_channels
            pool = nn.AvgPool2d(2, 2)
            layers.append(nn.Sequential(
                starter,
                activation(),
                grow,
                activation(),
                pool
            ))

        final = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        layers.append(final)

        self.layers = layers
        self.output_shape = (out_channels, 2**output_power, 2**output_power)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ProGANConv(nn.Module):
    pass


class ProGANDecoder(nn.Module):
    """
    latent (512)
    unflatten (512x1x1)
    deconv (1x1 -> 4x4)
    conv 3x3 -> x1
    {
        upsample x2
        conv 3x3 -> //2
        conv 3x3 -> x1
    }
    to_rgb 1x1 -> 3
    """
    def __init__(self, latent_size, input_power, output_power, end_filters=16, max_filters=512, activation=nn.ReLU):
        """

        Args:
            input_power: the power of two that the side length of the input will be
            output_power: the power of two that the side length of the output will be
            activation: The activation function used by each layer
        """
        super().__init__()

        # layers = nn.ModuleList([])

        reverse_layers = []

        num_layers = input_power - output_power
        in_channels = end_filters
        out_channels = in_channels

        to_rgb = nn.Conv2d(out_channels, 3, 1, 1)

        for i in range(num_layers):

            same = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

            in_channels = min(out_channels * 2, max_filters)

            half = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)  # need padding to not shrink size

            upsample = nn.Upsample(scale_factor=2, mode='nearest')

            out_channels = in_channels

            # add those in reverse order
            reverse_layers.append(nn.Sequential(
                upsample,
                half,
                activation(),
                same,
                activation()
            ))

        ordered_layers = reverse_layers.reverse()

        unflatten = nn.ConvTranspose2d(latent_size, out_channels, kernel_size=2**input_power)

        top_same = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        top_block = nn.Sequential(
            unflatten,
            activation(),
            top_same,
            activation()
        )

        self.layers = nn.ModuleList([])
        self.layers.append(top_block)
        self.layers.extend(ordered_layers)
        self.layers.append(to_rgb)
        self.output_shape = (end_filters, 2**output_power, 2**output_power)

    def forward(self, x):
        # need to unflatten the input
        x = unflatten(x)
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    # plt.ion()  # interactive mode

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256, pad_if_needed=True)
    ])

    face_dataset = VGImagesDataset(root_dir=VG_PATH, transform=transform)

    fig = plt.figure()

    index = random.randrange(len(face_dataset))
    print(index)

    image = face_dataset[index]

    plt.imshow(image)
    plt.show()



if __name__ == "__main__":
    main()


