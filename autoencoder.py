import glob
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

from utilites import conv2d_factory

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


class ProGANEncoder(nn.Module):
    def __init__(self, input_shape, layer_factory, max_final_width=4, activation=F.relu):
        """

        Args:
            input_shape: a 3-element tuple with the number of channels, the height, and the width. (channels first)
            max_final_width: the maximum desired final width of the network.
            activation: The activation function used by each layer
        """
        super().__init__()

        next_shape = input_shape
        self.activation = activation
        self.output_shape = next_shape

        # compute input shapes
        self.shapes = [] # shape coming from the previous layer

        while min(next_shape[1:]) > max_final_width:
            self.shapes.append(next_shape)

        self.conv_layers = nn.ModuleList([])
        # add filters with stride until our output size is less than 6
        while min(next_shape[1:]) >= max_final_width:
            conv, next_shape, layer_params = conv2d_factory(next_shape, stride=2,
                                                            padding=1, filter_multiplier=2)
            self.conv_layers.append(conv)
            self.output_shape = next_shape

    def forward(self, x):
        for layer in self.conv_layers:
            x = self.activation(layer(x))
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


