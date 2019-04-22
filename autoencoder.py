import glob
import math
import random

import matplotlib.pyplot as plt
import numpy
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, random_split
import torch.nn.functional as F


# Ignore warnings
import warnings

from torchvision.transforms import ToTensor, transforms
from tqdm import tqdm

from utilities import conv2d_factory, unflatten, flat_shape, flatten

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
        # print(filename)
        image = Image.open(filename).convert("RGB")

        sample = self.transform(image)

        return (sample, 0)  # return tuple for reasons


class ProGANAutoencoder(nn.Module):
    def __init__(self, latent_size, input_power, output_power, start_filters=16, max_filters=512, activation=nn.ReLU):
        super().__init__()
        self.latent_size = latent_size

        self.encoder = ProGANEncoder(input_power, output_power, start_filters, max_filters, activation)

        self.fc_enc = nn.Linear(flat_shape(self.encoder.output_shape), latent_size)

        self.decoder = ProGANDecoder(latent_size, output_power, input_power, start_filters, max_filters, activation)

    def forward(self, input):
        x = self.encoder(input)
        x = flatten(x)
        latent = self.fc_enc(x)
        output = self.decoder(latent)

        return output, latent

    def loss(self, input, output, latent):
        reconstruction = F.mse_loss(input, output)

        return reconstruction


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

        assert input_power > output_power

        layers = nn.ModuleList([])

        num_layers = input_power - output_power
        in_channels = start_filters
        out_channels = in_channels

        from_rgb = nn.Conv2d(3, start_filters, 1, 1)
        layers.append(from_rgb)
        layers.append(activation())

        for i in range(num_layers):
            out_channels = min(in_channels * 2, max_filters)
            layer = ProGANEncoderLayer(2, in_channels, out_channels, activation)
            in_channels = out_channels
            layers.append(layer)

        final = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        layers.append(final)

        self.layers = layers
        self.output_shape = (out_channels, 2**output_power, 2**output_power)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ProGANEncoderLayer(nn.Module):
    def __init__(self, num_layers: int, in_channels, out_channels, activation):
        super().__init__()
        num_starter = num_layers - 1
        if num_starter < 0:
            raise ValueError("Must have at least one layer")
        starter_layers = nn.ModuleList([])
        for i in range(num_starter):
            starter = nn.Conv2d(in_channels, in_channels, 3, 1, padding=1)  # need padding to not shrink size
            starter_layers.append(starter)
            starter_layers.append(activation())
        grow = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        pool = nn.AvgPool2d(2, 2)
        self.starters = starter_layers
        self.layer = nn.Sequential(
            grow,
            activation(),
            pool
        )

    def forward(self, x):
        for layer in self.starters:
            x = layer(x)
        x = self.layer(x)
        return x


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

        assert output_power > input_power

        reverse_layers = []

        num_layers = output_power - input_power
        in_channels = end_filters
        out_channels = in_channels

        to_rgb = nn.Conv2d(out_channels, 3, 1, 1)

        for i in range(num_layers):
            in_channels = min(out_channels * 2, max_filters)
            layer = ProGANDecoderLayer(2, in_channels, out_channels, activation)
            out_channels = in_channels
            reverse_layers.append(layer)

        ordered_layers = reversed(reverse_layers)

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


class ProGANDecoderLayer(nn.Module):
    def __init__(self, num_layers: int, in_channels, out_channels, activation):
        super().__init__()
        num_same = num_layers - 1
        if num_same < 0:
            raise ValueError("Must have at least one layer")

        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        shrink = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)

        self.layer = nn.Sequential(
            upsample,
            shrink,
            activation(),
        )

        same_layers = nn.ModuleList([])
        for i in range(num_same):
            starter = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)  # need padding to not shrink size
            same_layers.append(starter)
            same_layers.append(activation())

        self.same = same_layers

    def forward(self, x):
        x = self.layer(x)
        for layer in self.same:
            x = layer(x)
        return x


def train_autoencoder(net, optimizer, device, trainset, trainloader, batch_size, epochs, callback=None):
    train_batches = math.ceil(len(trainset) / batch_size)
    running_loss = 0.0

    for epoch in range(epochs):  # loop over the dataset multiple times

        # loss_steps = 8000 / batch_size
        loss_steps = 5

        with tqdm(enumerate(trainloader, 0), total=train_batches, unit="batch") as t:
            for i, data in t:
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, latents = net(inputs)
                loss = net.loss(inputs, outputs, latents)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % loss_steps == loss_steps - 1:  # print every 2000 mini-batches
                    string = '[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, running_loss / loss_steps)
                    t.set_postfix_str(string)
                    running_loss = 0.0
            if callback:
                callback()


def view_dataset():

    view_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256, pad_if_needed=True),
    ])

    face_dataset = VGImagesDataset(root_dir=VG_PATH, transform=view_transform)

    fig = plt.figure()

    index = random.randrange(len(face_dataset))
    print(index)

    image = face_dataset[index]

    print(image.size())

    plt.imshow(image)
    plt.show()


def main():
    # plt.ion()  # interactive mode

    # BATCH_SIZE = 128
    # BATCH_SIZE = 32
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    EPOCHS = 1
    MOMENTUM = 0.9
    IN_POWER = 3

    in_dim = 2 ** IN_POWER



    net_transform = transforms.Compose([
        transforms.Resize(in_dim),
        transforms.RandomCrop(in_dim, pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = VGImagesDataset(root_dir=VG_PATH, transform=net_transform)

    net = ProGANAutoencoder(512, IN_POWER, 2)
    print(net)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)

    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    train_autoencoder(net, optimizer, device, dataset, trainloader, BATCH_SIZE, EPOCHS)

    path = "saved_nets/autoencoder.mod"
    print("Saving Model to %s" % path)
    torch.save(net.state_dict(), path)




if __name__ == "__main__":
    main()


