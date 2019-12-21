import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.conv as conv

from coordconv import CoordConv2d, AddCoords


def generate_image(coord, shape):
    image = np.zeros(shape)
    image[coord[0], coord[1]] = 1
    return image


def generate_single_dot_dataset(height, width):
    coordinates = []
    for h in range(height):
        for w in range(width):
            coordinates.append((h, w))

    images = []
    for coord in coordinates:
        images.append((generate_image(coord, (height, width)), coord))

    return images


def random_split(data, train_split):
    """
    Randomly split the data into a train and test split
    """
    to_shuffle = data.copy
    random.shuffle(to_shuffle)
    train_end = int(len(to_shuffle) * train_split)
    train = to_shuffle[:train_end]
    test = to_shuffle[train_end:]
    return train, test


def find_quadrant(coord, shape):
    center_y = shape[0] // 2
    center_x = shape[0] // 2
    y = coord[0] - center_y
    x = coord[1] - center_x
    if y < 0:
        if x > 0:
            return 1
        else:
            return 2
    else:
        if x > 0:
            return 3
        else:
            return 4


def quadrant_split(data, quadrant, shape):
    """
    Split the data into a train and test split by quadrant
    """
    train = []
    test = []
    for item in data:
        image, coord = item
        if quadrant == find_quadrant(coord, shape):
            test.append(item)
        else:
            train.append(item)

    return train, test


class WMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CoordConv2d(3, 32, 4)
        self.conv2 = CoordConv2d(32, 64, 4)
        self.conv3 = CoordConv2d(64, 128, 4)
        self.conv4 = CoordConv2d(128, 246, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x

class CoordConvTranspose2d(conv.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.ConvTranspose2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input, output_size=None):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input)
        out = self.conv(out)

        return out

class WMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = CoordConvTranspose2d(1024, 128, 5)
        self.deconv2 = CoordConvTranspose2d(128, 64, 5)
        self.deconv3 = CoordConvTranspose2d(64, 32, 6)
        self.deconv4 = CoordConvTranspose2d(32, 3, 6)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.sigmoid(self.deconv4(x))
        return x



