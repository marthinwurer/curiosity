import math
import unittest

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from image_dataset import ImageListDataset
from single_dot_dataset import CoordConvTranspose2d, WMAutoencoder, train_batch
from utilities import nan_canary


class TestDeconv(unittest.TestCase):
    def test_deconv_shape(self):
        data = torch.zeros((1, 3, 8, 8))#.cuda()
        mod = CoordConvTranspose2d(3, 50, 3)#.cuda()
        output = mod(data)
        print(output.shape)
        self.assertEqual(torch.Size([1, 50, 10, 10]), output.shape)
        nan_canary(output)

    def test_deconv_nan(self):
        data = torch.ones((256, 3, 8, 8))#.cuda()
        mod = CoordConvTranspose2d(3, 50, 3, stride=2)#.cuda()
        output = F.relu(mod(data))
        print(output.shape)
        nan_canary(output)

    def test_deconv_1x1(self):
        data = torch.ones((4, 3, 1, 1))#.cuda()
        mod = CoordConvTranspose2d(3, 50, 3, stride=2)#.cuda()
        output = F.relu(mod(data))
        print(output.shape)
        nan_canary(output)

class TestWMAE(unittest.TestCase):
    def test_forward(self):
        batch = torch.rand((32, 3, 64, 64))#.cuda()

        model = WMAutoencoder()#.cuda()

        output, latent = model(batch)

    def test_forward_with_real_data(self):
        mean = (0.5, 0.5, 0.5)
        stddev = (0.5, 0.5, 0.5)
        in_dim = 64
        batch_size = 32

        net_transform = transforms.Compose([
            transforms.Resize(in_dim),
            transforms.RandomCrop(in_dim, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize(mean, stddev)
        ])

        dataset = ImageListDataset("local_imagenet_items.txt", transform=net_transform)

        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=20
        )

        train_batches = math.ceil(len(dataset) / batch_size)

        model = WMAutoencoder().cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        with tqdm(enumerate(trainloader, 0), total=train_batches, unit="batch") as t:
            for i, data in t:
                # get the inputs
                loss = train_batch(data, model, optimizer)

