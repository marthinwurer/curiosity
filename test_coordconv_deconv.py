import unittest

import torch
from torch.nn import functional as F

from single_dot_dataset import CoordConvTranspose2d
from utilities import nan_canary


class TestDeconv(unittest.TestCase):
    def test_deconv_shape(self):
        data = torch.zeros((1, 3, 8, 8)).cuda()
        mod = CoordConvTranspose2d(3, 50, 3).cuda()
        output = mod(data)
        print(output.shape)
        self.assertEqual(torch.Size([1, 50, 10, 10]), output.shape)
        nan_canary(output)

    def test_deconv_nan(self):
        data = torch.ones((1, 3, 8, 8)).cuda()
        mod = CoordConvTranspose2d(3, 50, 3, stride=2).cuda()
        output = F.relu(mod(data))
        print(output.shape)
        nan_canary(output)


