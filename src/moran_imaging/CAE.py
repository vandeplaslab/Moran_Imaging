import math
import torch.nn as nn
import torch.nn.functional as functional


def conv2d_hout(height, padding, dilation, kernel_size, stride):
    tmp = math.floor(((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
    return 1 if tmp < 1 else tmp


def conv2d_wout(width, padding, dilation, kernel_size, stride):
    tmp = math.floor(((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
    return 1 if tmp < 1 else tmp


def conv2d_hwout(height, width, padding, dilation, kernel_size, stride):
    h = math.floor(((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
    h = 1 if h < 1 else h
    w = math.floor(((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
    w = 1 if w < 1 else w

    return h, w


class CAE(nn.Module):
    def __init__(
        self,
        height,
        width,
        train_mode=True,
    ):
        super(CAE, self).__init__()
        self.train_mode = train_mode
        # self.fc_h1, self.fc_h2 = 768, 256
        self.encoder_dim = 7
        self.k1, self.k2 = (3, 3), (3, 3)
        self.s1, self.s2 = (2, 2), (3, 3)
        self.height = height
        self.width = width
        self.d1, self.d2 = 8, 16

        # encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.d1, kernel_size=self.k1, stride=self.s1, padding=(0, 0), dilation=(1, 1), bias=False),
            nn.BatchNorm2d(self.d1, momentum=0.01),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.d1, self.d2, kernel_size=self.k2, stride=self.s2, padding=(0, 0), bias=False),
            nn.BatchNorm2d(self.d2, momentum=0.01),
            nn.ReLU(),
        )

        # Calculating dimensions of encoding
        self.l1height = conv2d_hout(height=height, padding=(0, 0), dilation=(1, 1), kernel_size=self.k1, stride=self.s1)
        self.l1width = conv2d_wout(width=width, padding=(0, 0), dilation=(1, 1), kernel_size=self.k1, stride=self.s1)
        self.l2height = conv2d_hout(
            height=self.l1height, padding=(0, 0), dilation=(1, 1), kernel_size=self.k2, stride=self.s2
        )
        self.l2width = conv2d_wout(
            width=self.l1width, padding=(0, 0), dilation=(1, 1), kernel_size=self.k2, stride=self.s2
        )

        self.encoder = nn.Linear(self.l2height * self.l2width * self.d2, self.encoder_dim)

        # decoder
        self.fc3 = nn.Linear(self.encoder_dim, self.l2height * self.l2width * self.d2)
        self.ct1 = nn.ConvTranspose2d(
            in_channels=self.d2,
            out_channels=self.d1,
            kernel_size=self.k2,
            stride=self.s2,
            padding=(0, 0),
            dilation=(1, 1),
            output_padding=(1, 1),
        )
        self.tbn1 = nn.BatchNorm2d(self.d1, momentum=0.01)
        self.trelu1 = nn.ReLU()
        self.ct2 = nn.ConvTranspose2d(
            in_channels=self.d1,
            out_channels=1,
            kernel_size=self.k1,
            stride=self.s1,
            padding=(0, 0),
            dilation=(1, 1),
            output_padding=(1, 1),
        )
        self.tbn2 = nn.BatchNorm2d(1, momentum=0.01)
        self.trelu2 = nn.ReLU()

    def convtrans1(self, x, output_size):
        x = self.ct1(x, output_size=output_size)
        x = self.tbn1(x)
        x = self.trelu1(x)
        return x

    def convtrans2(self, x, output_size):
        x = self.ct2(x, output_size=output_size)
        x = self.tbn2(x)
        x = self.trelu2(x)
        return x

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        return x

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(-1, 16, self.l2height, self.l2width)
        z = self.convtrans1(z, output_size=(self.l1height, self.l1width))
        z = self.convtrans2(z, output_size=(self.height, self.width))
        z = functional.interpolate(z, size=(self.height, self.width), mode="bilinear")
        return z

    def forward(self, x):
        # mu, sigma = self.encode(x)
        z = self.encode(x)
        xp = self.decode(z)
        return xp
