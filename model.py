import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        self.cbam = CBAM(dim_out, 16)

    # 수정
    def forward(self, x):
        return x + self.cbam(self.main(x))
        # return self.cbam(x + self.main(x))


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=10, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class ResidualBlockDis(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01)
        self.cbam = CBAM(dim_out, 16)

    def forward(self, x):
        return x + self.cbam(self.main(x))


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=[200, 300], conv_dim=64, c_dim=10, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        # 컬러로 할 거면 바로 밑에 nn.Conv2d(1, ...) => nn.Conv2d(3, ...)
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU
        0.01))
        ## 요기두 줘볼까요
        curr_dim = conv_dim
        for i in range(1, repeat_num):
        # 컬러로 할 거면 바로 밑에 nn.Conv2d(curr_dim, ...) => nn.Conv2d(3*curr_dim, ...)
        # layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=2, padding=1))
        # layers.append(nn.LeakyReLU(0.01))
        # layers.append(CBAM(curr_dim*2, 16))
            layers.append(ResidualBlockDis(dim_in=curr_dim, dim_out=curr_dim))
        curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # 여기 kernel_size=(4,5)는 인풋이 (200, 300)라 출력의 크기 맞추기 위함
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=(4, 5), bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)

        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))