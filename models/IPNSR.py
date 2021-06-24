import torch
import math
from kornia.filters import gaussian_blur2d


class IPNSR(torch.nn.Module):
    def __init__(self, n_channels, n_feats, n_layers, n_iters, factor=4):
        super(IPNSR, self).__init__()
        self.T = n_iters
        self.gen = GenPyramid(n_levels=n_layers)
        self.pyramid = IPNet(n_channels=n_channels, n_feats=n_feats, n_layers=n_layers)

    def forward(self, ms, lms):
        y = lms
        for _ in range(self.T):
            y = self.gen(y)
            y = self.pyramid(y) + lms
        return y

class GenPyramid(torch.nn.Module):
    def __init__(self, n_levels=4):
        super(GenPyramid, self).__init__()
        self.levels = n_levels
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        gau = []
        y = x
        # gaussian
        for i in range(self.levels):
            gau.append(y)
            y = gaussian_blur2d(y, kernel_size=(3, 3), sigma=(2, 2))
            y = self.pool(y)
        return gau


class ResBlock(torch.nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, bn=False, act=torch.nn.ReLU(True), res_scale=1.0):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(torch.nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias))
            if bn:
                m.append(torch.nn.BatchNorm2d(n_feats))
            if i==0:
                m.append(act)
        self.body = torch.nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class SpectralModule(torch.nn.Module):
    def __init__(self, n_channels, n_feats):
        super(SpectralModule, self).__init__()
        self.in_conv = torch.nn.Conv2d(in_channels=n_channels, out_channels=n_feats*4, kernel_size=1)
        self.out_conv = torch.nn.Conv2d(in_channels=n_feats*4, out_channels=n_channels, kernel_size=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv1 = torch.nn.Conv2d(in_channels=n_feats*4, out_channels=n_feats*4, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels=n_feats*4, out_channels=n_feats*4, kernel_size=1)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        tmp = self.in_conv(x)
        y = self.pool(tmp)
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y)) * tmp
        y = self.out_conv(y)
        return y


class IPNet(torch.nn.Module):
    def __init__(self, n_channels, n_feats, n_layers):
        super(IPNet, self).__init__()
        self.N = n_layers
        # construct pyramids
        up = []
        down = []
        for i in range(n_layers):
            if i<2:
                up.append(SpatialModule(n_channels, n_feats, n_blocks=3, res_scale=1))
            else:
                up.append(SpectralModule(n_channels, n_feats))
        for _ in range(n_layers):
            down.append(FuseBlock(n_feats=n_feats))
        self.up = torch.nn.ModuleList(up)
        self.down = torch.nn.ModuleList(down)

    def forward(self, x):
        upstair = []
        for i in range(self.N):
            y = self.up[i](x[i])
            upstair.append(y)
        for i in range(self.N-1):
            index = self.N - 2 - i
            y = self.down[i](y, upstair[index])
        return y


class FuseBlock(torch.nn.Module):
    def __init__(self, n_feats):
        super(FuseBlock, self).__init__()
        self.deconv = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        y = self.deconv(x1) + x2
        return y


class SpatialModule(torch.nn.Module):
    def __init__(self, n_channels, n_feats, n_blocks, res_scale):
        super(SpatialModule, self).__init__()
        self.in_conv = torch.nn.Conv2d(in_channels=n_channels, out_channels=n_feats, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU(True)
        net = []
        for i in range(n_blocks):
            net.append(ResBlock(n_feats=n_feats, kernel_size=3, res_scale=res_scale))
        self.net = torch.nn.Sequential(*net)
        self.out_conv = torch.nn.Conv2d(in_channels=n_feats, out_channels=n_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.relu(self.in_conv(x))
        y = self.net(y)
        y = self.out_conv(y)
        return y


if __name__ == '__main__':
    net = IPNSR(n_channels=128, n_layers=4, n_feats=256, n_iters=5)
    print(net)
    x = torch.randn(1,128,16,16)
    lms = torch.randn(1, 128, 64, 64)
    y = net(x, lms)
    print(y.shape)
