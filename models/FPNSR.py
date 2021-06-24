import torch
import math


"""Hyperspectral image super-resolution using feature pyramid network"""
class FPNSR(torch.nn.Module):
    def __init__(self, n_channels, n_feats, n_layers):
        super(FPNSR, self).__init__()
        self.in_conv = torch.nn.Conv2d(in_channels=n_channels, out_channels=n_feats*4, kernel_size=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.py = FPNBlock(n_feats=n_feats*4, n_layers=n_layers, use_tail=True)
        self.net = EDSR(n_feats=n_feats, n_blocks=12, res_scale=0.1)
        # self.upsample = Upsampler(scale=2, n_feats=n_feats)
        self.out_conv = torch.nn.Conv2d(in_channels=n_feats, out_channels=n_channels, kernel_size=3, padding=1)

    def forward(self, x, lms):
        y = self.in_conv(lms)
        y = self.py(y)
        y = self.net(y)
        out = self.out_conv(y)
        return out


class ResBlock(torch.nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, bn=False, act=torch.nn.ReLU(True), res_scale=1.0):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(torch.nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias))
            if bn:
                m.append(torch.nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = torch.nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResGroupBlock(torch.nn.Module):
    def __init__(self, n_feats, n_groups, res_scale=1.0):
        super(ResGroupBlock, self).__init__()
        self.N = n_feats
        self.G = n_groups
        self.res_scale = res_scale
        self.conv1 = torch.nn.Conv2d(in_channels=self.N, out_channels=self.N, kernel_size=3, padding=1, groups=n_groups)
        self.conv2 = torch.nn.Conv2d(in_channels=self.N, out_channels=self.N, kernel_size=3, padding=1, groups=n_groups)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y)).mul(self.res_scale)
        y += x
        return y


class FPNBlock(torch.nn.Module):
    def __init__(self, n_feats, n_layers, use_tail=True):
        super(FPNBlock, self).__init__()
        self.N = n_layers
        self.use_tail = use_tail
        # construct pyramids
        up = []
        down = []
        for i in range(n_layers):
            up.append(ResGroupBlock(n_feats=n_feats, n_groups=4, res_scale=0.1))
        for j in range(n_layers):
            down.append(FuseBlock(n_feats=n_feats))
        self.up = torch.nn.Sequential(*up)
        self.down = torch.nn.Sequential(*down)
        self.smooth = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = torch.nn.Conv2d(in_channels=n_feats, out_channels=n_feats//4, kernel_size=1)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        upstair = []
        for i in range(self.N):
            x = self.up[i](x)
            upstair.append(x)
            x = self.smooth(x)
        y = self.conv(x)
        for i in range(self.N):
            index = self.N - 1 - i
            y = self.down[i](y, upstair[index])
        return y


class FuseBlock(torch.nn.Module):
    def __init__(self, n_feats):
        super(FuseBlock, self).__init__()
        self.deconv = Upsampler(scale=2, n_feats=n_feats//4)
        self.conv = torch.nn.Conv2d(in_channels=n_feats, out_channels=n_feats//4, kernel_size=1)

    def forward(self, x1, x2):
        y = self.deconv(x1) + self.conv(x2)
        return y


class EDSR(torch.nn.Module):
    def __init__(self, n_feats, n_blocks, res_scale):
        super(EDSR, self).__init__()
        net = []
        for i in range(n_blocks):
            net.append(ResBlock(n_feats=n_feats, kernel_size=3, res_scale=res_scale))
        self.net = torch.nn.Sequential(*net)
        # self.conv = Upsampler(scale=2, n_feats=n_feats)

    def forward(self, x):
        y = self.net(x)
        return y


class Upsampler(torch.nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(torch.nn.Conv2d(n_feats, 4 * n_feats, kernel_size=3, padding=1, bias=bias))
                m.append(torch.nn.PixelShuffle(2))
                if bn:
                    m.append(torch.nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(torch.nn.ReLU(True))
                elif act == 'prelu':
                    m.append(torch.nn.PReLU(n_feats))

        elif scale == 3:
            m.append(torch.nn.Conv2d(n_feats, 9 * n_feats, kernel_size=3, padding=1, bias=bias))
            m.append(torch.nn.PixelShuffle(3))
            if bn:
                m.append(torch.nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(torch.nn.ReLU(True))
            elif act == 'prelu':
                m.append(torch.nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


if __name__ == '__main__':
    net = FPNSR(n_channels=128, n_feats=256, n_layers=4)
    print(net)
    x = torch.randn(1,128,16,16)
    lms = torch.randn(1,128,64,64)
    y = net(x, lms)
    print(y.shape)
