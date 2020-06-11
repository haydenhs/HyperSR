import torch
import math


class SRFPN(torch.nn.Module):
    def __init__(self, n_channels, n_feats, n_layers):
        super(SRFPN, self).__init__()
        self.C = n_channels
        self.N = n_layers
        self.B = 3
        self.in_conv = torch.nn.Conv2d(in_channels=n_channels, out_channels=n_feats*4, kernel_size=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.py = FPNBlock(n_feats=n_feats*4, n_layers=n_layers, use_tail=True)
        # self.t = torch.nn.Conv2d(in_channels=n_feats, out_channels=n_feats*4, kernel_size=1)
        # self.s = torch.nn.Conv2d(in_channels=n_feats*4, out_channels=n_feats, kernel_size=1)
        self.net = EDSR(n_feats=n_feats, n_blocks=12, res_scale=0.1)
        # self.upsample = Upsampler(scale=2, n_feats=n_feats)
        self.out_conv = torch.nn.Conv2d(in_channels=n_feats, out_channels=n_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.in_conv(x)
        # y = self.t(y)
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
        for j in range(n_layers-1):
            down.append(FuseBlock(n_feats=n_feats))
        self.up = torch.nn.Sequential(*up)
        self.down = torch.nn.Sequential(*down)
        self.smooth = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.ret = torch.nn.Conv2d(in_channels=n_feats, out_channels=n_feats//4, kernel_size=1)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        upstair = []
        for i in range(self.N):
            x = self.up[i](x)
            upstair.append(x)
            x = self.smooth(x)
        # y starts from the last output in upstream
        y = self.ret(upstair[self.N-1])
        if self.use_tail:
            down_cases = self.N - 1
        else:
            down_cases = self.N - 2
        for i in range(down_cases):
            if self.use_tail:
                index = down_cases - i - 1
            else:
                index = down_cases - i
            y = self.down[i](y, upstair[index])
        return y


class FuseBlock(torch.nn.Module):
    def __init__(self, n_feats):
        super(FuseBlock, self).__init__()
        self.deconv = Upsampler(scale=2, n_feats=n_feats//4)
        # self.pool = Downsampler(scale=2, n_feats=n_feats)
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


class Downsampler(torch.nn.Sequential):
    def __init__(self, scale, n_feats):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(torch.nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, stride=2))
                m.append(torch.nn.LeakyReLU(0.02, inplace=True))
        elif scale == 3:
            m.append(torch.nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, stride=3))
            m.append(torch.nn.LeakyReLU(0.02, inplace=True))
        else:
            raise NotImplementedError
        super(Downsampler, self).__init__(*m)
