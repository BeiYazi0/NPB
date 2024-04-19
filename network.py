from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    expansion = 1

    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, 3, padding=1, stride=strides, bias=False)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, 1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(num_channels)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.bn3(self.conv3(X))
        Y += X
        return F.relu(Y)


def ResNet(block=Residual, num_block=[3, 3, 3], nChannels=[16, 16, 32, 64], num_classes=10):
    def _resnet_block(input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for num in range(num_residuals):
            if num == 0 and not first_block:
                blk.append(block(input_channels, num_channels,
                                 use_1x1conv=True, strides=2))
            else:
                blk.append(block(num_channels, num_channels, strides=1))
        return blk

    # 1st conv before any network block
    conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
    conv = []
    for i in range(1, len(nChannels)):
        conv.extend(_resnet_block(nChannels[i - 1], nChannels[i], num_block[i - 1], i == 1))
    net = nn.Sequential(
        conv1, nn.BatchNorm2d(nChannels[0]), nn.ReLU(),
        *conv,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(), nn.Linear(nChannels[-1] * block.expansion, num_classes))
    net.__class__.__name__ = 'ResNet'
    return net


def ResNet20():
    return ResNet(Residual, [3, 3, 3], [16, 16, 32, 64], 10)


def ResNet18():
    return ResNet(Residual, [2, 2, 2, 2], [64, 64, 128, 256, 512], 200)


def Vgg(num_classes=100, conv_arch=((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))):  # vgg19
    def _vgg_block(num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return layers

    conv_blks = []
    in_channels = 3
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.extend(_vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    conv_blks[-1] = nn.AvgPool2d(kernel_size=2, stride=2)
    net = nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(512, num_classes))
    net.__class__.__name__ = 'VGG'
    return net