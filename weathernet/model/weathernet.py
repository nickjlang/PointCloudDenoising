import torch
import torch.hub as hub
import torch.nn as nn
import torch.nn.functional as F

pretrained_models = {
    'kitti': {
        'url': 'https://github.com/TheCodez/pytorch-WeatherNet/releases/download/0.1/WeatherNet_45.5-75c06618.pth',
        'num_classes': 4
    }
}


def WeatherNet(pretrained=None, num_classes=13):
    """Constructs a WeatherNet model.

    Args:
        pretrained (string): If not ``None``, returns a pre-trained model. Possible values: ``kitti``.
        num_classes (int): number of output classes. Automatically set to the correct number of classes
            if ``pretrained`` is specified.
    """
    if pretrained is not None:
        model = WeatherNet(pretrained_models[pretrained]['num_classes'])
        model.load_state_dict(hub.load_state_dict_from_url(pretrained_models[pretrained]['url']))
        return model

    model = WeatherNet(num_classes)
    return model


class WeatherNet(nn.Module):
    """
    Implements WeatherNet model from
    `"CNN-based Lidar Point Cloud De-Noising in Adverse Weather"
    <https://arxiv.org/pdf/1912.03874.pdf>`_.

    Arguments:
        num_classes (int): number of output classes
    """

    def __init__(self, num_classes=4):
        super(WeatherNet, self).__init__()

        self.lila1 = LiLaBlock(2, 32)
        self.lila2 = LiLaBlock(32, 64)
        self.lila3 = LiLaBlock(64, 96)
        self.lila4 = LiLaBlock(96, 96)
        self.drop_layer = nn.Dropout()
        self.lila5 = LiLaBlock(96, 64)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, distance, reflectivity):
        # print("distance: '{}'".format(distance.shape))
        # print("reflectivity: '{}'".format(reflectivity.shape))
        x = torch.cat([distance, reflectivity], 1)
        x = self.lila1(x)
        x = self.lila2(x)
        x = self.lila3(x)
        x = self.lila4(x)
        x = self.drop_layer(x)
        x = self.lila5(x)

        x = self.classifier(x)

        return x


class LiLaBlock(nn.Module):

    def __init__(self, in_channels, n):
        super(LiLaBlock, self).__init__()

        self.branch1 = BasicConv2d(in_channels, n, kernel_size=(7, 3), padding=(2, 0), stride=(1,1))
        self.branch2 = BasicConv2d(in_channels, n, kernel_size=3, stride=(1,1))
        self.branch3 = BasicConv2d(in_channels, n, kernel_size=3, dilation=(2,2), padding=1, stride=(1,1))
        self.branch4 = BasicConv2d(in_channels, n, kernel_size=(3, 7), padding=(0, 2), stride=(1,1))
        self.conv = BasicConv2d(n * 4, n, kernel_size=1, padding=1, stride=(1,1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = torch.cat([branch1, branch2, branch3, branch4], 1)
        output = self.conv(output)

        return output


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__ == '__main__':
    num_classes, height, width = 4, 64, 512

    model = WeatherNet(num_classes)  # .to('cuda')
    inp = torch.randn(5, 1, height, width)  # .to('cuda')

    out = model(inp, inp)
    assert out.size() == torch.Size([5, num_classes, height, width])

    print('Pass size check.')
