import torch
from torch import nn
from torchvision.transforms.functional import center_crop
from activation import PySiLU


class DoubleConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, activation: str = 'relu') -> None:
        """
        Initialize the DoubleConv module.

        Args:
            in_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
        """
        super().__init__()
        act_map = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'pysilu': PySiLU
        }
        assert activation in act_map
        act = act_map[activation]

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            act(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            act()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DoubleConv module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the convolutional layers.
        """
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channel, features=[64, 128, 256, 512], activation: str = 'relu'):
        super().__init__()
        self.down_blocks = nn.ModuleList()
        for feature in features:
            self.down_blocks.append(DoubleConv(in_channel, feature, activation))
            in_channel = feature
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_list = []
        for block in self.down_blocks:
            x = block(x)
            x = self.pool(x)
            x_list.append(x)
        return x, x_list


class UpBlock(nn.Module):
    def __init__(self, out_channel=1, features=[1024, 512, 256, 128], activation: str = 'relu') -> None:
        super().__init__()
        self.up_smaples = nn.ModuleList()
        self.convs = nn.ModuleList()
        for feature in features:
            self.up_smaples.append(
                nn.ConvTranspose2d(feature, feature // 2, kernel_size=2, stride=2)
            )
            self.convs.append(
                DoubleConv(feature, feature // 2, activation)
            )
        self.final_conv = nn.Conv2d(features[-1] // 2, out_channel, kernel_size=1, stride=1)

    def forward(self, x, x_list):
        for i in range(len(self.up_smaples)):
            x = self.up_smaples[i](x)
            skip_x = center_crop(x_list[i], x.shape[2:])
            x = self.convs[i](torch.cat([skip_x, x], dim=1))

        return self.final_conv(x)
    

class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, features=[64, 128, 256, 512], activation: str = 'relu') -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.features = features
        up_features = []
        for feature in features[::-1]:
            up_features.append(feature * 2)

        self.down_block = DownBlock(in_channel, features, activation)
        self.mid_block = DoubleConv(features[-1], features[-1] * 2, activation)
        self.up_block = UpBlock(out_channel, up_features, activation)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x, x_list = self.down_block(x)
        x = self.mid_block(x)
        x = self.up_block(x, x_list[::-1])
        return self.act(x)
        

if __name__ == "__main__":
    x = torch.rand(3, 3, 160, 240)
    model = UNet(3, 1, activation='silu')
    y = model(x)
    print(y.shape)