import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.pool0 = nn.MaxPool2d(2, stride=2)

        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )

        # decoder (upsampling)
        self.upconv3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )

        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.upconv0 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(self.pool0(e0))
        e2 = self.enc_conv2(self.pool1(e1))
        e3 = self.enc_conv3(self.pool2(e2))

        # bottleneck
        b = self.bottleneck_conv(self.pool3(e3))

        # decoder
        d3 = self.dec_conv3(torch.cat([self.upconv3(b), e3], 1))
        d2 = self.dec_conv2(torch.cat([self.upconv2(d3), e2], 1))
        d1 = self.dec_conv1(torch.cat([self.upconv1(d2), e1], 1))
        d0 = self.dec_conv0(torch.cat([self.upconv0(d1), e0], 1))

        return torch.sigmoid(self.final_conv(d0))


class UnetBlock(nn.Module):
    """
    UNet block
    It can be used to sequrntially build a larger UNet from the bottom up.
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels=None,
        layers=1,
        sub_network=None,
        filter_size=3,
        unet_block="cnn",
    ):
        super().__init__()

        # Define which type the encoder/decoder block come from

        block = self.cnn_layer
        if unet_block.lower().strip() == "resnet":
            block = self.resnet_layer

        # Encoder layers
        in_layers = [block(in_channels, mid_channels, filter_size)]

        # Set the multiplier for the concatenation cnn's of the decoder
        if sub_network is None:
            inputs_to_outputs = 1
        else:
            inputs_to_outputs = 2

        # Decoder layers
        out_layers = [
            block(mid_channels * inputs_to_outputs, mid_channels, filter_size)
        ]

        # Sequentially build up the encoder and decoder networks
        for _ in range(layers - 1):
            in_layers.append(block(mid_channels, mid_channels, filter_size))
            out_layers.append(block(mid_channels, mid_channels, filter_size))

        # Convolution to preserve size of image
        if out_channels is not None:
            out_layers.append(nn.Conv2d(mid_channels, out_channels, 1, padding=0))

        # Unpack the encoder layers in a Sequential module for forward
        self.in_model = nn.Sequential(*in_layers)

        # Create a bottleneck layer ( from the subnetworks (if they exist) or a simple conv2d that preserves size and channels
        if sub_network is not None:
            self.bottleneck = nn.Sequential(
                # Downscale
                nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    filter_size,
                    padding=filter_size // 2,
                    stride=2,
                ),
                sub_network,
                # Upscale
                nn.ConvTranspose2d(
                    mid_channels,
                    mid_channels,
                    filter_size,
                    padding=filter_size // 2,
                    output_padding=1,
                    stride=2,
                ),
            )
        else:
            self.bottleneck = None

        self.out_model = nn.Sequential(*out_layers)

    def forward(self, x):
        full_scale_result = self.in_model(x)

        if self.bottleneck is not None:
            bottle_result = self.bottleneck(full_scale_result)
            full_scale_result = torch.cat([full_scale_result, bottle_result], dim=1)

        return self.out_model(full_scale_result)

    def cnn_layer(self, in_channels, out_channels, kernel_size=3, bn=True):
        padding = kernel_size // 2  # To preserve img dimensions. Equal to int((k-1)/2)
        cnn_bias = False if bn else True  # Fewer parameters to save
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, bias=cnn_bias
            ),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.LeakyReLU(),
        )

    def resnet_layer(self, in_channels, out_channels, kernel_size=3, bn=True):
        padding = kernel_size // 2  # To preserve img dimensions. Equal to int((k-1)/2)
        bias = False if bn else True  # Fewer parameters to save
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, bias=bias
            )
        )
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU())
        layers.append(
            nn.Conv2d(
                out_channels, out_channels, kernel_size, padding=padding, bias=bias
            )
        )
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))

        block = nn.Sequential(*layers)

        if in_channels == out_channels:
            identity = nn.Identity()
        else:
            identity = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        return ResidualBlock(block, identity)


class ResidualBlock(nn.Module):
    def __init__(self, block, shortcut):
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x):
        return F.leaky_relu(self.block(x) + self.shortcut(x))


class UNetBlocked(nn.Module):
    """
    Creates a UNet from UnetBlock blocks
    """

    def __init__(self, in_channels, out_channels, unet_block="cnn"):
        """
        in_channels: input image channels, usually 3 for rgb or 1 for grayscale
        out_channels: 1 for 1 class segmentation (0,1) or n for n classes
        """
        super().__init__()

        layers_per_building_block = 2

        # Create UNet from UNetBlock 's based on the constructor arguments
        self.unet_model = nn.Sequential(
            UnetBlock(
                in_channels,
                32,
                layers=layers_per_building_block,
                unet_block=unet_block,
                sub_network=UnetBlock(
                    32,
                    64,
                    out_channels=32,
                    layers=layers_per_building_block,
                    unet_block=unet_block,
                    sub_network=UnetBlock(
                        64,
                        128,
                        out_channels=64,
                        layers=layers_per_building_block,
                        unet_block=unet_block,
                        sub_network=UnetBlock(
                            128,
                            256,
                            out_channels=128,
                            layers=layers_per_building_block,
                            unet_block=unet_block,
                            sub_network=None,
                        ),
                    ),
                ),
            ),
            nn.Conv2d(32, out_channels, 3, padding=1),
        )

    def forward(self, x):
        x = self.unet_model(x)
        return x