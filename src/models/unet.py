import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp


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




## Encoder-Decoder 
# Encoder(Atrous Cnvoilutional Networks, ResNet-101) + Decoder(Atrous Spatial Pyramid Pooling)
class ASPP0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)  # (out_channels * 5) because we concatenate five different paths
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))
        x2 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x)))
        x3 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x)))
        x4 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x)))

        x5 = self.avg_pool(x)
        x5 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(x5)))
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(x)))

        return x
## not working    
#class ASPP2(nn.Module):
#    def __init__(self, in_channels, out_channels, rates=[6, 12, 18, 24]):
#        super(ASPP, self).__init__()
#        self.convs = nn.ModuleList()
#        self.bns = nn.ModuleList()
#
#        # 1x1 convolution
#        self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
#        self.bns.append(nn.BatchNorm2d(out_channels))
#
#        # Convolutions at different dilation rates
#        for rate in rates:
#            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate))
#            self.bns.append(nn.BatchNorm2d(out_channels))
#
#        # Image-level features
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
#        self.bns.append(nn.BatchNorm2d(out_channels))
#
#        # Calculate the correct number of input channels for the final 1x1 convolution
#        self.final_in_channels = out_channels * (len(rates) + 2)
#
#        # Final 1x1 convolution
#        self.conv_1x1_output = nn.Conv2d(self.final_in_channels, out_channels, kernel_size=1)
#        self.bn_output = nn.BatchNorm2d(out_channels)
#
#    def forward(self, x):
#        res = []
#
#        # Apply all convolutions and batch norms
#        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
#            conv_result = conv(x)
#            bn_result = bn(conv_result)
#            res.append(F.relu(bn_result))
#            print(f"Conv {i}: {conv_result.shape}")
#
#        # Global average pooling
#        image_features = self.avg_pool(x)
#        image_features = F.relu(self.bns[-1](self.convs[-1](image_features)))
#        image_features = F.interpolate(image_features, size=x.size()[2:], mode='bilinear', align_corners=False)
#        res.append(image_features)
#        print(f"Image features: {image_features.shape}")
#        print(f"Number of feature maps before concatenation: {len(res)}")  # Check the number of feature maps
#        
#        # Concatenate along the channel dimension
#        x = torch.cat(res, dim=1)
#        print(f"After concatenation: {x.shape}")
#
#        # Final 1x1 conv
#        x = self.conv_1x1_output(x)
#        x = self.bn_output(x)
#        x = F.relu(x)
#
#        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        # Existing dilated convolutions
        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)
        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)
        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

        # Additional dilated convolutions
        self.conv_3x3_4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=24, dilation=24)
        self.bn_conv_3x3_4 = nn.BatchNorm2d(out_channels)
        self.conv_3x3_5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=30, dilation=30)
        self.bn_conv_3x3_5 = nn.BatchNorm2d(out_channels)

        # Image-level features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

        # Final 1x1 convolution
        self.conv_1x1_3 = nn.Conv2d(out_channels * 7, out_channels, kernel_size=1)  # Adjust the multiplier according to the number of concatenated feature maps
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)                         

    def forward(self, x):
        x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))
        x2 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x)))
        x3 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x)))
        x4 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x)))
        x5 = F.relu(self.bn_conv_3x3_4(self.conv_3x3_4(x)))
        x6 = F.relu(self.bn_conv_3x3_5(self.conv_3x3_5(x)))

        x7 = self.avg_pool(x)
        x7 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(x7)))
        x7 = F.interpolate(x7, size=x4.size()[2:], mode='bilinear', align_corners=False)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7), dim=1)

        x = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(x)))

        return x




class DeepLabv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepLabv3, self).__init__()
        # Use a pre-trained ResNet model as the backbone for feature extraction
        self.backbone = models.resnet34(pretrained=False) # resnet50
        self.backbone_layers = list(self.backbone.children())[:-2]  # Remove the last two layers (average pooling and fully connected layers)
        self.backbone = nn.Sequential(*self.backbone_layers)

        # Replace the first convolution layer if the input channels are not equal to 3 (RGB)
        if in_channels != 3:
            self.backbone[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # ASPP module
        if self.backbone == models.resnet50(pretrained=False) or self.backbone == models.resnet101(pretrained=False):
            self.aspp = ASPP(2048, 256)  # 2048 is the number of channels in the output of ResNet-101, Resnet-50
        else:
            self.aspp = ASPP(512, 256) # For resnet34
        
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)  # Replace input_image_height and input_image_width with the desired output size
        return x
    
    
def deeplabv3_smp():
    model = smp.DeepLabV3(
        encoder_name='resnet34', encoder_depth=5,
        encoder_weights=None, encoder_output_stride=16, 
        decoder_channels=256, decoder_atrous_rates=(12, 24, 36),
        in_channels=1, classes=3, activation=None, upsampling=4, aux_params=None)
    return model

def deeplabv3plus_smp():
    model = smp.DeepLabV3Plus(
        encoder_name='resnet34', encoder_depth=5,
        encoder_weights=None, encoder_output_stride=16, 
        decoder_channels=256, decoder_atrous_rates=(12, 24, 36),
        in_channels=1, classes=3, activation=None, upsampling=4, aux_params=None)
    return model