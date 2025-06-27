import torch
import torch.nn as nn
import torch.nn.functional as F

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.1, 
                 kernel_size=5, num_convs=2, padding=None, depthwise_conv=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.kernel_size = kernel_size
        self.depthwise_conv = depthwise_conv
        
        if padding is None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding

        layers = []

        # First layer from in_channels to mid_channels (or directly to out_channels if num_convs=1)
        if depthwise_conv:
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=self.padding, bias=False, groups=in_channels))                # Apply depthwise convolution
            layers.append(nn.Conv2d(in_channels, mid_channels if num_convs > 1 else out_channels, kernel_size=1, bias=False))                           # 1x1 pointwise convolution to combine features across channels
        else:
            layers.append(nn.Conv2d(in_channels, mid_channels if num_convs > 1 else out_channels, kernel_size=kernel_size, padding=self.padding, bias=False))    # Standard convolution
        layers.append(nn.BatchNorm2d(mid_channels if num_convs > 1 else out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout2d(p=dropout_rate))

        # Mid layers from mid_channels to mid_channels (only if num_convs > 2)
        for _ in range(num_convs - 2):
            layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=self.padding, bias=False, groups=(mid_channels if depthwise_conv else 1)))
            if depthwise_conv: 
                layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(mid_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(p=dropout_rate))

        # Last layer from mid_channels to out_channels (only if num_convs > 1)
        if num_convs > 1:
            layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=self.padding, bias=False, groups=(mid_channels if depthwise_conv else 1)))
            if depthwise_conv:
                layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(p=dropout_rate))

        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, num_convs=2, kernel_size=3, padding=None, depthwise_conv: bool = False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate, num_convs=num_convs, kernel_size=kernel_size, padding=padding, depthwise_conv=depthwise_conv)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.1, 
                 num_convs=2, kernel_size=3, padding=None, depthwise_conv: bool = False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate, 
                                   num_convs=num_convs, kernel_size=kernel_size, padding=padding, depthwise_conv = depthwise_conv)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate, 
                                   num_convs=num_convs, kernel_size=kernel_size, padding=padding, depthwise_conv = depthwise_conv)

    def forward(self, x1, x2):
        # Upsample x1 first
        # print("Shape of up block input before upsampling :", x1.shape)

        x1 = self.up(x1)
        
        # print("Shape of up block input :", x1.shape, x2.shape)

        # Determine minimum height and width
        min_height = min(x1.size(2), x2.size(2))
        min_width = min(x1.size(3), x2.size(3))
        
        # Crop the center regions of x1 and x2 to match min_height and min_width
        # Note: if (proper) padding is used, this won't change anything
        x1 = x1[:, :, (x1.size(2) - min_height) // 2 : (x1.size(2) + min_height) // 2,
                    (x1.size(3) - min_width) // 2 : (x1.size(3) + min_width) // 2]
        x2 = x2[:, :, (x2.size(2) - min_height) // 2 : (x2.size(2) + min_height) // 2,
                    (x2.size(3) - min_width) // 2 : (x2.size(3) + min_width) // 2]
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)

        # print("Shape of up block output:", x.shape)
        
        # Pass through the convolutional layer
        return self.conv(x)


    # def forward(self, x1, x2):
    #     x1 = self.up(x1)
    #     # input is CHW
    #     diffY = x2.size()[2] - x1.size()[2]
    #     diffX = x2.size()[3] - x1.size()[3]

    #     x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    #                     diffY // 2, diffY - diffY // 2])

    #     x = torch.cat([x2, x1], dim=1)
    #     print("Decoder pass:", x.shape)
    #     return self.conv(x)

class CNNCommunicator(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, kernel_size=3, padding=1):
        super(CNNCommunicator, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        # Second convolutional layer
        self.conv2 = nn.Conv2d((in_channels + out_channels) // 2, (in_channels + out_channels) // 2, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d((in_channels + out_channels) // 2)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        # Third convolutional layer
        self.conv3 = nn.Conv2d((in_channels + out_channels) // 2, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        #print(x.shape)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        return x

class UpResNet(nn.Module):
    """
    UpResNet module for image processing tasks.

    Args:
        in_channels_1 (int): Number of input channels for the first input.
        in_channels_2 (int): Number of input channels for the second input.
        out_channels (int): Number of output channels.
        bilinear (bool, optional): Whether to use bilinear interpolation. Default is True.
        dropout_rate (float, optional): Dropout rate. Default is 0.1.
    """

    def __init__(self, in_channels_1, in_channels_2, out_channels, bilinear=True, dropout_rate=0.1):
        super(UpResNet, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels_1, in_channels_1 // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels_1 // 2 + in_channels_2, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        """
        Forward pass for the UpResNet module.

        Args:
            x1 (torch.Tensor): Input tensor from the first pathway.
            x2 (torch.Tensor): Input tensor from the second pathway.

        Returns:
            torch.Tensor: Output tensor.
        """

        x1 = self.up(x1)

        # Pad x1 to have the same size as x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class UpComm(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.1):
        super(UpComm, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(2 * in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetCommunicator(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, bilinear=False):
        super(UNetCommunicator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, in_channels))
        self.down1 = (Down(in_channels, in_channels, dropout_rate=dropout_rate))
        self.down2 = (Down(in_channels, in_channels, dropout_rate=dropout_rate))
        self.down3 = (Down(in_channels, in_channels, dropout_rate=dropout_rate))

        factor = 2 if bilinear else 1
        
        self.up1 = (UpComm(in_channels, in_channels, dropout_rate=dropout_rate, bilinear=bilinear))
        self.up2 = (UpComm(in_channels, in_channels, dropout_rate=dropout_rate, bilinear=bilinear))
        self.up3 = (UpComm(in_channels, in_channels, dropout_rate=dropout_rate, bilinear=bilinear))
        self.outc = (OutConv(in_channels, out_channels))

    def forward(self, data):
        x1 = self.inc(data)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        output = self.outc(x)

        return output