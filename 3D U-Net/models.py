from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

Home_dir = Path(__file__).parent


class ConvBlock(torch.nn.Module):
    """Double convolution block with batch normalization and ReLU"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


    
class DoubleConv3D(torch.nn.Module):
    """
    Helper Class which implements the intermediate Convolutions
    """
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, 3, padding=1),
                                        torch.nn.BatchNorm3d(out_channels),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.BatchNorm3d(out_channels),
                                        torch.nn.ReLU())
        
    def forward(self, X):
        return self.step(X)


class Down(nn.Module):
    """Downsampling block: MaxPool2d + ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # keep both 2d and 3d versions
        self.down = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)
 

    def forward(self, x):
        return self.conv(self.down(x))


class Up(nn.Module):
    """Upsampling block: ConvTranspose2d + concatenation + ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = ConvBlock(in_channels, out_channels, in_channels//2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class UNet2D(torch.nn.Module):
    """2D UNet for image segmentation"""
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        """
        A single model that performs segmentation

        Args:
            in_channels: int, number of input channels
            num_classes: int, number of output classes
        """
        super().__init__()

        # downsampling path
        self.input = ConvBlock(in_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)

        # upsampling path
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)

        # Output layer
        self.logits_layer = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used in training, takes an image and returns raw logits.

        Args:
            x (torch.FloatTensor): image with shape (b, in_channels, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: logits (b, num_classes, h, w)
        """
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)

        logits = self.logits_layer(out)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction pass

        Args:
            x (torch.FloatTensor): image with shape (b, in_channels, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: logits (b, num_classes, h, w)
        """
        return self(x)
   
class UNet3D(torch.nn.Module):
    """
    This class implements a UNet for the Segmentation
    We use 3 down- and 3 UpConvolutions and two Convolutions in each step
    """

    def __init__(self,in_channels: int =1, num_classes: int =2):
        """Sets up the U-Net Structure
        """
        super().__init__()
        
        
        ############# DOWN #####################
        self.layer1 = DoubleConv3D(in_channels, 32)
        self.layer2 = DoubleConv3D(32, 64)
        self.layer3 = DoubleConv3D(64, 128)
        self.layer4 = DoubleConv3D(128, 256)

        #########################################

        ############## UP #######################
        self.layer5 = DoubleConv3D(256 + 128, 128)
        self.layer6 = DoubleConv3D(128+64, 64)
        self.layer7 = DoubleConv3D(64+32, 32)
        self.layer8 = torch.nn.Conv3d(32, num_classes, 1)  # Output: 2 values -> background
        #########################################

        self.maxpool = torch.nn.MaxPool3d(2)

    def forward(self, x):
        
        ####### DownConv 1#########
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)
        ###########################
        
        ####### DownConv 2#########        
        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)
        ###########################

        ####### DownConv 3#########        
        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)
        ###########################
        
        ##### Intermediate Layer ## 
        x4 = self.layer4(x3m)
        ###########################

        ####### UpCONV 1#########        
        x5 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x4)  # Upsample with a factor of 2
        x5 = torch.cat([x5, x3], dim=1)  # Skip-Connection
        x5 = self.layer5(x5)
        ###########################

        ####### UpCONV 2#########        
        x6 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x5)        
        x6 = torch.cat([x6, x2], dim=1)  # Skip-Connection    
        x6 = self.layer6(x6)
        ###########################
        
        ####### UpCONV 3#########        
        x7 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x6)
        x7 = torch.cat([x7, x1], dim=1)       
        x7 = self.layer7(x7)
        ###########################
        
        ####### Predicted segmentation#########        
        logits = self.layer8(x7)

        return logits

class DoubleConv3D_PLUS(nn.Module):
    """
    Residual double 3D conv: requires in_channels == out_channels.
    """
    def __init__(self, channels):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.out_act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        return self.out_act(self.step(x) + x)


class DOWN_3D(nn.Module):
    """Two residual blocks + strided conv downsample."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res1 = DoubleConv3D_PLUS(in_channels)
        self.res2 = DoubleConv3D_PLUS(in_channels)
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        skip = x
        down = self.down(x)
        return down, skip


class UP_3D(nn.Module):
    """
    Transposed conv upsample from in_channels -> out_channels,
    concat with skip (skip_channels), then two residual blocks
    operating on (out_channels + skip_channels) channels.
    """
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        cat_channels = out_channels + skip_channels
        self.res1 = DoubleConv3D_PLUS(cat_channels)
        self.res2 = DoubleConv3D_PLUS(cat_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # B, (out+skip), D,H,W
        x = self.res1(x)
        x = self.res2(x)
        return x


class UNet3D_PLUS(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()

        # Encoder
        self.down0 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(16, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.down1 = DOWN_3D(16, 32)   # skip: 16, down: 32
        self.down2 = DOWN_3D(32, 64)   # skip: 32, down: 64
        self.down3 = DOWN_3D(64, 128)  # skip: 64, down: 128

        self.bottleneck = nn.Sequential(
            DoubleConv3D_PLUS(128),
            DoubleConv3D_PLUS(128),
        )

       
        self.up3 = UP_3D(in_channels=128, out_channels=64,  skip_channels=64)   # output: 64+64=128
        self.up2 = UP_3D(in_channels=128, out_channels=32,  skip_channels=32)   # output: 32+32=64
        self.up1 = UP_3D(in_channels=64,  out_channels=16,  skip_channels=16)   # output: 16+16=32

       
        self.final_reduce = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(16, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.output_layer = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.down0(x)             
        x1, s1 = self.down1(x0)         
        x2, s2 = self.down2(x1)         
        x3, s3 = self.down3(x2)         
        b  = self.bottleneck(x3)        

        # Decoder
        u3 = self.up3(b,  s3)          
        u2 = self.up2(u3, s2)           
        u1 = self.up1(u2, s1)           
        x  = self.final_reduce(u1)     
        logits = self.output_layer(x)   
        return logits



class LinearUNETR(torch.nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int=2):
        super().__init__()
        self.linear_unetr = Linear_UNETR(
            input_dim=in_channels,
            output_dim=num_classes,
            img_shape=(96, 96, 32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear_unetr(x)
      
        return logits
    


MODEL_FACTORY = {
    "UNet2D":       UNet2D,
    "UNet3D":       UNet3D,
    "UNet3D_PLUS":  UNet3D_PLUS,
    "LinearUNETR":  LinearUNETR,

}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
   
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = Home_dir / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}"
            ) from e

  
    model_size_mb = calculate_model_size_mb(m)

    print(f"{model_name} model size: {model_size_mb:.2f} MB")

    return m

def save_model(model: torch.nn.Module, kfold_idx: int=None) -> str:
    """
    Save the model's state_dict to disk.
    If kfold_idx is provided, include it in the filename.

    Args:
        model: torch.nn.Module, the model to save
        kfold_idx: int or None, optional fold index for k-fold experiments

    Returns:
        str: Path to the saved model file
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")
    
    if kfold_idx is not None:
        model_name_new = f"{model_name}_fold{kfold_idx}"

    experiment_dir = Home_dir / "experiment"
    experiment_dir.mkdir(exist_ok=True)
    output_path = experiment_dir / f"{model_name_new}.th"
    torch.save(model.state_dict(), output_path)

    return str(output_path)


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024





