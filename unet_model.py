import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import CenterCrop

# basic convolutional block 
class UNetConvolutionalBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        
        super().__init__()
        
        # conv 3x3 -> relu -> conv 3x3 -> relu 
        # and, as in the original paper, convolutions are unpadded
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.relu_ = nn.ReLU()
        
        # note, as in the original paper, we are not applying any normalisation
        
    
    # do the forward pass through the UNetConvolutionalBlock
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu_(out)
        out = self.conv2(out)
        out = self.relu_(out)
        return out


# u-net encoder module
class Encoder(nn.Module):
    
    def __init__(self, enc_channels=[3, 64, 128, 256, 512, 1024]):
        
        super().__init__()
        
        # get all the convolutional blocks in the encoder network
        block_list = [
            UNetConvolutionalBlock(enc_channels[i], enc_channels[i+1]) \
            for i in range(len(enc_channels) - 1)
        ]
        self.encoder_blocks = nn.ModuleList(block_list)
        
        # get the pooling layer that will follow each convolutional block
        self.pool = nn.MaxPool2d(2, 2)
        
      
    # do the forward pass through the UNet encoder (getting the feature output
    # at each stage)
    def forward(self, x):
        
        features = list()
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
            
        return features
    
    
# u-net decoder module
class Decoder(nn.Module):
    
    def __init__(self, dec_channels=[1024, 512, 256, 128, 64]):
        
        super().__init__()
        self.channels = dec_channels
        
        # get the up-convolutions
        up_list = [
            nn.ConvTranspose2d(dec_channels[i], dec_channels[i+1], 2, 2) \
            for i in range(len(dec_channels) - 1)
        ]
        self.upconvs  = nn.ModuleList(up_list)
                
        # get all the convolutional blocks in the decoder network
        block_list = [
            UNetConvolutionalBlock(dec_channels[i], dec_channels[i+1]) \
            for i in range(len(dec_channels) - 1)
        ]
        self.decoder_blocks = nn.ModuleList(block_list)
        
        
    # do the forward pass through the UNet decoder. here we upsample (through 
    # up-convolutions), concatenate the feature map from the encoder, and 
    # apply our basic convolutional block
    def forward(self, x, encoder_features):
        
        for i in range(len(self.channels) - 1):            
            x = self.upconvs[i](x)
            y = self.crop(encoder_features[i], x)
            x = torch.cat([x, y], dim=1)
            x = self.decoder_blocks[i](x)
            
        return x
                        
        
    # take the centre crop of the feature map
    def crop(self, features, x):
        
        _, _, h, w = x.shape
        features = CenterCrop([h, w])(features)
        return features
    
    
# put everything together to form the unet model
class UNet(nn.Module):
    
    def __init__(self, num_classes=1,
                 enc_channels=[3, 64, 128, 256, 512, 1024],
                 dec_channels=[1024, 512, 256, 128, 64]):
        
        super().__init__()
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        self.head    = nn.Conv2d(dec_channels[-1], num_classes, 1)
        
        
    def forward(self, x):
        features = self.encoder(x)
        out      = self.decoder(features[::-1][0], features[::-1][1:])
        out      = self.head(out)
        
        return(out)