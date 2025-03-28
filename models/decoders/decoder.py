from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, input_channels, k=50, adaptive=True, serial=False, bias=False):
        super(Decoder, self).__init__()
        self.k = k
        self.adaptive = adaptive
        self.serial = serial
        
        self.conv = nn.ModuleList([nn.Conv2d(in_ch, 256, kernel_size=1, bias=bias) for in_ch in input_channels])
        
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.out5 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8, mode='nearest')
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='nearest')
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.probability_map = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, bias=bias),
            nn.Sigmoid()
        )
        
        if self.adaptive:
            in_thresh_channels = 256
            if self.serial:
                in_thresh_channels += 1 
            self.threshold_map = nn.Sequential(
                nn.Conv2d(in_thresh_channels, 64, kernel_size=3, padding=1, bias=bias),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=bias),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, bias=bias),
                nn.Sigmoid()
            )
        
        self.apply(self.weights_init)
        if self.adaptive:
            self.threshold_map.apply(self.weights_init)
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('BatchNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight.data, 1)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 1e-4)
    
    def approximate_binary_map(self, x, y):
        return torch.sigmoid(self.k * (x - y))
    
    def step_function(self, probability_map, threshold_map):
        return (probability_map > threshold_map).float()
    
    def forward(self, inputs, training_mode):
        '''
        inputs:
            (N, 64, H/4, W/4)
            (N, 128, H/8, W/8)
            (N, 256, H/16, W/16)
            (N, 512, H/32, W/32)
        
        outputs:
            training:
                adaptive=True:
                    probability_map: (N, 1, H, W)
                    threshold_map: (N, 1, H, W)
                    binary_map: (N, 1, H, W)
                adaptive=False:
                    probability_map: (N, 1, H, W)
            inference:
                adaptive=True:
                    probability_map: (N, 1, H, W)
                    binary_map: (N, 1, H, W)
                adaptive=False:
                    probability_map: (N, 1, H, W)
                    binary_map: (N, 1, H, W)
        '''
        
        in2 = self.conv[0](inputs[0]) # (N, 256, H/4, W/4)
        in3 = self.conv[1](inputs[1]) # (N, 256, H/8, W/8)
        in4 = self.conv[2](inputs[2]) # (N, 256, H/16, W/16)
        in5 = self.conv[3](inputs[3]) # (N, 256, H/32, W/32)
        
        out4 = self.up5(in5) + in4 # (N, 256, H/16, W/16)
        out3 = self.up4(out4) + in3 # (N, 256, H/8, W/8)
        out2 = self.up3(out3) + in2 # (N, 256, H/4, W/4)
        
        p5 = self.out5(in5) # (N, 64, H, W)
        p4 = self.out4(out4) # (N, 64, H, W)
        p3 = self.out3(out3) # (N, 64, H, W)
        p2 = self.out2(out2) # (N, 64, H, W)
        
        fuse = torch.cat([p5, p4, p3, p2], dim=1) # (N, 256, H, W)
        probability_map = self.probability_map(fuse) # (N, 1, H, W)
        
        if training_mode:
            if self.adaptive:
                if self.serial:
                    fuse = torch.cat((fuse, F.interpolate(probability_map, size=fuse.shape[2:], mode='nearest')), dim=1)
                threshold_map = self.threshold_map(fuse) # (N, 1, H, W)
                binary_map = self.approximate_binary_map(probability_map, threshold_map) # (N, 1, H, W)
                return OrderedDict({
                    "probability_map": probability_map,
                    "threshold_map": threshold_map,
                    "binary_map": binary_map
                })
            else:
                return OrderedDict({
                    "probability_map": probability_map,
                })
        else:
            binary_map = (probability_map > 0.2).float()
            
            return OrderedDict({
                "probability_map": probability_map,
                "binary_map": binary_map
            })