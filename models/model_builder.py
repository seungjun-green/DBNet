import torch
import torch.nn as nn
from models.backbones.resnet import resnet18, resnet34, resnet50
from models.decoders.decoder import Decoder

class DBNet(nn.Module):
    def __init__(self, backbone, input_channels, k, adaptive, serial, bias):
        super(DBNet, self).__init__()
        
        # initalize backbone
        if backbone == "resnet18":
            self.backbone = resnet18()
        elif backbone == "resnet34":
            self.backbone = resnet34()
        elif backbone == "resnet50":
            self.backbone = resnet50()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # initalize the decoder
        self.decoder = Decoder(input_channels=input_channels, k=k, adaptive=adaptive, serial=serial, bias=bias)
        
    def forward(self, x, training_mode):
        features = self.backbone(x) # returns x1, x2, x3, x4
        outputs = self.decoder(features, training_mode)
        return outputs        