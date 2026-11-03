import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

from .base import BaseModel

logger = logging.getLogger(__name__)


class DecoderBlock(nn.Module):
    """FPN :  skip  1×1 ， previous """


    def __init__(self,
                 previous,   # skip feature 
                 out,         #
                 ksize=3,      #
                 num_convs=1,     #
                 norm=nn.BatchNorm2d,   #
                 padding="zeros"):       #padding 
        super().__init__()
        layers = []         #
        for i in range(num_convs):
            conv = nn.Conv2d(
                previous if i == 0 else out,
                out,
                kernel_size=ksize,
                padding=ksize // 2,
                bias=norm is None, padding_mode=padding,
            )
            layers.append(conv)
            if norm is not None:
                layers.append(norm(out))
            layers.append(nn.ReLU(inplace=True))     #rulu
        self.layers = nn.Sequential(*layers)

    def forward(self, previous,  #FPN
                skip            #Encoder
                ):
        """previous: , skip: """
        _, _, hp, wp = previous.shape
        _, _, hs, ws = skip.shape
        # （2、4、8...）
        scale = 2 ** np.round(np.log2(np.array([hs / hp, ws / wp])))
        upsampled = F.interpolate(previous, scale_factor=scale.tolist(),      #
                                   mode="bilinear", align_corners=False)

        # ：
        _, _, hu, wu = upsampled.shape
        _, _, hs, ws = skip.shape
        if (hu <= hs) and (wu <= ws):
            skip = skip[:, :, :hu, :wu]          # 
        elif (hu >= hs) and (wu >= ws):
            skip = F.pad(skip, [0, wu - ws, 0, hu - hs])  # 
        else:
            raise ValueError(f"Inconsistent shapes: {(hs, ws)}, {(hu, wu)}")

        return self.layers(skip) + upsampled     # : skip + 


class FPN(nn.Module):
    """
     (Feature Pyramid Network)

    :
      layer4 (, ) →  + layer3 → ... → layer1 → 
       + 
    """

    def __init__(self, in_channels_list, out_channels, **kw):
        """
        in_channels_list:  [C0, C1, C2, C3, C4]
         ResNet50: [64, 256, 512, 1024, 2048]
        """
        super().__init__()
        #  1×1 
        self.first = nn.Conv2d(in_channels_list[-1], out_channels, 1, padding=0, bias=True)
        # : 
        self.blocks = nn.ModuleList([
            DecoderBlock(c, out_channels, ksize=1, **kw)
            for c in in_channels_list[::-1][1:]  # ，
        ])
        # 
        self.out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, layers):
        """layers: {'relu': [B,64,...], 'layer1': [B,256,...], ...}"""
        feats = None
        for idx, x in enumerate(reversed(layers.values())):  # 
            if feats is None:
                feats = self.first(x)               # layer4 → 1×1 conv
            else:
                feats = self.blocks[idx - 1](feats, x)  #  + skip
        return self.out(feats)


def remove_conv_stride(conv):
    conv_new = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        bias=conv.bias is not None,
        stride=1,
        padding=conv.padding,
    )
    conv_new.weight = conv.weight
    conv_new.bias = conv.bias
    return conv_new


class FeatureExtractor(BaseModel):
    """
    UAV : ResNet50  + FPN 

    : [B, 3, 256, 256]  RGB  (ImageNet )
    : [B, 128, 128, 128]  128 

     ResNet50 + FPN？
      ResNet  → 
      FPN  →  + 
      128×128  128  = ""
    """

    default_conf = {
        "pretrained": True,
        "input_dim": 3,
        "output_dim": 128,
        "encoder": "resnet50",
        "remove_stride_from_first_conv": False,
        "num_downsample": None,
        "decoder_norm": "nn.BatchNorm2d",
        "do_average_pooling": False,
        "checkpointed": False,
    }
    mean = [0.485, 0.456, 0.406]   # ImageNet 
    std = [0.229, 0.224, 0.225]    # ImageNet 

    def build_encoder(self, conf):
        """:  torchvision  pretrained ResNet/VGG，"""
        assert isinstance(conf.encoder, str)
        if conf.pretrained:
            assert conf.input_dim == 3
        Encoder = getattr(torchvision.models, conf.encoder)

        kw = {}
        if conf.encoder.startswith("resnet"):
            #  ResNet  5 
            layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
            kw["replace_stride_with_dilation"] = [False, False, False]
        elif conf.encoder == "vgg13":
            layers = ["features.3", "features.8", "features.13", "features.18", "features.23"]
        elif conf.encoder == "vgg16":
            layers = ["features.3", "features.8", "features.15", "features.22", "features.29"]
        else:
            raise NotImplementedError(conf.encoder)

        if conf.num_downsample is not None:
            layers = layers[: conf.num_downsample]
        encoder = Encoder(weights="DEFAULT" if conf.pretrained else None, **kw)
        encoder = create_feature_extractor(encoder, return_nodes=layers)  # 
        if conf.encoder.startswith("resnet") and conf.remove_stride_from_first_conv:
            encoder.conv1 = remove_conv_stride(encoder.conv1)  # 

        return encoder, layers

    def _init(self, conf):
        # ImageNet （ buffer， CPU/GPU）
        self.register_buffer("mean_", torch.tensor(self.mean), persistent=False)
        self.register_buffer("std_", torch.tensor(self.std), persistent=False)

        #  + 
        self.encoder, self.layers = self.build_encoder(conf)
        s = 128
        inp = torch.zeros(1, 3, s, s)
        features = list(self.encoder(inp).values())
        self.skip_dims = [x.shape[1] for x in features]        # 
        self.layer_strides = [s / f.shape[-1] for f in features] # 
        self.scales = [self.layer_strides[0]]

        #  FPN 
        norm = eval(conf.decoder_norm) if conf.decoder_norm else None
        self.decoder = FPN(self.skip_dims, out_channels=conf.output_dim, norm=norm)

    def _forward(self, data):
        """
        : data["image"]  [B, 3, H, W]  (0~1)
        : feature_maps [B, 128, 128, 128]  (128 )
        """
        image = data["image"]
        image = (image - self.mean_[:, None, None]) / self.std_[:, None, None]  # 

        skip_features = self.encoder(image)   # ResNet 
        output = self.decoder(skip_features)  # FPN 
        return {"feature_maps": [output], "skip_features": skip_features}
