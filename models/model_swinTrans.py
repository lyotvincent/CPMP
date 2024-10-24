# borrowed from https://github.com/Xiyue-Wang/TransPath

# IMPORTANT!!!
# for using ctranspath model,
# need to download the modified timm library and Install it into the target path.  [Thus it will not influnce the original timm library in the environment]
# `pip install timm-0.5.4.tar --target=~/anaconda3/envs/torch20/lib/python3.8/site-packages/timm-0.5.4`

# Then, we can import them along with dependencies on demand
# Here we can use sys.path.insert,this will temporarily add the path for import 
import sys, os # 
sys.path.insert(1, os.path.abspath('/data/cyyan/miniconda3/envs/torch20/lib/python3.8/site-packages/timm-0.5.4'))

import timm
from timm.models.layers.helpers import to_2tuple
# import torchvision.models.swin_transformer
import torch
import torch.nn as nn


class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def ctranspath():
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    model.head = nn.Identity()
    return model


if __name__ == "__main__":
    model = ctranspath()
    
    pretext_model = torch.load("../pretrained_model_weights/ctranspath.pth")
    model.load_state_dict(pretext_model['model'], strict=True)