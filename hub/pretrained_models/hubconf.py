
"""
In order to load a pre-trained model using torch.hub.load() from a local directory, 
You need to have the hubconf.py in your torch.hub.get_dir().
you cand find the full script here: https://github.com/pytorch/vision/blob/main/hubconf.py
If you have access to the interent, you don't need to be worried about, since
pytorch will automatically handle it from its pre-trained models' repository.
"""
# Optional list of dependencies required by the package
dependencies = ["torch"]

from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201
from torchvision.models.efficientnet import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
)
from torchvision.models.resnet import (
    resnet101,
    resnet152,
    resnet18,
    resnet34,
    resnet50,
    resnext101_32x8d,
    resnext101_64x4d,
    resnext50_32x4d,
    wide_resnet101_2,
    wide_resnet50_2,
)
