import torch.nn as nn
from torchvision import models
import timm

def build_resnet50(num_classes: int, pretrained: bool = True):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
def build_densenet121(num_classes: int, pretrained: bool = True):
    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m
def build_efficientnet_b0(num_classes: int, pretrained: bool = True):#'efficientnet_b0'
    m = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    return m

def build_swin_tiny(num_classes: int,  pretrained: bool = True):
    m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=num_classes )
    return m