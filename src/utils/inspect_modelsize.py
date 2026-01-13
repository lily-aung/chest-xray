import torch
import torch.nn as nn
from torchvision import models
import timm

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def inspect_model(name, model):
    total, trainable = count_params(model)
    print("=" * 60)
    print(f"Model: {name}")
    print(f"Total params:      {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Classifier layer: {model.__class__.__name__}")
    print()


def main(num_classes=3):
    # ------------------------
    # ResNet-50
    # ------------------------
    resnet = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2 )
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    inspect_model("ResNet-50", resnet)

    # ------------------------
    # DenseNet-121
    # ------------------------
    densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 )
    densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)
    inspect_model("DenseNet-121", densenet)

    # ------------------------
    # EfficientNet-B0
    # ------------------------
    effnet = timm.create_model(  "efficientnet_b0", pretrained=True, num_classes=num_classes)
    inspect_model("EfficientNet-B0", effnet)
    
if __name__ == "__main__":
    main()
