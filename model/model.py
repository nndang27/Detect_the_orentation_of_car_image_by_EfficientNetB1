import torch.nn as nn
import torch
from torchvision import models
def build_model(num_classes):
    # model = torch.hub.load('pytorch/vision:v0.13.0', 'resnext50_32x4d', weights="IMAGENET1K_V2")
 
    # model.fc = nn.Sequential(
    #             nn.Linear(2048, 128),
    #             nn.ReLU(inplace=True),
    #             nn.Dropout(0.2),
    #             nn.Linear(128, num_classes))
    model = models.efficientnet_b1(pretrained=True)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model