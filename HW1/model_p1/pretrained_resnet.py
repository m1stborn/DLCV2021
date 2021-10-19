import torch.nn as nn
from torchvision import models


class PretrainedResnet(nn.Module):
    def __init__(self, num_classes=50):
        super(PretrainedResnet, self).__init__()
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        self.features = nn.Sequential(*list(model.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2048)
        out = self.classifier(x)
        return out
