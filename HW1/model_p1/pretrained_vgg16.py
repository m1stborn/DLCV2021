import torch.nn as nn
from torchvision import models


class PretrainedVGG16(nn.Module):
    def __init__(self, num_classes=50):
        super(PretrainedVGG16, self).__init__()
        model = models.vgg16(pretrained=True)
        for i in range(17):
            for param in model.features[i].parameters():
                param.requires_grad = False
        # for param in model.parameters():
        #     param.requires_grad = False
        self.features = model.features

        self.avgpool = model.avgpool

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.classifier(x)
        return x
