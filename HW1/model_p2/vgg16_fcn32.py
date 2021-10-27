import torch.nn as nn
from torchvision import models


class Vgg16FCN32(nn.Module):
    def __init__(self, num_classes=7):
        super(Vgg16FCN32, self).__init__()
        model = models.vgg16(pretrained=True)
        # TODO: consider layer to freeze
        # freeze pool 1 to pool 3
        for i in range(17):
            for param in model.features[i].parameters():
                param.requires_grad = False
        self.features = model.features  # output size [-1, 512, 16, 16]

        # self.avgpool = model.avgpool  # output size [-1, 512, 16, 16]

        self.fcn6 = nn.Sequential(
            # nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(512, 4096, 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.fcn7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        # self.score = nn.Conv2d(4096, num_classes, kernel_size=(1, 1))
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32, bias=False)
        # self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=(64, 64), stride=(32, 32), bias=False)

    def forward(self, x):
        x = self.features(x)
        x = self.fcn6(x)
        x = self.fcn7(x)
        x = self.score_fr(x)
        x = self.upscore(x)
        return x
