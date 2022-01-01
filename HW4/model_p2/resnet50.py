import torch.nn as nn
from torchvision import models


class Resnet(nn.Module):
    def __init__(self, model=None, num_classes=65, freeze=False, hidden_dim=4096):
        super(Resnet, self).__init__()
        self.num_classes = num_classes
        self.freeze = freeze
        self.hidden_dim = hidden_dim

        if model is None:
            model = models.resnet50(pretrained=False)

        if self.freeze:
            for name, child in model.named_children():
                for param in child.parameters():
                    param.requires_grad = False

        self.features = nn.Sequential(*list(model.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(2048, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(2048, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(self.hidden_dim, self.num_classes)
        # )

        # self.classifier = nn.Sequential(
        #     nn.Linear(2048, self.num_classes)
        # )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2048)
        out = self.classifier(x)
        return out


if __name__ == '__main__':
    net = Resnet()
    from torchsummary import summary
    net.to('cuda')
    summary(net, (3, 128, 128))
