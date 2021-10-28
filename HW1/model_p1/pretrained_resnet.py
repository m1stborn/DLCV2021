import torch.nn as nn
from torchvision import models


class PretrainedResnet(nn.Module):
    def __init__(self, num_classes=50):
        super(PretrainedResnet, self).__init__()
        model = models.resnet50(pretrained=True)
        for name, child in model.named_children():
            if name not in ['layer3', 'layer4']:
                for param in child.parameters():
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


class ExtractedResnet(PretrainedResnet):
    def __init__(self, num_classes=50):
        super().__init__(num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2048)
        return x


# if __name__ == '__main__':
    # net = PretrainedResnet()
    # net.to("cuda")
    # print(net)

    # net = ExtractedResnet()
    # net.to("cuda")
    # import torch
    # inputs = torch.randn(3, 3, 224, 224).to(device='cuda')
    # out = net(inputs)
    # print(out.size())

    # x = torch.randn(3, 2048)
    # lst = [x for i in range(3)]
    # print(len(lst))
    # print(torch.cat(lst).size())
    # val_pred = torch.cat(lst).cpu().numpy()
    # print(val_pred.shape)

