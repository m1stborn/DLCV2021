import torch.nn as nn
from torchvision import models


class ResnetFCN32(nn.Module):
    def __init__(self, num_classes=7):
        super(ResnetFCN32, self).__init__()
        model = models.resnet50(pretrained=True)

        for name, child in model.named_children():
            if name not in ['layer3', 'layer4']:
                for param in child.parameters():
                    param.requires_grad = False

        self.features = nn.Sequential(*list(model.children())[:-2])
        # output : [-1, 2048, 7, 7]

        self.classifier = nn.Sequential(
            nn.Conv2d(2048, num_classes, kernel_size=1),
            # nn.Sigmoid()
            nn.ReLU()
        )

        self.upsampling32 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64,
                                               stride=32, bias=False)

    def forward(self, x):
        o = x
        for feature in self.features:
            o = feature(o)
        o = self.classifier(o)
        o = self.upsampling32(o)
        cx = int((o.shape[3] - x.shape[3]) / 2)
        cy = int((o.shape[2] - x.shape[2]) / 2)
        o = o[:, :, cy:cy + x.shape[2], cx:cx + x.shape[3]]

        return o


# TODO:remove for submission

if __name__ == '__main__':
    from torchsummary import summary

    net = ResnetFCN32()
    net.to("cuda")
    # summary(net, (3, 512, 512))

    import torch
    import numpy as np
    inputs = torch.randn(1, 3, 512, 512).to(device='cuda')
    out = net(inputs)
    print(type(out))
    print(out.size())

    om = torch.argmax(out, dim=1).detach().cpu().numpy()
    print(om.shape)
    print(np.unique(om))
