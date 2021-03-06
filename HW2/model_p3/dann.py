from abc import ABC

import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.alpha
        return grad_output, None

    def grad_reverse(x, alpha):
        return GradReverse.apply(x, alpha)


class DANN(nn.Module):
    def __init__(self, num_classes=10):
        super(DANN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 48, 5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),

            nn.Linear(100, num_classes),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )

    def forward(self, x, alpha):
        feat = self.feature(x)
        feat = feat.view(-1, 48 * 4 * 4)

        class_out = self.classifier(feat)

        domain_out = GradReverse.grad_reverse(feat, alpha)
        domain_out = self.discriminator(domain_out)

        return class_out, domain_out


class DigitClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitClassifier, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 48, 5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),

            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        feat = self.feature(x)
        feat = feat.view(-1, 48 * 4 * 4)

        class_out = self.classifier(feat)

        return class_out


class ExtractedDANN(DANN):
    def __init__(self, num_classes=10):
        super().__init__(num_classes)

    def forward(self, x, alpha):
        feat = self.feature(x)
        feat = feat.view(-1, 48 * 4 * 4)
        return feat


if __name__ == '__main__':
    from torchsummary import summary

    # net = Usps2Svhn()
    # net.to("cuda")
    # summary(net, (3, 28, 28))
    # r = torch.randn((10, 3, 28, 28), device="cuda")
    # a, b = net(r, alpha=0.5)

    # print(out.size())

    # classifier = Classifier()
    # classifier.to("cuda")
    # summary(classifier, (48*4*4,))
    # r = torch.randn((10, 48 * 4 * 4), device="cuda")
    # print(classifier(r).size())
    #
    # dis = Discriminator()
    # dis.to("cuda")
    # # summary(dis, (48*4*4,))
    # r = torch.randn((10, 48 * 4 * 4), device="cuda")
    # print(dis(r, alpha=0.5).size())
    net = DigitClassifier()
    net.to("cuda")
    summary(net, (3, 28, 28))
    r = torch.randn((10, 3, 28, 28), device="cuda")
    print(net(r).size())
