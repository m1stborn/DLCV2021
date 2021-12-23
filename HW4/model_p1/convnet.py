import torch
import torch.nn as nn


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def cosine_similarity(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    return cos(a, b)


class ParametricDist(nn.Module):
    def __init__(self):
        super().__init__()
        self.dist = nn.Sequential(
            nn.Linear(1600, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )

    def forward(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        return self.dist(a-b).squeeze(-1)


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    # since our goal is to minimize the loss
    # => place negative sign to distance
    # => the closer the distance, the smaller the loss
    return logits


if __name__ == '__main__':
    # from torchsummary import summary
    #
    # net = Convnet()
    # net.to('cuda')
    # summary(net, (3, 84, 84))
    # r = torch.randn(1, 3, 84, 84, device="cuda")
    # print(net(r).size())

    # v1 = torch.Tensor([[-0.7715, -0.6205, -0.2562]])
    # v2 = torch.Tensor([[-1.7715, -0.6205, -0.2562]])
    # v3 = torch.Tensor([[-2.7715, -0.6205, -0.2562]])
    #
    # print(cosine_similarity(v1, v2))  # 0.93
    # print(cosine_similarity(v1, v3))  # 0.8877
    # print(cosine_similarity(v2, v3))
    # print(cosine_similarity(v1, v1))

    # logits = torch.Tensor([[0.9, 0.05, 0.05]])
    # label = torch.Tensor([0]).long()
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(logits, label)
    # print(loss)

    v1 = torch.randn(1, 1600)
    v2 = torch.randn(10, 1600)

    dist = ParametricDist()
    output = dist(v1, v2)
    # print(output.size())
    # print(output[0])
    print(dist)
    # output2 = euclidean_metric(v1, v2)
    # print(output2.size())
    # print(output2[0])
    #
    # output3 = cosine_similarity(v1, v2)
    # print(output3.size())
    # print(output3[0])

