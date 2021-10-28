import torch.nn as nn
from torchvision import models


class Vgg16FCN8(nn.Module):
    def __init__(self, num_classes=7):
        super(Vgg16FCN8, self).__init__()
        model = models.vgg16(pretrained=True)

        self.pool1_to_3 = nn.Sequential(*list(model.features[:17]))

        self.pool4 = nn.Sequential(*list(model.features[17:24]))

        self.pool5 = nn.Sequential(*list(model.features[24:]))

        # TODO: consider adding fcn7
        self.fcn6 = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),
            # TODO: consider sigmoid
            # nn.Sigmoid()
            nn.ReLU()
        )

        self.upsampling2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4,
                                              stride=2, bias=False)
        self.upsampling8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16,
                                              stride=8, bias=False)

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        # TODO: consider use another up-sampling 2?
        # self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)

    def forward(self, x):
        h = x

        h = self.pool1_to_3(h)
        pool3 = h  # 1/8
        # print('pool3', pool3.size())

        h = self.pool4(h)
        pool4 = h  # 1/16
        # print('pool4', pool4.size())

        h = self.pool5(h)  # 1/32
        # print('pool5', h.size())

        h = self.fcn6(h)
        h = self.upsampling2(h)
        # upscore2 = h
        upscore2 = h[:, :, 1:1 + pool4.size()[2], 1:1 + pool4.size()[3]]  # 1/16
        # print('upscore2', upscore2.size())  # torch.Size([1, 7, 34, 34])

        h = self.score_pool4(pool4)
        # print('before score_pool4c', h.size())  # torch.Size([1, 7, 32, 32])
        # h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16
        # print('score_pool4c', score_pool4c.size())

        h = upscore2 + score_pool4c  # 1/16 + 1/16
        h = self.upsampling2(h)
        upscore_pool4 = h[:, :, 1:1 + pool3.size()[2], 1:1 + pool3.size()[2]]  # 1/8
        # print('upscore_pool4', upscore_pool4.size()) # torch.Size([1, 7, 64, 64])

        h = self.score_pool3(pool3)
        # print('before score_pool4c', h.size())
        # h = h[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8
        # print('score_pool3c', score_pool3c.size())

        h = upscore_pool4 + score_pool3c  # 1/8 + 1/8

        h = self.upsampling8(h)
        # print('before out', h.size())

        h = h[:, :, 4:4 + x.size()[2], 4:4 + x.size()[3]].contiguous()
        # print('out', h.size())

        return h


# if __name__ == '__main__':
    # """
    # Test if weights is correct after copy
    # """
    # from torchsummary import summary

    # net1 = models.vgg16(pretrained=True)
    # net1.to('cuda')
    # summary(net, (3, 512, 512))

    # for name, child in net1.named_children():
    #     print(name)
    #
    # for idx, layer in enumerate(net1.features):
    #     print(idx, ':', layer)

    # for i in net1.features:
    #     print(i)

    # net2 = Vgg16FCN8()
    # net2.to('cuda')
    # print(net2.features[0].weight.data)
    # print(net1.features[0].weight.data)
    # import torch
    #
    # print(torch.all(net2.features[0].weight.data.eq(net1.features[0].weight.data)))
    # output:true
    # """
    # Test if Vgg16FCN8 works correctly
    # """
    # import torch

    # net = Vgg16FCN8()
    # net.to("cuda")
    # inputs = torch.randn(1, 3, 512, 512).to(device='cuda')
    # out = net(inputs)
    # print(out.size())
    # from torchsummary import summary
    # summary(net, (3, 512, 512))

    # print(net)
