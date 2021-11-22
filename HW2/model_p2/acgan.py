import torch
import torch.nn as nn


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self,
                 nz=100,  # Size of z latent vector (i.e. size of generator input)
                 ngf=64,  # Size of feature maps in generator
                 nc=3,  # Number of channels in the training images. For color images this is 3
                 ):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # TODO: 1x1 convolution instead?
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 1, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, x):
        output = self.main(x)
        # print(output.size())
        output = output[:, :, 2:2+28, 2:2+28].contiguous()
        return output


class Discriminator(nn.Module):
    def __init__(self,
                 nc=3,
                 ndf=64,
                 num_classes=10,
                 ):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64 => 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32 => 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16 => 7

            # TODO: remove one layer?
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8 => 7
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        self.adv = nn.Sequential(
            nn.Linear(ndf * 8 * 16, 1),
            nn.Sigmoid()
        )
        #
        self.aux = nn.Sequential(
            nn.Linear(ndf * 8 * 16, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.main(x)
        output = output.view(output.shape[0], -1)
        # output = torch.flatten(output, 1)
        validity = self.adv(output)
        label = self.aux(output)

        return validity.view(-1, 1).squeeze(1), label


# def sample_noise_with_class(batch_size)


# TODO:remove for submission


if __name__ == '__main__':
    from torchsummary import summary
    netD = Discriminator()
    # netD.to("cuda")
    # netD.apply(weights_init)
    # summary(netD, (3, 28, 28))
    # x = torch.randn((10, 3, 28, 28), device="cuda")
    # out1, out2 = netD(x)
    # print(out1.size(), out2.size())

    netG = Generator()
    # netG.to("cuda")
    # netG.apply(weights_init)
    # summary(netG, (100, 1, 1))
    # x = torch.randn((10, 100, 1, 1), device="cuda")
    # out = netG(x)
    # print(out.size())
    # import numpy as np
    # nb_label = 10
    # batch_size = 16
    # nz = 100
    # label = np.random.randint(0, nb_label, batch_size)
    # noise_ = np.random.normal(0, 1, (batch_size, nz))
    # label_onehot = np.zeros((batch_size, nb_label))
    # label_onehot[np.arange(batch_size), label] = 1
    # # noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]
    # noise_[:, :nb_label] = label_onehot
    # noise_ = torch.from_numpy(noise_)
    # y = noise_.view(batch_size, nz, 1, 1)
    # print(y.view(batch_size, nz)[0])


    # fake_noise = torch.randn((1000, 100, 1, 1), device="cuda")
    # fake_img = netG(fake_noise)
    # print(fake_img.size())
    #
    # lst = [i for i in range(1000)]
    # for i in range(0, 1000, 100):
    #     # print(i, i+100)
    #     print(lst[i:i+100])
    print(netG)
    print(netD)
