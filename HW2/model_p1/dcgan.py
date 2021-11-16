import numpy as np
import skimage.io
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
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self,
                 nc=3,
                 ndf=64
                 ):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)


if __name__ == '__main__':
    from torchsummary import summary

    netG = Generator()
    netG.to("cuda")
    netG.apply(weights_init)
    summary(netG, (100, 1, 1))

    z = torch.randn(1, 100, 1, 1, device="cuda")
    print(z.size())
    print(netG(z).size())
    # out = netG(z).cpu().detach()
    # out_denorm = out.add(1).mul(255).mul(0.5)
    # print(torch.max(out), torch.min(out))
    # print(torch.max(out_denorm), torch.min(out_denorm))

    netD = Discriminator()
    netD.to("cuda")
    netD.apply(weights_init)
    summary(netD, (3, 64, 64))

    x = torch.randn(1, 3, 64, 64, device="cuda")
    print(x.size())
    print(netD(x).size())  # torch.Size([1])
    # print(netD(x).view(-1).size())

    noise = torch.randn(1000, 100, 1, 1, device="cuda")
    # for i in range(1000):
    #     img = netG(noise[i].unsqueeze(0))
    #     img = img.squeeze(0).add(1).mul(255).mul(0.5)
    #     img = img.cpu().detach().numpy()
    #     img = np.transpose(img, (1, 2, 0))
    #     img = img.astype(np.uint8)
    #     skimage.io.imsave("./test/{}.png".format(i), img, check_contrast=False)
    #
