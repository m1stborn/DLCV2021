import os
import time
import uuid
import skimage.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from model_p1.face_dataset import FaceDataset
from model_p1.dcgan import Generator, Discriminator, weights_init
from parse_config import create_parser
from utils import save_checkpoint, load_checkpoint, progress_bar, experiment_record_p1, calculate_is_score

from pytorch_fid.fid_score import calculate_fid_given_paths

# step 0: fix random seed for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if __name__ == '__main__':
    # init constants:
    parser = create_parser()
    configs = parser.parse_args()

    uid = str(uuid.uuid1())
    best_epoch = 0
    latent_size = 100
    prev_fid = 9223372036854775807
    prev_is_mean = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # - Create batch of latent vectors that we will use to visualize
    # - the progression of the generator
    fixed_noise = torch.randn(1000, 100, 1, 1, device=device)

    # step 1: prepare dataset
    train_dataset = FaceDataset(configs.p1_train_dir)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=configs.batch_size,
                                             shuffle=True)
    total_steps = len(dataloader)

    # step 2: init network
    netG = Generator()
    netD = Discriminator()

    netG.to(device)
    netD.to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # step 3: define loss function and optimizer
    criterion = nn.BCELoss()

    # - Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    optimizerD = torch.optim.Adam(netD.parameters(), lr=configs.lr, betas=(configs.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=configs.lr, betas=(configs.beta1, 0.999))

    # step 4: check if resume training
    start_epoch = 0
    if configs.resume:
        ckpt = load_checkpoint(configs.ckpt)
        netG.load_state_dict(ckpt['netG'])
        netD.load_state_dict(ckpt['netD'])
        start_epoch = ckpt['epoch'] + 1
        optimizerG.load_state_dict(ckpt['optimG'])
        optimizerD.load_state_dict(ckpt['optimD'])
        uid = ckpt['uid']
        prev_is_mean = ckpt['fid']
        prev_fid = ckpt['is']

        for state in optimizerG.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        for state in optimizerD.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        print("Checkpoint restored, start from epoch {}.".format(start_epoch + 1))

    # step 5: main loop
    #  Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    for epoch in range(start_epoch, start_epoch + configs.epochs):
        netD.train()
        netG.train()
        for i, data in enumerate(dataloader):

            # 1. Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            # - Train with all-real batch
            real_image = data.to(device)
            label = torch.full((configs.batch_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_image)
            # - Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # - Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # - Train with all-fake batch
            # - Generate batch of latent vectors
            noise = torch.randn(configs.batch_size, latent_size, 1, 1, device=device)
            # - Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # - Classify all fake batch with D
            output = netD(fake.detach())
            # - Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # - Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # - Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # - Update D
            optimizerD.step()

            # 2. Train Generator: maximize log(D(G(z)))
            netG.zero_grad()
            # - GAN hacks: flip label to
            label.fill_(real_label)

            output = netD(fake)

            errG = criterion(output, label)
            # - Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # - Update G
            optimizerG.step()

            # print statistics
            iters += 1
            prefix = 'Epoch [{}/{}]-'.format(epoch + 1, start_epoch + configs.epochs)
            if (i + 1) % 50 == 0:  # print every 5 mini-batches
                suffix = 'Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f} '.format(
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)

                progress_bar(i + 1, total_steps, prefix, suffix)
            if configs.test_run:
                break

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # save training process per 500 iterations
            if iters % 500 == 0:
                with torch.no_grad():
                    fake = netG(fixed_noise[:32]).detach().cpu()
                    plt.figure(figsize=(8, 8))
                    plt.axis("off")
                    plt.title("Training Process Epoch:{} Iteration: {}".format(epoch, iters))
                    plt.imshow(
                        np.transpose(make_grid(fake.to("cuda"), padding=2, normalize=True).cpu(), (1, 2, 0))
                    )
                    plt.savefig(os.path.join('./p1_result/first_32_progress', "{}-Epoch_{}-Iters_{}.png".format(
                        uid[:8], epoch+1, iters))
                    )

        # print IS score per epoch
        netG.eval()
        with torch.no_grad():

            for i in range(1000):
                img = netG(fixed_noise[i].unsqueeze(0))
                img = img.squeeze(0).add(1).mul(255*0.5)
                img = img.cpu().numpy()
                img = np.transpose(img, (1, 2, 0)).astype(np.uint8)

                filename = os.path.join(configs.p1_output_temp, f'{str(i+1).zfill(4)}.png')
                skimage.io.imsave(filename, img, check_contrast=False)

            is_score_mean, is_score_std = calculate_is_score(configs.p1_output_temp)

            path2 = './hw2_data/face/test'
            fid = calculate_fid_given_paths([configs.p1_output_temp, path2], 50, 'cuda', 2048, 2)

            print('\nIS: {} FID: {}'.format(is_score_mean, fid))

            # save checkpoint
            if is_score_mean > 2 and fid < prev_fid:
                checkpoint = {
                    'netD': netD.state_dict(),
                    'netG': netG.state_dict(),
                    'epoch': epoch,
                    'optimD': optimizerD.state_dict(),
                    'optimG': optimizerG.state_dict(),
                    'uid': uid,
                    'is_mean': is_score_mean,
                    'is_std': is_score_std,
                    'fid': fid,
                    'noise': fixed_noise,
                    'Gloss': G_losses,
                    'Dloss': D_losses,
                }

                save_checkpoint(checkpoint,
                                os.path.join(configs.ckpt_path, "DCGAN-{}.pt".format(uid[:8])))
                print(f'Epoch {epoch+1} Saved!')
                prev_is_mean = is_score_mean
                prev_fid = fid
                best_epoch = epoch + 1

    experiment_record_p1("./ckpt/p1/p1_log.txt",
                         uid, time.ctime(),
                         configs.batch_size,
                         configs.lr,
                         best_epoch,
                         is_score_mean,
                         fid)
