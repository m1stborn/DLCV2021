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

from digit_classifier import Classifier, load_classifier_checkpoint
from model_p2.digit_dataset import DigitDataset
from model_p2.acgan import Generator, Discriminator, weights_init
from parse_config import create_parser
from utils import save_checkpoint, load_checkpoint, progress_bar, experiment_record_p2, sample_idx

# step 0: fix random seed for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

if __name__ == '__main__':
    # init constants:
    parser = create_parser()
    configs = parser.parse_args()

    uid = str(uuid.uuid1())
    best_epoch = 0
    latent_size = 100
    num_classes = 10
    prev_acc = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # - Create batch of latent vectors with one hot classes that we will use to visualize
    # - the progression of the generator
    # fixed_noise = torch.randn(1000, 100, 1, 1, device=device)
    fixed_noise = []
    fixed_noise_label = []
    size_per_classes = 100
    for i in range(10):
        label = np.full((size_per_classes,), i)
        # label = np.random.randint(0, num_classes, configs.batch_size)
        noise = np.random.normal(0, 1, (size_per_classes, latent_size))
        label_onehot = np.zeros((size_per_classes, num_classes))
        label_onehot[np.arange(size_per_classes), label] = 1
        noise[:, :num_classes] = label_onehot
        noise = torch.from_numpy(noise).float()
        noise = noise.view(size_per_classes, latent_size, 1, 1)

        fixed_noise.append(noise)
        fixed_noise_label.append(torch.from_numpy(label))
    fixed_noise = torch.cat(fixed_noise, 0).to(device)
    fixed_noise_label = torch.cat(fixed_noise_label, 0).to(device)

    # step 1: prepare dataset
    train_dataset = DigitDataset(configs.p2_input_csv, configs.p2_input_dir)

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
    s_criterion = nn.BCELoss()
    c_criterion = nn.NLLLoss()

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
        prev_acc = ckpt['acc']

        for state in optimizerG.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        for state in optimizerD.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        print("Checkpoint restored, start from epoch {}.".format(start_epoch + 1))

    # step 5: load digit classifier for model evaluation
    netDigit = Classifier()
    load_classifier_checkpoint("Classifier.pth", netDigit)
    netDigit.to(device)
    netDigit.eval()

    # step 6: main loop
    #  Lists to keep track of progress
    # TODO: modify iters for resume training
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
            real_img, c_label = data[0].to(device), data[1].to(device)

            s_label = torch.full((configs.batch_size,), real_label, dtype=torch.float, device=device)
            s_output, c_output = netD(real_img)

            s_errD_real = s_criterion(s_output, s_label)
            c_errD_real = c_criterion(c_output, c_label)
            errD_real = s_errD_real + c_errD_real
            errD_real.backward()

            D_x = s_output.mean().item()

            # - Train with all-fake batch
            label = np.random.randint(0, num_classes, configs.batch_size)
            noise = np.random.normal(0, 1, (configs.batch_size, latent_size))
            label_onehot = np.zeros((configs.batch_size, num_classes))
            label_onehot[np.arange(configs.batch_size), label] = 1
            noise[np.arange(configs.batch_size), :num_classes] = label_onehot[np.arange(configs.batch_size)]
            noise = (torch.from_numpy(noise).to(device).float())
            noise = noise.view(configs.batch_size, latent_size, 1, 1)
            c_label.data.resize_(configs.batch_size).copy_(torch.from_numpy(label))

            fake = netG(noise)
            s_label.fill_(fake_label)

            s_output, c_output = netD(fake.detach())

            s_errD_fake = s_criterion(s_output, s_label)
            c_errD_fake = c_criterion(c_output, c_label)
            errD_fake = s_errD_fake + c_errD_fake
            errD_fake.backward()

            D_G_z1 = s_output.mean().item()
            # TODO:  errD = s_errD_real + s_errD_fake instead?
            errD = errD_real + errD_fake
            # errD = s_errD_real + s_errD_real

            optimizerD.step()

            # 2. Train Generator: maximize log(D(G(z)))
            netG.zero_grad()

            s_label.fill_(real_label)
            s_output, c_output = netD(fake)

            s_errG = s_criterion(s_output, s_label)
            c_errG = c_criterion(c_output, c_label)
            errG = s_errG + c_errG
            errG.backward()
            D_G_z2 = s_output.data.mean()

            optimizerG.step()

            # print statistics
            iters += 1

            prefix = 'Epoch [{}/{}]-'.format(epoch + 1, start_epoch + configs.epochs)
            if (i + 1) % 50 == 0:  # print every 50 mini-batches
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

                    idx = sample_idx()

                    fake = netG(fixed_noise[idx]).detach().cpu()
                    # plt.figure(figsize=(10, 10))
                    plt.axis("off")
                    plt.title("Training Process Epoch:{} Iteration: {}".format(epoch, iters))
                    plt.imshow(
                        np.transpose(make_grid(fake.to("cuda"), padding=2, normalize=True, nrow=10).cpu(), (1, 2, 0))
                    )
                    plt.savefig(os.path.join('./p2_result/first_100_progress', "{}-Epoch_{}-Iters_{}.png".format(
                        uid[:8], epoch + 1, iters))
                                )

        # print Acc score per epoch
        netG.eval()
        with torch.no_grad():
            # generate fake image at once
            fake_img = netG(fixed_noise)
            outputs = netDigit(fake_img)
            _, predicted = torch.max(outputs.data, 1)

            # idx = sample_idx()
            # print("\n", predicted[idx], fixed_noise_label[idx], idx)

            correct = (predicted == fixed_noise_label).sum().item()

            # save fake image one by one
            counter = [0 for i in range(10)]

            for i, (img, label) in enumerate(zip(fake_img, fixed_noise_label)):
                label = label.item()
                counter[label] += 1

                img = img.squeeze(0).add(1).mul(255 * 0.5)
                img = img.cpu().numpy()
                img = np.transpose(img, (1, 2, 0)).astype(np.uint8)

                filename = os.path.join(configs.p2_output_temp, f"{label}_{str(counter[label]).zfill(3)}.png")
                skimage.io.imsave(filename, img, check_contrast=False)

            acc = correct / 1000
            print('\nACC on 1000 image: {:.4f}'.format(acc))

            if acc > prev_acc:
                checkpoint = {
                    'netD': netD.state_dict(),
                    'netG': netG.state_dict(),
                    'epoch': epoch,
                    'optimD': optimizerD.state_dict(),
                    'optimG': optimizerG.state_dict(),
                    'uid': uid,
                    'acc': acc,
                    'noise': fixed_noise,
                    'noise_label': fixed_noise_label,
                    'Gloss': G_losses,
                    'Dloss': D_losses,
                }

                save_checkpoint(checkpoint,
                                os.path.join(configs.ckpt_path, "ACGAN-{}.pt".format(uid[:8])))
                print(f'Epoch {epoch + 1} Saved!')
                prev_acc = acc

                best_epoch = epoch + 1

    experiment_record_p2("./ckpt/p2/p2_log.txt",
                         uid,
                         time.ctime(),
                         configs.batch_size,
                         configs.lr,
                         best_epoch,
                         prev_acc)
