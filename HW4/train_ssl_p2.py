import os
import time
import uuid
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from byol_pytorch import BYOL
from torchvision import models

from model_p2.dataset import ImageDataset
from model_p2.configuration import Config
from utils import *

# fix random seeds for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="SSL learning")
    parser.add_argument('--ckpt', type=str, help="Model checkpoint path")
    parser.add_argument('--test_run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Path args
    parser.add_argument('--ckpt_path', default='./ckpt/p2', type=str,
                        help="Model checkpoint path")

    parser.add_argument('--train_csv', default='./hw4_data/mini/train.csv', type=str,
                        help="Training images csv file")
    parser.add_argument('--train_data_dir', default='./hw4_data/mini/train', type=str,
                        help="Training images directory")
    parser.add_argument('--output_csv', default='./result/p1/train_output.csv', type=str, help="Output filename")

    return parser.parse_args()


if __name__ == '__main__':
    # Init constants:
    config = Config()
    args = parse_args()
    uid = str(uuid.uuid1())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    prev_loss = 100
    trlog = dict()
    trlog['tr_loss'] = []

    # step 1: Prepare dataset
    train_dataset = ImageDataset(args.train_csv, args.train_data_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  shuffle=True)
    total_steps = len(train_dataloader)
    # step 2: Init network
    resnet = models.resnet50(pretrained=False)

    net = BYOL(
        resnet,
        image_size=128,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
    )

    resnet.to(device)
    net.to(device)

    # step 3: Define loss function and optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    # step 4: check if resume training
    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.ckpt)
        net.load_state_dict(ckpt['net'])
        resnet.load_state_dict(ckpt['resnet'])
        start_epoch = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optim'])
        uid = ckpt['uid']
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        print("Checkpoint restored, start from epoch {}.".format(start_epoch + 1))

    # step 5: Main loop
    for epoch in range(start_epoch, start_epoch + config.epochs):
        net.train()
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            images = batch[0].to(device)  # Label is not needed

            loss = net(images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.update_moving_average()

            # print statistics
            running_loss += loss.item()
            prefix = 'Epoch [{}/{}]-'.format(epoch + 1, start_epoch + config.epochs)
            if (i + 1) % 10 == 0:  # print every 10 mini-batches
                suffix = 'Train Loss: {:.4f} Avg Loss: {:.4f}'.format(
                    loss.item(), running_loss / (i + 1))
                progress_bar(i + 1, total_steps, prefix, suffix)

            if args.test_run and i > 9:
                break

        cur_loss = running_loss / len(train_dataloader)
        trlog['tr_loss'].append(cur_loss)

        # step 6: Save checkpoint
        # Save best checkpoint
        if prev_loss > cur_loss:
            checkpoint = {
                'net': net.state_dict(),
                'epoch': epoch,
                'optim': optimizer.state_dict(),
                'uid': uid,
                'resnet': resnet.state_dict(),
                'trlog': trlog
            }
            save_checkpoint(checkpoint,
                            os.path.join(args.ckpt_path, "BYOL-{}-best.pt".format(uid[:8])))
            print(f'\nEpoch {epoch + 1} Saved!')
            prev_loss = cur_loss
        # Save current checkpoint
        checkpoint = {
            'net': net.state_dict(),
            'epoch': epoch,
            'optim': optimizer.state_dict(),
            'uid': uid,
            'resnet': resnet.state_dict(),
            'trlog': trlog
        }
        save_checkpoint(checkpoint,
                        os.path.join(args.ckpt_path, "BYOL-{}.pt".format(uid[:8])))
        if args.test_run:
            break
    # step 5: Finetune loop
