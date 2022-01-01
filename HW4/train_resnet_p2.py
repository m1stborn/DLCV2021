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
from torchvision import models, transforms

from model_p2.dataset import ImageDataset
from model_p2.configuration import ConfigResnet
from model_p2.resnet50 import Resnet
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
    parser = argparse.ArgumentParser(description="Resnet fine tune")
    parser.add_argument('--ckpt', type=str, help="Pretrained Resnet from BYOL")
    parser.add_argument('--ckpt_resnet', type=str, help="Restore Resnet")
    parser.add_argument('--test_run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no_pretrained', action='store_true',
                        help="NOT to load pretrained resnet")
    parser.add_argument('--no_freeze', action='store_true',
                        help="NOT to load pretrained resnet")
    # Path args
    parser.add_argument('--ckpt_path', default='./ckpt/p2', type=str,
                        help="Model checkpoint path")

    parser.add_argument('--train_csv', default='./hw4_data/office/train.csv', type=str,
                        help="Training images csv file")
    parser.add_argument('--train_data_dir', default='./hw4_data/office/train', type=str,
                        help="Training images directory")

    parser.add_argument('--val_csv', default='./hw4_data/office/val.csv', type=str,
                        help="Valid images csv file")
    parser.add_argument('--val_data_dir', default='./hw4_data/office/val', type=str,
                        help="Valid images directory")

    parser.add_argument('--output_csv', default='./result/p1/train_output.csv', type=str, help="Output filename")

    return parser.parse_args()


if __name__ == '__main__':
    # Init constants:
    config = ConfigResnet()
    args = parse_args()
    uid = str(uuid.uuid1())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    prev_val_acc = 0
    best_epoch = 0

    pretrained = not args.no_pretrained
    freeze = not args.no_freeze
    print(f"Pretrained: {not args.no_pretrained} Freeze: {freeze}")

    # step 1: Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomAutocontrast(p=0.3),
        transforms.ColorJitter(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(args.train_csv, args.train_data_dir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  shuffle=True)

    val_dataset = ImageDataset(args.val_csv, args.val_data_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size,
                                shuffle=False)

    total_steps = len(train_dataloader)

    # step 2: Init network
    pretrained_net = None
    if not args.no_pretrained:
        # Load pretrained net from BYOL checkpoint
        print("Load pretrained resnet.")
        ckpt = load_checkpoint(args.ckpt)
        pretrained_net = models.resnet50(pretrained=False)
        pretrained_net.load_state_dict(ckpt['resnet'])

    net = Resnet(model=pretrained_net, num_classes=65, freeze=freeze)
    net.to(device)

    # For checking if the weight freeze
    param_before = net.features[0].weight[0][0].clone().detach()

    # step 3: Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    # optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=config.lr, momentum=0.9)

    # step 4: check if resume training
    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.ckpt_resnet)
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optim'])
        uid = ckpt['uid']
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    # step 5: Main loop
    for epoch in range(start_epoch, start_epoch + config.epochs):
        # Training
        net.train()
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            images, labels = batch[0].to(device), batch[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            prefix = 'Epoch [{}/{}]-'.format(epoch + 1, start_epoch + config.epochs)
            if (i + 1) % 10 == 0:  # print every 10 mini-batches
                suffix = 'Train Loss: {:.4f} Avg Loss: {:.4f}'.format(
                    loss.item(), running_loss / (i + 1))
                progress_bar(i + 1, total_steps, prefix, suffix)

            if args.test_run and i > 9:
                break

        # Validation
        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            running_val_loss = 0.0
            for batch in val_dataloader:
                images, labels = batch[0].to(device), batch[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)

                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('\nValid ACC: {:.4f} Valid Loss: {:.4f}'
                  .format(correct / total, running_val_loss / len(val_dataloader)))

        if args.test_run:
            break

        # step 6: Save checkpoint if better than previous
        if prev_val_acc < (correct / total):
            checkpoint = {
                'net': net.state_dict(),
                'epoch': epoch,
                'optim': optimizer.state_dict(),
                'uid': uid,
                'acc': (correct / total),
                'pretrained': pretrained,
                'freeze': freeze
            }
            save_checkpoint(checkpoint,
                            os.path.join(args.ckpt_path, "Resnet-{}.pt".
                                         format(uid[:8])))
            print(f'Epoch {epoch + 1} Saved!')
            prev_val_acc = (correct / total)
            best_epoch = epoch + 1

    # step 7: Logging experiment
    if not args.test_run:
        use_backbone = not args.no_pretrained
        experiment_record_p2_resnet('./ckpt/p2/resnet_log.txt',
                                    uid,
                                    time.ctime(),
                                    freeze,
                                    use_backbone,
                                    best_epoch,
                                    prev_val_acc)

    # Assert
    print("Check Freeze:", torch.equal(param_before, net.features[0].weight[0][0]))
