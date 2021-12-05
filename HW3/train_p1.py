import os
import time
import uuid
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_pretrained_vit import ViT

from model_p1.dogcat_dataset import DogCatDataset
from parse_config import create_parser
from utils import save_checkpoint, load_checkpoint, progress_bar, experiment_record_p1

torch.manual_seed(1)
torch.cuda.manual_seed(1)

if __name__ == '__main__':
    # init constants:
    parser = create_parser()
    configs = parser.parse_args()

    uid = str(uuid.uuid1())
    best_epoch = 0
    pre_val_acc = 0.0

    # step 1: prepare dataset
    train_dataset = DogCatDataset('./hw3_data/p1_data/train')
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size,
                                  shuffle=True)

    val_dataset = DogCatDataset('./hw3_data/p1_data/val')
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size,
                                shuffle=False)

    total_steps = len(train_dataloader)

    # step 2: init network
    net = ViT('B_16_imagenet1k', pretrained=True, num_classes=37, num_heads=8, num_layers=6)
    net.to("cuda")

    # step 3: define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=configs.lr, momentum=0.9)

    # step 4: check if resume training
    start_epoch = 0
    if configs.resume:
        ckpt = load_checkpoint(configs.ckpt)
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optim'])
        uid = ckpt['uid']
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        print("Checkpoint restored, start from epoch {}.".format(start_epoch + 1))

    # step 5: move Net to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    net.to(device)

    # step 6: main loop
    for epoch in range(start_epoch, start_epoch + configs.epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            prefix = 'Epoch [{}/{}]-'.format(epoch + 1, start_epoch + configs.epochs)
            if (i + 1) % 10 == 0:  # print every 10 mini-batches
                suffix = 'Train Loss: {:.4f}'.format(running_loss / (i + 1))
                progress_bar(i + 1, total_steps, prefix, suffix)
            if configs.test_run:
                break

        # print Valid Accuracy per epoch
        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for val_data in val_dataloader:
                images, labels = val_data[0].to(device), val_data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('\nValid ACC: {:.4f}'
                  .format(correct / total))

        # step 6: save checkpoint if better than previous
        if pre_val_acc < (correct / total):
            checkpoint = {
                'net': net.state_dict(),
                'epoch': epoch,
                'optim': optimizer.state_dict(),
                'uid': uid,
                'acc': (correct / total)
            }
            save_checkpoint(checkpoint,
                            os.path.join(configs.ckpt_path, "ViT-{}.pt".format(uid[:8])))
            print(f'Epoch {epoch + 1} Saved!')
            pre_val_acc = correct / total
            best_epoch = epoch + 1

    # step 7: logging experiment
    experiment_record_p1('./ckpt/p1/p1_log.txt',
                         uid,
                         time.ctime(),
                         configs.batch_size,
                         configs.lr,
                         best_epoch,
                         pre_val_acc)

