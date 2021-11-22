import os
import time
import uuid
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from itertools import cycle

from model_p3.dann import DigitClassifier
from model_p3.digit_dataset import DigitDataset
from parse_config import create_parser
from utils import save_checkpoint, load_checkpoint, progress_bar, experiment_record_p2, eval_src_net

# step 0: fix random seed for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if __name__ == '__main__':
    # init constants:
    parser = create_parser()
    configs = parser.parse_args()

    uid = str(uuid.uuid1())
    best_epoch = 0
    num_classes = 10
    prev_acc = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # step 1: prepare dataset
    # MNIST-M → USPS / SVHN → MNIST-M / USPS → SVHN
    mnist_dir = './hw2_data/digits/mnistm/train'
    mnist_csv = './hw2_data/digits/mnistm/train.csv'
    svhn_dir = './hw2_data/digits/svhn/train'
    svhn_csv = './hw2_data/digits/svhn/train.csv'
    usps_dir = './hw2_data/digits/usps/train'
    usps_csv = './hw2_data/digits/usps/train.csv'

    mnist_test_dir = './hw2_data/digits/mnistm/test'
    mnist_test_csv = './hw2_data/digits/mnistm/test.csv'
    svhn_test_dir = './hw2_data/digits/svhn/test'
    svhn_test_csv = './hw2_data/digits/svhn/test.csv'
    usps_test_dir = './hw2_data/digits/usps/test'
    usps_test_csv = './hw2_data/digits/usps/test.csv'

    src_dir = mnist_dir
    src_csv = mnist_csv
    test_src_dir = mnist_test_dir
    test_src_csv = mnist_test_csv
    tgt_dir = usps_dir
    tgt_csv = usps_csv
    test_dir = usps_test_dir
    test_csv = usps_test_csv
    if configs.src_mode == "svhn":
        src_dir = svhn_dir
        src_csv = svhn_csv
        test_src_dir = svhn_test_dir
        test_src_csv = svhn_test_csv
        tgt_dir = mnist_dir
        tgt_csv = mnist_csv
        test_dir = mnist_test_dir
        test_csv = mnist_test_csv
    elif configs.src_mode == "usps":
        src_dir = usps_dir
        src_csv = usps_csv
        test_src_dir = usps_test_dir
        test_src_csv = usps_test_csv
        tgt_dir = svhn_dir
        tgt_csv = svhn_csv
        test_dir = svhn_test_dir
        test_csv = svhn_test_csv

    src_dataset = DigitDataset(src_csv, src_dir)
    if configs.src_mode == "usps":
        src_dataset = DigitDataset(src_csv, src_dir, transform=transform)
    src_dataloader = torch.utils.data.DataLoader(src_dataset, batch_size=configs.batch_size,
                                                 shuffle=True)

    test_tgt_dataset = DigitDataset(test_csv, test_dir)
    test_tgt_dataloader = torch.utils.data.DataLoader(test_tgt_dataset, batch_size=configs.batch_size,
                                                      shuffle=True)

    test_src_dataset = DigitDataset(test_src_csv, test_src_dir)
    test_src_dataloader = torch.utils.data.DataLoader(test_src_dataset, batch_size=configs.batch_size,
                                                      shuffle=True)

    total_steps = len(src_dataloader)
    print(f"Total steps: {total_steps}, Src Test Steps:{len(test_src_dataloader)}")
    # step 2: init network
    net = DigitClassifier()
    net.to(device)

    # step 3: define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # lr = 0.01
    optimizer = torch.optim.SGD(net.parameters(), lr=configs.lr, momentum=0.9)

    # step 4: check if resume training
    start_epoch = 0
    if configs.resume:
        ckpt = load_checkpoint(configs.ckpt)
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optim'])
        uid = ckpt['uid']
        prev_acc = ckpt['tgtacc']

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        print("Checkpoint restored, start from epoch {}.".format(start_epoch + 1))

    # step 5: main loop
    for epoch in range(start_epoch, start_epoch + configs.epochs):
        net.train()
        running_loss = 0.0
        # TODO: cycle shorter dataloader
        src_iter = iter(src_dataloader)

        for i, src_data in enumerate(src_dataloader):
            img_src, class_label_src = src_data[0].to(device), src_data[1].to(device)

            # zero gradients for optimizer
            net.zero_grad()
            optimizer.zero_grad()

            # train on source domain
            outputs = net(img_src)
            loss = criterion(outputs, class_label_src)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            prefix = 'Epoch [{}/{}]-'.format(epoch + 1, start_epoch + configs.epochs)
            if (i + 1) % 10 == 0:  # print every 10 mini-batches
                suffix = 'Train Loss: {:.4f} '.format(
                    running_loss / (i + 1))
                progress_bar(i + 1, total_steps, prefix, suffix)

            if configs.test_run:
                break

        # eval model
        src_acc = eval_src_net(net, test_src_dataloader, device)
        tgt_acc = eval_src_net(net, test_tgt_dataloader, device)
        print('\nSrcAcc: {:.4f} TgtAcc: {:.4f}'.format(src_acc, tgt_acc))

        # step 6: save checkpoint if better than previous
        if src_acc > prev_acc:
            checkpoint = {
                'net': net.state_dict(),
                'epoch': epoch,
                'optim': optimizer.state_dict(),
                'uid': uid,
                'tgtacc': tgt_acc,
                'srcacc': src_acc,
            }
            save_checkpoint(checkpoint,
                            os.path.join(configs.ckpt_path, f"{configs.src_mode}_only-{uid[:8]}.pt"))
            print(f'Epoch {epoch + 1} Saved!')
            prev_acc = src_acc
            best_epoch = epoch + 1

    # step 7: logging experiment
    experiment_record_p2("./ckpt/p3/p3_log.txt",
                         uid,
                         time.ctime(),
                         configs.batch_size,
                         configs.lr,
                         best_epoch,
                         prev_acc)
