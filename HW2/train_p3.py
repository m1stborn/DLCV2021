import os
import time
import uuid
import skimage.io
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import cycle

from model_p3.dann import DANN
from model_p3.digit_dataset import DigitDataset
from parse_config import create_parser
from utils import save_checkpoint, load_checkpoint, progress_bar, experiment_record_p2, eval_net

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
    src_dataloader = torch.utils.data.DataLoader(src_dataset, batch_size=configs.batch_size,
                                                 shuffle=True)

    tgt_dataset = DigitDataset(tgt_csv, tgt_dir)
    tgt_dataloader = torch.utils.data.DataLoader(tgt_dataset, batch_size=configs.batch_size,
                                                 shuffle=True)

    total_steps = max(len(src_dataloader), len(tgt_dataloader))
    print(f'src len:{len(src_dataloader)}, tge len:{len(tgt_dataloader)}')

    test_dataset = DigitDataset(test_csv, test_dir)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=configs.batch_size,
                                                  shuffle=True)
    test_src_dataset = DigitDataset(test_src_csv, test_src_dir)
    test_src_dataloader = torch.utils.data.DataLoader(test_src_dataset, batch_size=configs.batch_size,
                                                      shuffle=True)
    # step 2: init network
    net = DANN()
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
        tgt_iter = iter(tgt_dataloader)
        step_count = 0
        for i, src_data in enumerate(src_dataloader):
            try:
                tgt_data = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_dataloader)
                tgt_data = next(tgt_iter)

            net.zero_grad()
            # setup hyper parameters
            p = float(i + epoch * total_steps) / \
                configs.epochs / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # prepare data
            img_src, class_label_src = src_data[0].to(device), src_data[1].to(device)
            img_tgt = tgt_data[0].to(device)

            # prepare domain label
            size_src = len(img_src)
            size_tgt = len(img_tgt)
            domain_label_src = torch.zeros(size_src).long().to(device)  # source 0
            domain_label_tgt = torch.ones(size_tgt).long().to(device)  # target 1

            # zero gradients for optimizer
            optimizer.zero_grad()

            # train on source domain
            src_class_output, src_domain_output = net(img_src, alpha=alpha)
            src_loss_class = criterion(src_class_output, class_label_src)
            src_loss_domain = criterion(src_domain_output, domain_label_src)

            # train on target domain
            _, tgt_domain_output = net(img_tgt, alpha=alpha)
            tgt_loss_domain = criterion(tgt_domain_output, domain_label_tgt)

            # TODO: domain loss theta
            loss = src_loss_class + src_loss_domain + tgt_loss_domain

            # backward + optimize
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            prefix = 'Epoch [{}/{}]-'.format(epoch + 1, start_epoch + configs.epochs)
            if (i + 1) % 10 == 0:  # print every 10 mini-batches
                suffix = 'Train Loss: {:.4f} SrcClassLoss: {:.4f} SrcDomainLoss: {:.4f} TgtDomainLoss: {:.4f}'.format(
                    running_loss / (i + 1), src_loss_class, src_loss_domain, tgt_loss_domain)
                progress_bar(i + 1, total_steps, prefix, suffix)

            if configs.test_run:
                break

        # eval model
        train_acc, train_acc_domain = eval_net(net, tgt_dataloader, device, "target")
        tgt_acc, tgt_acc_domain = eval_net(net, test_dataloader, device, "target")
        src_acc, src_acc_domain = eval_net(net, test_src_dataloader, device, "source")
        print('\nTrainAcc: {:.4f} TrainDomainAcc: {:.4f}'.format(train_acc, train_acc_domain))
        print('TgtAcc: {:.4f} SrcAcc: {:.4f} TgtDomainAcc: {:.4f} SrcDomainAcc: {:.4f}'
              .format(tgt_acc, src_acc, tgt_acc_domain, src_acc_domain))

        # step 6: save checkpoint if better than previous
        if tgt_acc > prev_acc:
            checkpoint = {
                'net': net.state_dict(),
                'epoch': epoch,
                'optim': optimizer.state_dict(),
                'uid': uid,
                'tgtacc': tgt_acc,
                'srcacc': src_acc,
            }
            save_checkpoint(checkpoint,
                            os.path.join(configs.ckpt_path, f"{configs.src_mode}-{uid[:8]}.pt"))
            print(f'Epoch {epoch + 1} Saved!')
            prev_acc = tgt_acc
            best_epoch = epoch + 1

            step_count += 1

    # step 7: logging experiment
    experiment_record_p2("./ckpt/p3/p3_log.txt",
                         uid,
                         time.ctime(),
                         configs.batch_size,
                         configs.lr,
                         best_epoch,
                         prev_acc)
