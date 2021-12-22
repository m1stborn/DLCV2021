import os
import time
import uuid
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from model_p1.configuration import Config
from model_p1.convnet import Convnet, ParametricDist, cosine_similarity
from model_p1.mini_imgae_dataset import *
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
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--ckpt', type=str, help="Model checkpoint path")
    parser.add_argument('--test_run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Path args
    parser.add_argument('--ckpt_path', default='./ckpt/p1', type=str,
                        help="Model checkpoint path")

    parser.add_argument('--train_csv', default='./hw4_data/mini/train.csv', type=str,
                        help="Training images csv file")
    parser.add_argument('--train_data_dir', default='./hw4_data/mini/train', type=str,
                        help="Training images directory")
    parser.add_argument('--output_csv', default='./result/p1/train_output.csv', type=str, help="Output filename")

    parser.add_argument('--val_csv', default='./hw4_data/mini/val.csv', type=str,
                        help="Valid images csv file")
    parser.add_argument('--val_data_dir', default='./hw4_data/mini/val', type=str,
                        help="Valid images directory")
    parser.add_argument('--val_case_csv', default='./hw4_data/mini/val_testcase.csv', type=str,
                        help="Valid images directory")
    # Experiment parameters
    parser.add_argument('--exp2', default='euclidean', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # init constants:
    config = Config()
    args = parse_args()
    uid = str(uuid.uuid1())
    best_epoch = 0
    pre_val_acc = 0.0

    # step 1: prepare dataset
    train_dataset = MiniImageDataset(args.train_csv, args.train_data_dir)

    train_sampler = CategoriesSampler(train_dataset.label, n_batch=config.n_batch,
                                      n_way=config.n_way, n_shot=config.n_shot, n_query=config.n_query)

    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler,
                                  pin_memory=False, num_workers=config.n_worker, worker_init_fn=worker_init_fn)

    total_steps = len(train_dataloader)

    # test_dataset = MiniImageTestDataset('./hw4_data/mini/val.csv', './hw4_data/mini/val')
    val_dataset = MiniImageTestDataset(args.val_csv, args.val_data_dir)

    val_dataloader = DataLoader(val_dataset, batch_size=5 * (15 + 1), num_workers=config.n_worker,
                                pin_memory=False, worker_init_fn=worker_init_fn,
                                sampler=GeneratorSampler(args.val_case_csv))

    # step 2: init network
    net = Convnet()

    # step 3: define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=0.9)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # step 4: check if resume training
    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.ckpt)
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

    distance = euclidean_metric
    config.epochs = 50
    if args.exp2 == 'cosine':
        distance = cosine_similarity
    elif args.exp2 == 'parametric':
        distance = ParametricDist()
        distance.to(device)

    # step 6: main loop
    for epoch in range(start_epoch, start_epoch + config.epochs):
        # lr_scheduler.step()

        # Train
        net.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        for i, batch in enumerate(train_dataloader):
            images, ori_labels = batch[0].to(device), batch[2]
            p = config.n_way * config.n_shot
            images_sup, images_query = images[:p], images[p:]

            label_encoder = {ori_labels[i]: i for i in range(config.n_way)}
            query_label = torch.cuda.LongTensor(
                [label_encoder[class_name] for class_name in ori_labels[p:]])
            sup_label = torch.cuda.LongTensor(
                [label_encoder[class_name] for class_name in ori_labels[:p]])

            proto = net(images_sup)
            # reshape proto to n_shot x n_way x feat_dim and mean
            proto = proto.reshape(config.n_shot, config.n_way, -1).mean(dim=0)  # n_way x feat_dim = 30 x 1600

            logits = euclidean_metric(net(images_query), proto)  # (n_way * n_query) x n_way = 450 x 30
            loss = criterion(logits, query_label)

            # check below
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == query_label).type(torch.cuda.FloatTensor).mean().item()

            train_loss.add(loss.item())
            train_acc.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            prefix = 'Epoch [{}/{}]-'.format(epoch + 1, start_epoch + config.epochs)
            if (i + 1) % 10 == 0:  # print every 10 mini-batches
                suffix = 'Train Loss: {:.4f} Train Acc: {:.4f}'.format(
                    train_loss.item(), train_acc.item())
                progress_bar(i + 1, total_steps, prefix, suffix)

            if args.test_run:
                break

        # Validation
        # TODO: with no grad?
        net.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        for i, batch in enumerate(val_dataloader):
            images, labels = batch[0].to(device), batch[1]
            p = config.val_n_shot * config.val_n_way
            support_input, query_input = images[:p], images[p:]

            label_encoder = {labels[i]: i for i in range(config.val_n_way)}
            query_label = torch.cuda.LongTensor(
                [label_encoder[class_name] for class_name in labels[p:]])
            support_label = torch.cuda.LongTensor(
                [label_encoder[class_name] for class_name in labels[:p]])

            proto = net(support_input)
            proto = proto.reshape(config.val_n_shot, config.val_n_way, -1).mean(dim=0)  # n_way x feat_dim = 30 x 1600

            logits = distance(net(query_input), proto)  # (n_way * n_query) x n_way
            loss = criterion(logits, query_label)

            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == query_label).type(torch.cuda.FloatTensor).mean().item()

            val_loss.add(loss.item())
            val_acc.add(acc)

        print('\nValid Acc: {:.4f}'.format(val_acc.item()))

        if args.test_run:
            break

        # step 6: save checkpoint if better than previous
        if pre_val_acc < val_acc.item():
            checkpoint = {
                'net': net.state_dict(),
                'epoch': epoch,
                'optim': optimizer.state_dict(),
                'uid': uid,
                'acc': val_acc.item()
            }
            save_checkpoint(checkpoint,
                            os.path.join(args.ckpt_path, "ProtoNet-{}-{}.pt".format(args.exp2, uid[:8])))
            print(f'Epoch {epoch + 1} Saved!')
            pre_val_acc = val_acc.item()
            best_epoch = epoch + 1

    # step 7: logging experiment
    if not args.test_run:
        experiment_record_p1('./ckpt/p1/p1_log.txt',
                             uid,
                             time.ctime(),
                             best_epoch,
                             pre_val_acc)
