import os
import csv
import time
import uuid
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from model_p2.configuration import ConfigResnet
from model_p2.resnet50 import Resnet
from model_p2.dataset import ImageDataset, ImageTestDataset
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
    parser = argparse.ArgumentParser(description="Few shot learning: Inference")
    parser.add_argument('--ckpt', default='./ckpt/p2/Resnet-86d27ad9.pt', type=str, help="Model checkpoint path")
    # Path args
    parser.add_argument('--test_csv', default='./hw4_data/office/val.csv', type=str,
                        help="Test images csv file")
    parser.add_argument('--test_data_dir', default='./hw4_data/office/val', type=str,
                        help="Test images directory")

    parser.add_argument('--output_csv', default='./result/p2/output.csv', type=str, help="Output filename")

    return parser.parse_args()


if __name__ == '__main__':
    # Init constants:
    args = parse_args()
    config = ConfigResnet()

    ckpt = load_checkpoint(args.ckpt)
    print(f"Ckpt ACC: {ckpt['acc']}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    net = Resnet(num_classes=65, freeze=False)
    net.to(device)
    net.load_state_dict(ckpt['net'])
    net.eval()

    # val_dataset = ImageDataset(args.test_csv, args.test_data_dir, name2label=ckpt['name2label'])
    # val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size,
    #                             shuffle=False)

    test_dataset = ImageTestDataset(args.test_csv, args.test_data_dir, name2label=ckpt['name2label'])
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size,
                                 shuffle=False)

    with torch.no_grad():
        # Validation
        # val_pred = []
        # correct = 0
        # total = 0
        #
        # for batch in val_dataloader:
        #     images, labels = batch[0].to(device), batch[1].to(device)
        #     outputs = net(images)
        #     _, predicted = torch.max(outputs.data, 1)
        #
        #     total += labels.size(0)
        #     correct += (predicted == labels).sum().item()
        #
        #     val_pred.append(predicted)
        # print('Valid ACC: {:.4f} '.format(correct / total))
        #
        # val_pred = torch.cat(val_pred)

        # Testing:
        test_idx = []
        test_filename = []
        test_pred = []
        for batch in test_dataloader:
            images, filenames, indices = batch[0].to(device), batch[1], batch[2]

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            test_idx = test_idx + list(indices.cpu().detach().numpy())
            test_filename = test_filename + list(filenames)
            test_pred.append(predicted)

        label2name = {v: k for k, v in ckpt['name2label'].items()}

        test_pred = torch.cat(test_pred).cpu().numpy()
        test_pred_name = [label2name[label] for label in test_pred]

        with open(args.output_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'filename', 'label'])

            for row in zip(test_idx, test_filename, test_pred_name):
                writer.writerow(row)
