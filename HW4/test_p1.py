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

from model_p1.configuration import Config
from model_p1.convnet import Convnet
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
    parser = argparse.ArgumentParser(description="Few shot learning: Inference")
    parser.add_argument('--ckpt', default='./ckpt/p1/ProtoNet-ee1d3164.pt',type=str, help="Model checkpoint path")
    # Path args
    parser.add_argument('--test_csv', default='./hw4_data/mini/val.csv', type=str,
                        help="Test images csv file")
    parser.add_argument('--test_data_dir', default='./hw4_data/mini/val', type=str,
                        help="Test images directory")
    parser.add_argument('--test_case_csv', default='./hw4_data/mini/val_testcase.csv', type=str,
                        help="Test images directory")
    parser.add_argument('--output_csv', default='./result/p1/output.csv', type=str, help="Output filename")

    return parser.parse_args()


if __name__ == '__main__':
    # Init constants:
    args = parse_args()

    val_n_shot = 1
    val_n_way = 5
    val_n_query = 15

    ckpt = load_checkpoint(args.ckpt)
    print(f"Ckpt ACC: {ckpt['acc']}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    net = Convnet()
    net.to(device)
    net.load_state_dict(ckpt['net'])
    net.eval()
    # Prepare dataset
    test_dataset = MiniImageTestDataset(args.test_csv, args.test_data_dir)

    test_dataloader = DataLoader(test_dataset, batch_size=5 * (15 + 1), num_workers=3,
                                 pin_memory=False, worker_init_fn=worker_init_fn,
                                 sampler=GeneratorSampler(args.test_case_csv))
    test_pred = []
    for i, batch in enumerate(test_dataloader):
        images, labels = batch[0].to(device), batch[1]
        p = val_n_shot * val_n_way
        support_input, query_input = images[:p], images[p:]

        label_encoder = {labels[i * 1]: i for i in range(p)}
        query_label = torch.cuda.LongTensor(
            [label_encoder[class_name] for class_name in labels[p:]])
        support_label = torch.cuda.LongTensor(
            [label_encoder[class_name] for class_name in labels[:p]])

        proto = net(support_input)
        proto = proto.reshape(val_n_shot, val_n_way, -1).mean(dim=0)  # n_way x feat_dim = 30 x 1600

        logits = euclidean_metric(net(query_input), proto)  # (n_way * n_query) x n_way
        _, predicted = torch.max(logits.data, 1)

        test_pred.append(predicted)

    header = "episode_id,query0,query1,query2,query3,query4,query5,query6,query7,query8,query9,query10,query11," \
             "query12,query13,query14,query15,query16,query17,query18,query19,query20,query21,query22,query23," \
             "query24,query25,query26,query27,query28,query29,query30,query31,query32,query33,query34,query35," \
             "query36,query37,query38,query39,query40,query41,query42,query43,query44,query45,query46,query47," \
             "query48,query49,query50,query51,query52,query53,query54,query55,query56,query57,query58,query59," \
             "query60,query61,query62,query63,query64,query65,query66,query67,query68,query69,query70,query71," \
             "query72,query73,query74 "
    header_list = header.split(',')

    with open(args.output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header_list)
        for i, row in enumerate(test_pred):
            writer.writerow([i]+row.tolist())
