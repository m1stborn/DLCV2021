import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image

filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# mini-Imagenet Test/Valid dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            # filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = Image.open(os.path.join(self.data_dir, path))
        # image = self.transform(os.path.join(self.data_dir, path))
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data_df)


class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence)

    def __len__(self):
        return len(self.sampled_sequence)


def predict(args, model, data_loader):
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot, :, :, :]
            query_input = data[args.N_way * args.N_shot:, :, :, :]

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.N_shot]: i for i in range(args.N_way)}
            query_label = torch.cuda.LongTensor(
                [label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])

            # TODO: extract the feature of support and query data

            # TODO: calculate the prototype for each class according to its support data

            # TODO: classify the query data depending on the its distance with each prototype

    # return prediction_results
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')  # number of image per way
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', default='./hw4_data/mini/val.csv', type=str,
                        help="Testing images csv file")
    parser.add_argument('--test_data_dir', default='./hw4_data/mini/val', type=str,
                        help="Testing images directory")
    parser.add_argument('--testcase_csv', default='./hw4_data/mini/val_testcase.csv', type=str,
                        help="Test case csv")
    parser.add_argument('--output_csv', default='./result/p1/output.csv', type=str, help="Output filename")
    # batch size = N_way * (N_query + N_shot) = 5 * (15 + 1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    # Dataloader with multi worker
    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # Dataloader with single worker
    #     # test_loader = DataLoader(
    #     #     test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
    #     #     num_workers=3, pin_memory=False,
    #     #     sampler=GeneratorSampler(args.testcase_csv))
    #
    #     # batch1 = next(iter(test_loader))
    #     # print(batch1[0].size(), len(batch1[1]))
    #     # print(set(batch1[1]))
    for i, data in enumerate(test_loader):
        if i == 0:
            images, labels = data[0].to('cuda'), data[1]
            print(images.size())
            print(labels)
            # print(images)
            # print(labels)
            break

    # TODO: load your model

    # prediction_results = predict(args, model, test_loader)

    # TODO: output your prediction to csv
