import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import Sampler

# fix random seeds for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# mini-Imagenet dataset
class MiniImageDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Encode name label to number label
        self.origin_label = self.data_df["label"].tolist()
        self.filename = self.data_df["filename"].tolist()
        self.label = []
        self.classIdDict = dict()

        for idx, label in enumerate(set(self.origin_label)):
            self.classIdDict[label] = idx

        for origin_label in self.origin_label:
            self.label.append(self.classIdDict[origin_label])

        self.len = len(self.label)

    def __getitem__(self, index):
        path = self.filename[index]
        label = self.label[index]
        ori_label = self.origin_label[index]

        image = Image.open(os.path.join(self.data_dir, path))
        image = self.transform(image)

        return image, label, ori_label

    def __len__(self):
        return self.len


class CategoriesSampler:
    def __init__(self, label, n_batch, n_way, n_shot, n_query):
        self.n_batch = n_batch
        self.n_cls = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_per = n_shot + n_query

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


# mini-Imagenet Test/Valid dataset
class MiniImageTestDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]

        image = Image.open(os.path.join(self.data_dir, path))
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


if __name__ == '__main__':
    #  Train Dataset
    train_dataset = MiniImageDataset('./hw4_data/mini/train.csv', './hw4_data/mini/train')

    # print(train_dataset.classIdDict)
    # print(len(train_dataset.label))
    # print(set(train_dataset.label))
    # print(len(train_dataset))
    # print(train_dataset[1])

    # batch_size = N_way * (N_query + N_shot) = 30 * (15 + 1) = 480
    # n_batch = number of train data / batch_size = 38400 / 480 = 80
    train_sampler = CategoriesSampler(train_dataset.label, n_batch=100,
                                      n_way=30, n_shot=1, n_query=15)

    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler,
                                  pin_memory=False, num_workers=3, worker_init_fn=worker_init_fn)

    for i, data in enumerate(train_dataloader):
        if i == 0:
            images, labels, ori_labels = data[0].to('cuda'), data[1].to('cuda'), data[2]
            print(labels)
            # print(ori_labels)
            # print(labels.size())
            #
            # print(torch.unique(labels))
            # print(torch.unique(labels).size())

            label_encoder = {ori_labels[i * 1]: i for i in range(30)}
            query_label = torch.cuda.LongTensor(
                [label_encoder[class_name] for class_name in ori_labels[30:]])

            support_label = torch.cuda.LongTensor(
                [label_encoder[class_name] for class_name in ori_labels[:30]])
            print(support_label)
            print(query_label)

            break

    # Valid and Test Dataset
    test_dataset = MiniImageTestDataset('./hw4_data/mini/val.csv', './hw4_data/mini/val')

    test_loader = DataLoader(
        test_dataset, batch_size=5 * (15 + 1),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler('./hw4_data/mini/val_testcase.csv'))

    print(len(test_loader))

    for i, batch in enumerate(test_loader):
        if i == 0:
            images, labels = batch[0].to('cuda'), batch[1]
            print(images.size())
            print(labels)

            support_input = images[:5, :, :, :]
            query_input = images[5:, :, :, :]

            label_encoder = {labels[i * 1]: i for i in range(5)}
            query_label = torch.cuda.LongTensor(
                [label_encoder[class_name] for class_name in labels[5:]])

            support_label = torch.cuda.LongTensor(
                [label_encoder[class_name] for class_name in labels[:5]])

            print(support_label)
            print(query_label)

            break


