import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# mini-Imagenet dataset
class ImageDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if transform is not None:
            self.transform = transform

        # Encode name label to number label
        self.origin_label = self.data_df["label"].tolist()
        self.filename = self.data_df["filename"].tolist()
        self.label = []
        self.name2label = dict()

        for idx, label in enumerate(set(self.origin_label)):
            self.name2label[label] = idx

        for origin_label in self.origin_label:
            self.label.append(self.name2label[origin_label])

        self.len = len(self.label)
        self.num_classes = len(self.name2label)

    def __getitem__(self, index):
        path = self.filename[index]
        label = self.label[index]
        ori_label = self.origin_label[index]

        image = Image.open(os.path.join(self.data_dir, path))
        image = self.transform(image)

        return image, label, ori_label

    def __len__(self):
        return self.len


if __name__ == '__main__':
    #  Train Dataset
    train_dataset = ImageDataset('./hw4_data/mini/train.csv', './hw4_data/mini/train')
    print(train_dataset.name2label)
    print(train_dataset.num_classes)

    office_dataset = ImageDataset('./hw4_data/office/train.csv', './hw4_data/office/train')

    print(office_dataset.name2label)
    print(office_dataset.num_classes)
    print(office_dataset[0][0].size(), office_dataset[0][1], office_dataset[0][2])
