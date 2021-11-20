import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class DigitDataset(Dataset):
    def __init__(self, csv_filename, filepath, transform=None):
        self.filenames = []
        self.root = filepath
        self.csv_filename = csv_filename
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if transform is not None:
            self.transform = transform

        with open(self.csv_filename) as f:
            rows = csv.reader(f, delimiter=',')
            header = next(rows)

            for filename, l in rows:
                self.filenames.append((os.path.join(self.root, filename), l))

        self.len = len(self.filenames)

    def __getitem__(self, idx):
        img_filename, label = self.filenames[idx]
        img = Image.open(img_filename).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(int(label))
        return img, label

    def __len__(self):
        return self.len

if __name__ == '__main__':
    mnistm = DigitDataset('../hw2_data/digits/mnistm/train.csv', '../hw2_data/digits/mnistm/train')
    svhn = DigitDataset('../hw2_data/digits/svhn/train.csv', '../hw2_data/digits/svhn/train')
    usps = DigitDataset('../hw2_data/digits/usps/train.csv', '../hw2_data/digits/usps/train')

    print(mnistm[0])
    print(svhn[0])
    print(usps[0])





