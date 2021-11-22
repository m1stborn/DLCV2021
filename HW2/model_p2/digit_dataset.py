import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

torch.manual_seed(1)
torch.cuda.manual_seed(1)


class DigitDataset(Dataset):
    def __init__(self, csv_filename, filepath, transform=None):
        self.filenames = []
        self.root = filepath
        self.csv_filename = csv_filename
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if transform is not None:
            self.transform = transform

        with open(self.csv_filename) as f:
            rows = csv.reader(f, delimiter=',')
            header = next(rows)

            for filename, l in rows:
                # filename , l = row
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


# TODO:remove for submission

if __name__ == '__main__':
    train_dataset = DigitDataset('../hw2_data/digits/mnistm/train.csv', '../hw2_data/digits/mnistm/train')
    print(train_dataset[0])
    print(len(train_dataset))
    print(train_dataset[0][0].size())

    import torchvision.datasets as dset

    dataset = dset.CIFAR10(
        root='./data', download=True,
        transform=transforms.Compose([
            transforms.Scale(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    print(dataset[0])
