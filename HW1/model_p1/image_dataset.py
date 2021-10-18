import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filenames = []
        self.root = filepath
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # [transforms.ToTensor(),
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # [transforms.Resize((224, 224)),
        #  transforms.ToTensor(),
        #  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        if transform is not None:
            self.transform = transform

        # read filename
        for i, filename in enumerate(os.listdir(self.root)):
            label = filename.split('_')[0]
            self.filenames.append((self.root + filename, label))

        self.len = len(self.filenames)

    def __getitem__(self, idx):

        img_filename, label = self.filenames[idx]
        img = Image.open(img_filename).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(int(label))

        return img, label

    def __len__(self):
        return self.len
