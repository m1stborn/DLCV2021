import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class DogCatDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filenames = []
        self.root = filepath
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if transform is not None:
            self.transform = transform

        # read filename
        for i, filename in enumerate(os.listdir(self.root)):
            label = filename.split('_')[0]
            f = os.path.join(self.root, filename)
            self.filenames.append((f, label))

        self.len = len(self.filenames)

    def __getitem__(self, idx):
        img_filename, label = self.filenames[idx]
        img = Image.open(img_filename).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(int(label))

        return img, label

    def __len__(self):
        return self.len


class DogCatTestDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filenames = []
        self.root = filepath
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if transform is not None:
            self.transform = transform

        # read filename
        for i, filename in enumerate(os.listdir(self.root)):
            f = os.path.join(self.root, filename)
            self.filenames.append((f, filename))

        self.len = len(self.filenames)

    def __getitem__(self, idx):
        img_filename, origin_filename = self.filenames[idx]
        img = Image.open(img_filename).convert("RGB")
        img = self.transform(img)

        return img, origin_filename

    def __len__(self):
        return self.len


if __name__ == '__main__':
    train_dataset = DogCatDataset('./hw3_data/p1_data/train')
    # image, label = train_dataset[0]
    # print(image.size(), label)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8,
                                             shuffle=True)

    # print(len(train_dataset))
    real_batch = next(iter(dataloader))
    imgs, labels = real_batch[0].to("cuda"), real_batch[1].to("cuda")
    # print(imgs.size(), labels.size())

    # test pytorch_pretrained_vit
    # from pytorch_pretrained_vit import ViT

    # net = ViT('B_32', pretrained=True, num_classes=37, num_heads=8, num_layers=6)
    # net.to("cuda")
    # B_16_imagenet1k: patch_size = 16, 1356.92 MB
    # B_32_imagenet1k: patch_size = 32, 1356.92 MB

    # from torchsummary import summary
    # summary(net, (3, 384, 384))
    # print(net(imgs).size())

    # Test DogCatTestDataset
    test_dataset = DogCatTestDataset('./hw3_data/p1_data/val')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8,
                                                  shuffle=False)
    real_batch = next(iter(test_dataloader))
    test_img, filenames = real_batch[0].to("cuda"), real_batch[1]
    print(test_img.size(), filenames)
