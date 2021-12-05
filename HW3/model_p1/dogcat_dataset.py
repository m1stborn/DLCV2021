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
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
    from pytorch_pretrained_vit import ViT

    net = ViT('B_16_imagenet1k', pretrained=True, num_classes=37, num_heads=8, num_layers=6)
    net.to("cuda")
    # from torchsummary import summary
    # summary(net, (3, 384, 384))
    print(net(imgs).size())