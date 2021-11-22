import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

torch.manual_seed(1)
torch.cuda.manual_seed(1)


class FaceDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filenames = []
        self.root = filepath
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if transform is not None:
            self.transform = transform

        # read filename
        for i, filename in enumerate(os.listdir(self.root)):
            self.filenames.append((os.path.join(self.root, filename)))

        self.len = len(self.filenames)

    def __getitem__(self, idx):
        img_filename = self.filenames[idx]
        img = Image.open(img_filename).convert("RGB")
        img = self.transform(img)
        return img

    def __len__(self):
        return self.len


# TODO:remove for submission

if __name__ == '__main__':
    train_dataset = FaceDataset('../hw2_data/face/train')

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                             shuffle=True)
    import torch
    import torchvision.utils as vutils
    import numpy as np
    import matplotlib.pyplot as plt
    print(train_dataset[0].size())
    real_batch = next(iter(dataloader))
    print(real_batch.size())
    print(len(train_dataset))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    # -1 ~ 1
    print(torch.max(train_dataset[0]))
    print(torch.min(train_dataset[0]))

    plt.imshow(np.transpose(vutils.make_grid(real_batch.to("cuda")[:32], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig("batch_1.png")
    # plt.show()

    # test image saving
    import skimage.io
    img = train_dataset[0].add(1).mul(255*0.5).numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.uint8)
    skimage.io.imsave("picture 1.png", img, check_contrast=False)


