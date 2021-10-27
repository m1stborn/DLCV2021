import os
import torch
import imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class SatImageDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filenames = []
        self.root = filepath
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )

        if transform is not None:
            self.transform = transform

        files = [f for f in os.listdir(self.root) if f.endswith('.jpg')]
        for i, img_filename in enumerate(files):
            file_prefix = img_filename.split('_')[0]
            mask_filename = os.path.join(self.root, file_prefix + '_mask.png')
            # TODO: remove file_prefix
            self.filenames.append((os.path.join(self.root, img_filename), mask_filename, file_prefix))

        self.len = len(self.filenames)

    def __getitem__(self, idx):

        img_filename, mask_filename, file_prefix = self.filenames[idx]
        img = Image.open(img_filename).convert("RGB")
        img = self.transform(img)

        mask = read_mask(mask_filename)

        return img, torch.tensor(mask).long(), file_prefix

    def __len__(self):
        return self.len


class SatImageTestDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filenames = []
        self.root = filepath
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )

        if transform is not None:
            self.transform = transform

        files = [f for f in os.listdir(self.root) if f.endswith('.jpg')]
        for i, img_filename in enumerate(files):
            self.filenames.append((os.path.join(self.root, img_filename), img_filename))

        self.len = len(self.filenames)

    def __getitem__(self, idx):

        img_filename, origin_filename = self.filenames[idx]
        img = Image.open(img_filename).convert("RGB")
        img = self.transform(img)

        return img, origin_filename

    def __len__(self):
        return self.len


def read_mask(filename):

    mask = imageio.imread(os.path.join(filename))
    mask = (mask >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]

    out = np.empty((512, 512))
    out[mask == 3] = 0  # (Cyan: 011) Urban land
    out[mask == 6] = 1  # (Yellow: 110) Agriculture land
    out[mask == 5] = 2  # (Purple: 101) Rangeland
    out[mask == 2] = 3  # (Green: 010) Forest land
    out[mask == 1] = 4  # (Blue: 001) Water
    out[mask == 7] = 5  # (White: 111) Barren land
    out[mask == 0] = 6  # (Black: 000) Unknown

    return out


# TODO:remove for submission

# if __name__ == '__main__':
#
#     import numpy as np
#     train_dataset = SatImageDataset('./p2_data/validation')
#     from torch.utils.data import DataLoader
#     report_pic_idx = [10, 97, 107]
#     report_dataset = torch.utils.data.Subset(train_dataset, report_pic_idx)
#     report_dataloader = DataLoader(report_dataset, batch_size=len(report_dataset),shuffle=False)
#     batch = next(iter(report_dataloader))
#     print(batch[0].size(), batch[1].size(), batch[2])

