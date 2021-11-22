import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class GANDataset(Dataset):
    def __init__(self, filepath):
        self.figsize = 64
        self.images = []
        self.file_list = os.listdir(filepath)
        self.file_list.sort()

        # print("Load file from :", filepath)
        for i, file in enumerate(self.file_list):
            # print("\r%d/%d" % (i, len(self.file_list)), end="")
            img = Image.open(os.path.join(filepath, file)).convert('RGB')
            self.images.append(img)

        # print("")
        # print("Loading file completed.")

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.num_samples = len(self.images)

    def __getitem__(self, index):
        return self.transform(self.images[index])

    def __len__(self):
        return self.num_samples
