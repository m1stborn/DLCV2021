import torch
from torch.utils.data import DataLoader

from model_p1.model import VGG16
from model_p1.image_dataset import ImageDataset
from parse_config import create_parser
from utils import load_checkpoint

if __name__ == '__main__':
    # init configs from args
    parser = create_parser()
    configs = parser.parse_args()

    val_dataset = ImageDataset('./p1_data/val_50/')
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size,
                                shuffle=True)

    ckpt = load_checkpoint(configs.path_to_checkpoint + "vgg16.pt")

    net = VGG16()
    net.load_state_dict(ckpt['net'])

    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the valid dataset: {:.4f}'
              .format(correct / total))
