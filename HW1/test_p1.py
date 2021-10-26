import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model_p1.model import VGG16
from model_p1.pretrained_resnet import PretrainedResnet
from model_p1.image_dataset import ImageDataset
from parse_config import create_parser
from utils import load_checkpoint

if __name__ == '__main__':
    # init configs from args
    parser = create_parser()
    configs = parser.parse_args()

    val_transforms = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    val_dataset = ImageDataset(configs.p1_valid_dir, transform=val_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size,
                                shuffle=False)

    ckpt = load_checkpoint(configs.ckpt)

    net = PretrainedResnet()
    net.load_state_dict(ckpt['net'])
    net.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    net.to(device)

    with torch.no_grad():
        correct = 0
        total = 0
        for val_data in val_dataloader:
            images, labels = val_data[0].to(device), val_data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the valid dataset: {:.4f} ({}/{})'
              .format((correct / total), correct, total))
