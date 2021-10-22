import torch
from torch.utils.data import DataLoader

from model_p2.fcn32 import FCN32
from model_p2.image_dataset import ImageDataset
from model_p2.metrics import IOU
from parse_config import create_parser
from utils import load_checkpoint

if __name__ == '__main__':
    # init configs from args
    parser = create_parser()
    configs = parser.parse_args()

    val_dataset = ImageDataset('./p2_data/validation/')
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size,
                                shuffle=True)

    ckpt = load_checkpoint(configs.ckpt)

    net = FCN32()
    net.load_state_dict(ckpt['net'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    net.to(device)

    with torch.no_grad():
        val_metrics = IOU()
        for val_data in val_dataloader:
            images, labels = val_data[0].to(device), val_data[1]
            outputs = net(images)
            predicted = outputs.max(dim=1)[1].data.cpu().numpy()
            val_metrics.batch_iou(predicted, labels.cpu().numpy())

        val_metrics.update()
        print('\nValid mIoU: {:.4f}'
              .format(val_metrics.mean_iou))
