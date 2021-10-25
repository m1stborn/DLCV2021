import os
import numpy as np
import skimage.io
import torch
from torch.utils.data import DataLoader

from model_p2.fcn32 import FCN32
from model_p2.image_dataset import ImageDataset
from parse_config import create_parser
from utils import load_checkpoint
from mean_iou_evaluate import mean_iou_score, read_masks


def save_mask(filepath, pred, img_fn):
    masks_rgb = np.empty((len(pred), 512, 512, 3))
    for i, p in enumerate(pred):
        masks_rgb[i, p == 0] = [0, 255, 255]
        masks_rgb[i, p == 1] = [255, 255, 0]
        masks_rgb[i, p == 2] = [255, 0, 255]
        masks_rgb[i, p == 3] = [0, 255, 0]
        masks_rgb[i, p == 4] = [0, 0, 255]
        masks_rgb[i, p == 5] = [255, 255, 255]
        masks_rgb[i, p == 6] = [0, 0, 0]
    masks_rgb = masks_rgb.astype(np.uint8)
    for i, mask_rgb in enumerate(masks_rgb):
        # TODO: for submission file name should be same as input
        # skimage.io.imsave(os.path.join(filepath, img_fn[i]), mask_rgb)
        # mask_fn = img_fn[i].split('_')[0] + '_mask.png'
        mask_fn = img_fn[i] + '_mask.png'
        skimage.io.imsave(os.path.join(filepath, mask_fn), mask_rgb, check_contrast=False)


if __name__ == '__main__':
    # init configs from args
    parser = create_parser()
    configs = parser.parse_args()

    # TODO: chang file path to arg
    val_dataset = ImageDataset(configs.p2_input_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size,
                                shuffle=False, num_workers=0)

    ckpt = load_checkpoint(configs.ckpt)

    net = FCN32()
    net.load_state_dict(ckpt['net'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    net.to(device)

    with torch.no_grad():
        val_metrics = IOU()
        for val_data in val_dataloader:
            images, labels, img_filenames_prefix = val_data[0].to(device), val_data[1], val_data[2]
            outputs = net(images)
            predicted = outputs.max(dim=1)[1].data.cpu().numpy()
            save_mask(configs.p2_output_dir, predicted, img_filenames_prefix)
            # pred = torch.cat((pred, outputs.data.cpu()), 0)
            # masks = np.concatenate((masks, labels.cpu().numpy()), axis=0)
        # pred = pred.numpy()
        # print(pred.shape)
        # print(masks.shape)
        # print(np.argmax(pred, 1).shape)
        pred = read_masks(configs.p2_output_dir)
        labels = read_masks(configs.p2_input_dir)
        miou = mean_iou_score(pred, labels)
        print('Valid mIoU: {:.4f}'
              .format(miou))
