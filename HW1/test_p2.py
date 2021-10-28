import torch
from torch.utils.data import DataLoader

from model_p2.vgg16_fcn8 import Vgg16FCN8
from model_p2.sat_image_dataset import SatImageTestDataset, SatImageDataset
from parse_config import create_parser
from utils import load_checkpoint, save_mask
from mean_iou_evaluate import mean_iou_score, read_masks

if __name__ == '__main__':
    # init configs from args
    parser = create_parser()
    configs = parser.parse_args()

    test_dataset = SatImageTestDataset(configs.p2_input_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size,
                                 shuffle=False)

    ckpt = load_checkpoint(configs.ckpt)

    net = Vgg16FCN8()
    net.load_state_dict(ckpt['net'])
    net.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    net.to(device)

    with torch.no_grad():
        # Validating:
        # val_dataset = SatImageDataset(configs.p2_input_dir)
        # val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size,
        #                             shuffle=False)
        #
        # for val_data in val_dataloader:
        #     images, labels, img_filenames_prefix = val_data[0].to(device), val_data[1], val_data[2]
        #     outputs = net(images)
        #     predicted = outputs.max(dim=1)[1].data.cpu().numpy()
        #     save_mask('./p2_valid', predicted, img_filenames_prefix)
        #
        # pred = read_masks('./p2_valid')
        # labels = read_masks(configs.p2_input_dir)
        # miou = mean_iou_score(pred, labels)
        # print('Valid mIoU: {:.4f}'
        #       .format(miou))

        # Testing:
        for test_data in test_dataloader:
            images, filenames = test_data[0].to(device), test_data[1]
            outputs = net(images)
            predicted = outputs.max(dim=1)[1].data.cpu().numpy()
            save_mask(configs.p2_output_dir, predicted, filenames, mode='Test')

