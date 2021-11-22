import os
import csv
import torch
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from model_p3.dann import DANN
from model_p3.digit_dataset import DigitTestDataset, DigitDataset
from parse_config import create_parser
from utils import load_checkpoint, eval_net

# step 0: fix random seed for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if __name__ == '__main__':
    # init configs from args
    parser = create_parser()
    configs = parser.parse_args()

    # if configs.tgt_domain == "mnistm"
    # SVHN → MNIST-M
    ckpt_filename = './ckpt/best/svhn-fdd7084a.pt'
    if configs.tgt_domain == "usps":
        # MNIST-M → USPS
        ckpt_filename = './ckpt/best/mnistm-ff6a5caf.pt'
    elif configs.tgt_domain == "svhn":
        # USPS → SVHN
        ckpt_filename = './ckpt/best/usps-21417126.pt'

    ckpt = load_checkpoint(ckpt_filename)
    print(ckpt_filename, f"TgtAcc: {ckpt['tgtacc']} SrcAcc: {ckpt['srcacc']}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    net = DANN()
    net.load_state_dict(ckpt['net'])
    net.to(device)
    net.eval()

    test_dataset = DigitTestDataset(configs.p3_input_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size,
                                 shuffle=False)

    # TODO: set to false for submission
    # valid_mode = False
    # if valid_mode:
    #     mnist_test_csv = './hw2_data/digits/mnistm/test.csv'
    #     svhn_test_csv = './hw2_data/digits/svhn/test.csv'
    #     usps_test_csv = './hw2_data/digits/usps/test.csv'
    #     if configs.tgt_domain == "mnistm":
    #         valid_dataset = DigitDataset(mnist_test_csv, configs.p3_input_dir)
    #     elif configs.tgt_domain == "usps":
    #         valid_dataset = DigitDataset(usps_test_csv, configs.p3_input_dir)
    #     elif configs.tgt_domain == "svhn":
    #         valid_dataset = DigitDataset(svhn_test_csv, configs.p3_input_dir)
    #     valid_dataloader = DataLoader(valid_dataset, batch_size=configs.batch_size,
    #                                   shuffle=False)
    #
    #     train_acc, train_acc_domain = eval_net(net, valid_dataloader, device, "target")
    #     print('TrainAcc: {:.4f} TrainDomainAcc: {:.4f}'.format(
    #         train_acc, train_acc_domain
    #     ))

    with torch.no_grad():
        # Testing:
        test_filename = []
        test_pred = []
        for test_data in test_dataloader:
            images, filenames = test_data[0].to(device), test_data[1]
            class_output, _ = net(images, alpha=0)

            _, predicted_class = torch.max(class_output.data, 1)

            test_pred.append(predicted_class)
            test_filename = test_filename + list(filenames)

        test_pred = torch.cat(test_pred).cpu().numpy()

        with open(configs.p3_output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image_name', 'label'])

            for row in zip(test_filename, test_pred):
                writer.writerow(row)
