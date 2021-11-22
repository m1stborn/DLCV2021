import os
import csv
import torch
import skimage.io
import pickle
import itertools
import numpy as np
import matplotlib
from matplotlib import markers
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from model_p3.dann import ExtractedDANN
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    net = ExtractedDANN()
    net.load_state_dict(ckpt['net'])
    net.to(device)
    net.eval()

    test_dataset = DigitTestDataset(configs.p3_input_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size,
                                 shuffle=False)

    mnist_test_csv = './hw2_data/digits/mnistm/test.csv'
    svhn_test_csv = './hw2_data/digits/svhn/test.csv'
    usps_test_csv = './hw2_data/digits/usps/test.csv'
    if configs.tgt_domain == "mnistm":
        src_dataset = DigitDataset(svhn_test_csv, './hw2_data/digits/svhn/test')
        tgt_dataset = DigitDataset(mnist_test_csv, configs.p3_input_dir)
    elif configs.tgt_domain == "usps":
        src_dataset = DigitDataset(mnist_test_csv, './hw2_data/digits/mnistm/test')
        tgt_dataset = DigitDataset(usps_test_csv, configs.p3_input_dir)
    elif configs.tgt_domain == "svhn":
        src_dataset = DigitDataset(usps_test_csv, './hw2_data/digits/usps/test')
        tgt_dataset = DigitDataset(svhn_test_csv, configs.p3_input_dir)

    tgt_dataloader = DataLoader(tgt_dataset, batch_size=configs.batch_size,
                                shuffle=False)
    src_dataloader = DataLoader(src_dataset, batch_size=configs.batch_size,
                                shuffle=False)

    features = []
    class_label = []
    domain_label = []

    with torch.no_grad():
        for data in tgt_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images, 0)

            size_src = len(images)
            features.append(outputs)
            class_label.append(labels)
            domain_label.append(torch.ones(size_src).long().to(device))  # target 1

        for data in src_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images, 0)

            size_src = len(images)
            features.append(outputs)
            class_label.append(labels)
            domain_label.append(torch.zeros(size_src).long().to(device))  # source 0

    features = torch.cat(features).cpu().numpy()
    class_label = torch.cat(class_label).cpu().numpy()
    domain_label = torch.cat(domain_label).cpu().numpy()

    print(features.shape)
    print(class_label.shape)
    print(domain_label.shape)

    # tSNE: reduce feature dim from 48 * 4 * 4 to 2
    tsne = TSNE(n_components=2, init='random', random_state=1, perplexity=50, n_jobs=4)
    x = tsne.fit_transform(features)
    pickle.dump(x, open(f"./p3_result/{configs.tgt_domain}_x.sav", 'wb'))

    # x = pickle.load(open(f"./p3_result/{configs.tgt_domain}_x.sav", 'rb'))

    plt.style.use('ggplot')
    m_styles = markers.MarkerStyle.markers
    colormap = plt.cm.tab10.colors

    N = 10
    my_dpi = 151
    plt.figure(1, figsize=(3840/my_dpi, 2160/my_dpi), dpi=my_dpi)
    for i, (marker, color) in zip(range(N), itertools.product(m_styles, colormap)):
        mask = class_label == i
        print(f"start drawing class_label {i}")
        plt.scatter(x[mask, 0], x[mask, 1], color=color, marker=marker, label=i)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1)
    plt.savefig(f"./p3_result/{configs.tgt_domain}_class.png")
    # plt.show()

    N = 2
    plt.figure(2, figsize=(3840/my_dpi, 2160/my_dpi), dpi=my_dpi)
    for i, (marker, color) in zip(range(N), itertools.product(m_styles, colormap)):
        mask = domain_label == i
        print(f"start drawing domain_label {i}")
        plt.scatter(x[mask, 0], x[mask, 1], color=color, marker=marker, label=i)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1)
    plt.savefig(f"./p3_result/{configs.tgt_domain}_domain.png")
    # plt.show()


