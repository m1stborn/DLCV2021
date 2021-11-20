import os
import sys
import torch
import numpy as np

from IS import inception_score
from model_p1.gan_dataset import GANDataset

# TODO: remove this for submission
import warnings

warnings.simplefilter('ignore')


def save_checkpoint(state, save_path: str):
    torch.save(state, save_path)


def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path)
    return ckpt


def progress_bar(count, total, prefix='', suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    # percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(prefix + '[%s]-Step [%s/%s]-%s\r' % (bar, count, total, suffix))
    sys.stdout.flush()

    # if count == total:
    #     print("\n")


def experiment_record_p1(fn, *args):
    with open(fn, 'a') as f:
        print("""=======================================================
UUID:       {}
Time:       {}
Batch size: {}
Lr:         {} 
Result:
    Epoch:  {}
    IS:     {}
    FID:    {}
=======================================================""".format(*args), file=f)


def experiment_record_p2(fn, *args):
    with open(fn, 'a') as f:
        print("""=======================================================
UUID:       {}
Time:       {}
Batch size: {}
Lr:         {} 
Result:
    Epoch:  {}
    ACC:     {}
=======================================================""".format(*args), file=f)


def calculate_is_score(fp='./train'):
    train_dataset = GANDataset(filepath=fp)
    is_score = inception_score(train_dataset, cuda=True, batch_size=32, resize=True, splits=10)
    return is_score


def sample_idx():
    idx = [i for i in range(10)]
    idxs = []
    for j in range(10):
        idxs += ([e + 100 * j for e in idx])
    return idxs


def eval_net(net, data_loader, device, flag):
    net.eval()
    correct_class = 0
    correct_domain = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            size = len(images)
            if flag == 'target':
                labels_domain = torch.ones(size).long().to(device)
            else:
                labels_domain = torch.zeros(size).long().to(device)
            class_output, domain_output = net(images, alpha=0)

            _, predicted_class = torch.max(class_output.data, 1)
            _, predicted_domain = torch.max(domain_output.data, 1)

            correct_class += (predicted_class == labels).sum().item()
            correct_domain += (predicted_domain == labels_domain).sum().item()

            total += size

    return correct_class / total, correct_domain / total
