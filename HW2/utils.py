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


def experiment_record_p1(*args):
    with open("ckpt/p1_log.txt", 'a') as f:
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


def calculate_is_score(fp='./p1_result'):
    train_dataset = GANDataset(filepath=fp)
    is_score = inception_score(train_dataset, cuda=True, batch_size=32, resize=True, splits=10)
    return is_score
