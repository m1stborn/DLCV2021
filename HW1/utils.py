import os
import shutil
import numpy as np
import sys
import torch


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

    sys.stdout.write(prefix + '[%s]-Step[%s/%s] - %s\r' % (bar, count, total, suffix))
    sys.stdout.flush()

    if count == total:
        print("\n")


def experiment_record(*args):
    with open("./ckpt/log.txt", 'a') as f:
        print("""============================================
UUID: {},
Time: {},
Result:
    Epoch: {},
    Valid ACC: {},
============================================""".format(*args))
