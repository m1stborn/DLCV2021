import sys
import torch

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
Result:
    Epoch:  {}
    ACC:    {}
=======================================================""".format(*args), file=f)


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    # since our goal is to minimize the loss
    # => place negative sign to distance
    # => the closer the distance, the smaller the loss
    return logits


class AverageMeter:

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
