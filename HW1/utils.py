import os
import sys
import shutil
import numpy as np
import skimage.io
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

    sys.stdout.write(prefix + '[%s]-Step [%s/%s]-%s\r' % (bar, count, total, suffix))
    sys.stdout.flush()

    # if count == total:
    #     print("\n")


def experiment_record(*args):
    with open("./ckpt/p2/log.txt", 'a') as f:
        print("""=======================================================
UUID:       {}
Time:       {}
Batch size: {}
Lr:         {} 
Result:
    Epoch:      {}
    Valid ACC:  {}
=======================================================""".format(*args), file=f)


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

