import os
import torch
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from model_p1.dcgan import Generator
from parse_config import create_parser
from utils import load_checkpoint, calculate_is_score

# step 0: fix random seed for reproducibility
torch.manual_seed(10)
torch.cuda.manual_seed(10)

if __name__ == '__main__':
    # init configs from args
    parser = create_parser()
    configs = parser.parse_args()

    ckpt = load_checkpoint(configs.ckpt)

    netG = Generator()
    netG.load_state_dict(ckpt['netG'])
    netG.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    netG.to(device)

    # random sample 1000 noise vector to generate 1000 images
    fixed_noise = torch.randn(1000, 100, 1, 1, device=device)

    with torch.no_grad():
        # generate 1000 images
        for i in range(1000):
            img = netG(fixed_noise[i].unsqueeze(0))
            img = img.squeeze(0).add(1).mul(255 * 0.5)
            img = img.cpu().numpy()
            img = np.transpose(img, (1, 2, 0)).astype(np.uint8)

            filename = os.path.join(configs.p1_output_dir, f"{i+1}.png".zfill(4))
            skimage.io.imsave(filename, img, check_contrast=False)

        # TODO: remove for submission
        # generate first 32 images in one (grid)
        first32_img = netG(fixed_noise[:32])
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("First 32 generated images")
        plt.imshow(
            np.transpose(make_grid(first32_img, padding=2, normalize=True).cpu(), (1, 2, 0))
        )
        plt.savefig(os.path.join(configs.p1_output_base, "First_32.png"))

    # TODO: remove for submission
    is_score_mean, is_score_std = calculate_is_score(configs.p1_output_dir)
    print('IS mean: {} IS std: {}'.format(is_score_mean, is_score_std))

