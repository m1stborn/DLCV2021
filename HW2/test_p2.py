import os
import torch
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from model_p2.acgan import Generator
from parse_config import create_parser
from utils import load_checkpoint, sample_idx
from digit_classifier import Classifier, load_classifier_checkpoint

# step 0: fix random seed for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)

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

    fixed_noise = ckpt['noise']
    fixed_noise_label = ckpt['noise_label']

    # TODO: remove for submission
    netDigit = Classifier()
    load_classifier_checkpoint("Classifier.pth", netDigit)
    netDigit.to(device)
    netDigit.eval()

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()

        # TODO: remove for submission
        outputs = netDigit(fake.to("cuda"))
        _, predicted = torch.max(outputs.data, 1)

        correct = (predicted == fixed_noise_label).sum().item()
        print(correct/1000)

        idx = sample_idx()
        plt.axis("off")
        plt.title("First 10 per digit")
        plt.imshow(
            np.transpose(make_grid(fake[idx].to("cuda"), padding=2, normalize=True, nrow=10).cpu(), (1, 2, 0))
        )
        plt.savefig("./p2_result/First_10.png")
        # ------------------------------

        idx_count = [0 for i in range(10)]
        for i, (img, label) in enumerate(zip(fake, fixed_noise_label)):
            label = label.item()
            idx_count[label] += 1

            img = img.squeeze(0).add(1).mul(255 * 0.5)
            img = img.cpu().numpy()
            img = np.transpose(img, (1, 2, 0)).astype(np.uint8)

            filename = os.path.join(configs.p2_output_dir, f"{label}_{str(idx_count[label]).zfill(3)}.png")
            skimage.io.imsave(filename, img, check_contrast=False)

