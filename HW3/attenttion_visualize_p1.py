import os.path

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_pretrained_vit import ViT
import matplotlib.pyplot as plt

from parse_config import create_parser
from utils import load_checkpoint


def visualize_grid_to_grid(image, att_map, grid_size=24):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    mask = att_map[0, 1:].detach().numpy().reshape(grid_size[0], grid_size[1])
    mask = mask / mask.max()
    # resize
    # mask = cv2.resize(mask / mask.max(), image.size)[..., np.newaxis]
    mask = Image.fromarray(mask).resize(image.size)

    # result = (mask_overlay * image).astype("uint8")
    # return mask
    return mask / np.max(mask)

def attention_visualize(img, att_mat):
    """
    args:
        img = PIL image
        att_mat = torch tensor
    """
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.

    # residual_att = torch.eye(att_mat.size(1))
    # aug_att_mat = att_mat + residual_att
    # aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    # joint_attentions = torch.zeros(aug_att_mat.size())
    # joint_attentions[0] = aug_att_mat[0]

    # Attention from the output token to the input space.
    # v = joint_attentions[-1]
    # v = aug_att_mat[1]
    v = att_mat
    grid_size = int(np.sqrt(att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
    result = (mask * img).astype("uint8")
    print(result.shape)
    return result


# step 0: fix random seed for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if __name__ == '__main__':
    # init configs from args
    parser = create_parser()
    configs = parser.parse_args()

    ckpt = load_checkpoint(configs.ckpt)
    print(f"Ckpt ACC: {ckpt['acc']}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    net = ViT('B_16_imagenet1k', pretrained=True, num_classes=37, num_heads=8, num_layers=6)
    net.to("cuda")
    net.load_state_dict(ckpt['net'])
    # net.eval()

    base_path = "./hw3_data/p1_data/val"
    image_filenames = ["26_5064.jpg",
                       "29_4718.jpg",
                       "31_4838.jpg"]

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # for layer in range(12):
    for layer in range(12):
        for fn in image_filenames:
            origin_img = Image.open(os.path.join(base_path, fn))
            im = transform(origin_img)

            output = net(im.unsqueeze(0).to(device))  # 12 x 577 x 577 (head x patches x patches )
            att_matrix = net.transformer.blocks[layer].attn.scores.squeeze(0)  # 12 x 577 x 577
            att_matrix_mean = torch.mean(att_matrix, dim=0)  # 577 x 577
            # result = attention_visualize(origin_img, att_matrix_mean.cpu())
            result = visualize_grid_to_grid(origin_img, att_matrix_mean.cpu())

            # Save image
            # my_dpi = 151
            # plt.figure(1, figsize=(3840 / my_dpi, 2160 / my_dpi), dpi=my_dpi)
            #
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
            ax1.set_title('Original')
            ax2.set_title('Attention Map')
            ax1.set_axis_off()
            ax2.set_axis_off()

            _ = ax1.imshow(origin_img)

            _ = ax2.imshow(origin_img)
            _ = ax2.imshow(result, alpha=0.6, cmap='rainbow')
            plt.savefig(f"./result/p1/{ckpt['uid'][:8]}-{fn.replace('.jpg', '')}_l{layer}.png")

            # @without imshow method
            # cmap = plt.get_cmap('rainbow')
            # rgba_img = cmap(result)
            # rgb_img = np.delete(rgba_img, 3, 2)
            #
            # alpha1 = 1  # background image alpha
            # alpha2 = 0.5  # foreground image alpha
            # arr1 = np.asarray(origin_img)
            # arr2 = (rgb_img*255).astype("uint8")
            #
            # overlap = np.asarray((alpha2 * arr2 + alpha1 * (1 - alpha2) * arr1) / (alpha2 + alpha1 * (1 - alpha2)),
            #                      dtype=np.uint8)
            #
            # plt.imsave(f"./result/p1/{fn.replace('.jpg', '')}_show.png", overlap)
