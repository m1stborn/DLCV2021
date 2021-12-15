import torch
import torch.nn as nn
import numpy as np
from pytorch_pretrained_vit import ViT
import matplotlib.pyplot as plt

from parse_config import create_parser
from utils import load_checkpoint

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
    net.eval()

    print(net.positional_embedding.pos_embedding.size())
    # num of patch: 24 x 24 = 576

    # print(net.positional_embedding.pos_embedding[:, :576, :].squeeze(0).size())
    # positional embedding matrix: 576 x 576
    x1 = net.positional_embedding.pos_embedding[:, 1:, :].squeeze(0)
    x2 = torch.transpose(x1, 0, 1)
    # x2 = x1
    print(x1.size(), x2.size())

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # cos_sim = cos(x1, x2)
    cos_sim = torch.matmul(x1, x2)  # 576 x 576

    norm_cos_sim = cos_sim - cos_sim.min()
    norm_cos_sim /= norm_cos_sim.max()

    fig, axes = plt.subplots(24, 24)

    im = None

    # for i, ax in enumerate(axes[:, 0]):
    #     ax.set_ylabel(str(i+1), size='large')
    #
    # for i, ax in enumerate(axes[-1, :]):
    #     ax.set_xlabel(str(i+1), size='large')

    for i in range(24):
        for j in range(24):
            axes[i, j].set_axis_off()
            # plt.setp(axes[i, j].get_xticklabels(), visible=False)
            # plt.setp(axes[i, j].get_yticklabels(), visible=False)
            # axes[i, j].tick_params(axis='both', which='both', length=0)
            im = axes[i, j].imshow(norm_cos_sim[24 * i + j].view(24, 24).detach().cpu().numpy(),
                                   cmap='viridis', interpolation='nearest')
    # for i in range(24):
    #     plt.setp(axes[-1, i], xlabel=str(i+1), size='small')
    #     plt.setp(axes[i, 0], ylabel=str(i+1), size='small')

    # fig.supylabel("input patch row")
    # fig.supxlabel("input patch column")

    fig.colorbar(im, ax=axes)
    plt.savefig(f'./result/p1/{ckpt["uid"][:8]}_position_embedding.png')
