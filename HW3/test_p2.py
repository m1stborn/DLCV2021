import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import BertTokenizer
from PIL import Image

from model_p2 import caption
from model_p2.configuration import Config
import os

import warnings
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image folder', default='./hw3_data/p2_data/images')
parser.add_argument('--ckpt', type=str, help='checkpoint path', default='./ckpt/p2/weight493084032.pth')
parser.add_argument('--output', type=str, help='path to output folder', default='./result/p2')
args = parser.parse_args()

config = Config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

net, _ = caption.build_model(config)
ckpt = torch.load(args.ckpt)
net.load_state_dict(ckpt['model'])
net.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


@torch.no_grad()
def evaluate(image):
    net.eval()
    caption_cut = [start_token]

    cap_and_mask = create_caption_and_mask(start_token, config.max_position_embeddings)
    cap, cap_mask = cap_and_mask[0].to(device), cap_and_mask[1].to(device)

    for i in range(config.max_position_embeddings - 1):
        predictions = net(image, cap, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, dim=-1)

        # print("cross_attn size", net.transformer.decoder.layers[-1].scores.size())
        # print(net.transformer.decoder.layers[-1].scores)

        # To include <end>
        caption_cut.append(predicted_id[0])

        if predicted_id[0] == 102:
            return cap, caption_cut

        cap[:, i + 1] = predicted_id[0]
        cap_mask[:, i + 1] = False

    return cap, caption_cut


def visualize_grid_to_grid(image, att_map, grid_size=19):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    mask = att_map.cpu().detach().numpy().reshape(grid_size[0], grid_size[1])
    mask = mask / mask.max()

    mask = Image.fromarray(mask).resize(image.size)

    # return mask / np.max(mask)
    return mask


transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    image_filenames = os.listdir(args.path)
    print(image_filenames)

    for fn in image_filenames:
        origin_img = Image.open(os.path.join(args.path, fn))
        img = transform(origin_img)
        img = img.unsqueeze(0)  # 3 x 299 x 299
        output, cut = evaluate(img.to(device))

        att_matrix = net.transformer.decoder.layers[-1].scores  # 1 x 128 x 361

        result = tokenizer.decode(cut, skip_special_tokens=False)
        # print(result.capitalize())

        # visualization one by one
        # for i, token_idx in enumerate(cut):
        #     word = tokenizer.decode([token_idx], skip_special_tokens=False)
        #     vis_mask = visualize_grid_to_grid(origin_img, att_matrix[0, i])
        #
        #     cmap = plt.get_cmap('rainbow')
        #     rgba_img = cmap(vis_mask)
        #     rgb_img = np.delete(rgba_img, 3, 2)
        #
        #     alpha1 = 1  # background image alpha
        #     alpha2 = 0.6  # foreground image alpha
        #     arr1 = np.asarray(origin_img)
        #     arr2 = (rgb_img * 255).astype("uint8")
        #
        #     overlap = np.asarray((alpha2 * arr2 + alpha1 * (1 - alpha2) * arr1) / (alpha2 + alpha1 * (1 - alpha2)),
        #                          dtype=np.uint8)
        #     # word = word.replace('[', '').replace(']', '')
        #     plt.imsave(f"../HW3/result/p2/{fn.replace('.jpg', '')}_{i}_{word}.png", overlap)

        # visualization all in one
        num_row = int(len(cut) / 5 + 1)
        fig, axs = plt.subplots(nrows=num_row, ncols=5, figsize=(16, 9))
        for ax in axs.flat:
            _ = ax.set_axis_off()

        for i, token_idx in enumerate(cut):
            ax = axs.flat[i]
            word = tokenizer.decode([token_idx], skip_special_tokens=False)
            vis_mask = visualize_grid_to_grid(origin_img, att_matrix[0, i])

            _ = ax.set_title(word)
            _ = ax.imshow(origin_img)
            if i != 0:
                _ = ax.imshow(vis_mask/np.max(vis_mask), alpha=0.6, cmap='rainbow')

        fig.tight_layout()
        output_filename = os.path.join(args.output, f"{fn.replace('.jpg', '')}.png")
        plt.savefig(output_filename)
