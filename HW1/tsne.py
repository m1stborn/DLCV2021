import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model_p1.pretrained_resnet import PretrainedResnet, ExtractedResnet
from model_p1.image_dataset import ImageDataset
from parse_config import create_parser
from utils import load_checkpoint


if __name__ == '__main__':
    # init configs from args
    parser = create_parser()
    configs = parser.parse_args()

    val_transforms = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    val_dataset = ImageDataset(configs.p1_valid_dir, transform=val_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size,
                                shuffle=False)

    ckpt = load_checkpoint(configs.ckpt)

    net = ExtractedResnet()
    net.load_state_dict(ckpt['net'])
    net.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    net.to(device)

    val_pred = []
    val_label = []

    with torch.no_grad():
        for val_data in val_dataloader:
            images, labels = val_data[0].to(device), val_data[1].to(device)
            outputs = net(images)

            val_pred.append(outputs)
            val_label.append(labels)

    val_pred = torch.cat(val_pred).cpu().numpy()
    val_label = torch.cat(val_label).cpu().numpy()

    # tSNE: reduce feature dim from 2048 to 2
    tsne = TSNE(n_components=2, init='pca', random_state=1, perplexity=50)
    x = tsne.fit_transform(val_pred)

    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=49.0)
    cmap = matplotlib.cm.get_cmap('rainbow')

    for i in np.unique(val_label):
        mask = val_label == i
        plt.scatter(x[mask, 0], x[mask, 1], label=i, color=cmap(norm(i)) )

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', ncol=5)
    # plt.show()


