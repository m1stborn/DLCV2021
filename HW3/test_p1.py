import csv
import torch
from torch.utils.data import DataLoader
from pytorch_pretrained_vit import ViT

from model_p1.dogcat_dataset import DogCatTestDataset
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

    test_dataset = DogCatTestDataset(configs.p1_input_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size,
                                 shuffle=False)

    with torch.no_grad():
        # Testing:
        test_filename = []
        test_pred = []

        for test_data in test_dataloader:
            images, filenames = test_data[0].to(device), test_data[1]
            class_output = net(images)

            _, predicted_class = torch.max(class_output.data, 1)

            test_pred.append(predicted_class)
            test_filename = test_filename + list(filenames)

        test_pred = torch.cat(test_pred).cpu().numpy()

        with open(configs.p1_output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image_name', 'label'])

            for row in zip(test_filename, test_pred):
                writer.writerow(row)
