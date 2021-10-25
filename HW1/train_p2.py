import os
import time
import uuid
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model_p2.vgg16_fcn32 import Vgg16FCN32
from model_p2.resnet_fcn32 import ResnetFCN32
from model_p2.image_dataset import ImageDataset
from parse_config import create_parser
from utils import save_checkpoint, load_checkpoint, progress_bar, experiment_record, save_mask
from mean_iou_evaluate import mean_iou_score, read_masks

# step 0: fix random seed for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if __name__ == '__main__':
    # init constants:
    parser = create_parser()
    configs = parser.parse_args()

    uid = str(uuid.uuid1())
    best_epoch = 0
    pre_val_miou = 0.0

    # step 1: prepare dataset
    train_dataset = ImageDataset('./p2_data/train')
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size,
                                  shuffle=True)

    val_dataset = ImageDataset('./p2_data/validation')
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size,
                                shuffle=False, num_workers=0)

    total_steps = len(train_dataloader)

    # step 2: init network
    net = ResnetFCN32()

    # step 3: define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=configs.lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=configs.lr, momentum=0.9)

    # step 4: check if resume training
    start_epoch = 0
    if configs.resume:
        ckpt = load_checkpoint(configs.ckpt)
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optim'])
        uid = ckpt['uid']
        pre_val_miou = ckpt['miou']
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        print("Checkpoint restored, start from epoch {}.".format(start_epoch + 1))

    # step 5: move Net to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    net.to(device)

    # step 6: main loop
    for epoch in range(start_epoch, start_epoch + configs.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            prefix = 'Epoch [{}/{}]-'.format(epoch + 1, start_epoch + configs.epochs)
            if (i + 1) % 10 == 0:  # print every 10 mini-batches
                suffix = 'Train Loss: {:.4f}'.format(running_loss / (i + 1))
                progress_bar(i + 1, total_steps, prefix, suffix)
            if configs.test_run:
                break

        # print Valid mIoU per epoch
        with torch.no_grad():
            # val_metrics = IOU()
            for val_data in val_dataloader:
                images, labels, img_fn_prefixs = val_data[0].to(device), val_data[1], val_data[2]
                outputs = net(images)
                predicted = torch.argmax(outputs, dim=1).cpu().numpy()
                # val_metrics.batch_iou(predicted, labels.cpu().numpy())

                # save predicted mask png file
                save_mask(configs.p2_output_dir, predicted, img_fn_prefixs)

            # val_metrics.update()
            # print('\nValid mIoU: {:.4f}'
            #       .format(val_metrics.miou()))

            # TA's mIoU:
            print('\n')
            pred = read_masks(configs.p2_output_dir)
            labels = read_masks(configs.p2_input_dir)
            miou = mean_iou_score(pred, labels)

            if pre_val_miou < miou:
                checkpoint = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'optim': optimizer.state_dict(),
                    'uid': uid,
                    'miou': miou
                }
                save_checkpoint(checkpoint,
                                os.path.join(configs.ckpt_path, "ResnetFCN32-{}.pt".format(uid[:8])))
                # pre_val_miou = val_metrics.mean_iou
                pre_val_miou = miou
                best_epoch = epoch

    # step 7: logging experiment
    experiment_record(uid, configs.batch_size, configs.lr, time.ctime(), best_epoch, pre_val_miou)
