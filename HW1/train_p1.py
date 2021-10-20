import time
import uuid
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model_p1.model import VGG16
from model_p1.pretrained_resnet import PretrainedResnet
from model_p1.pretrained_vgg16 import PretrainedVGG16
from model_p1.vgg16_batchnorm import VGG16BN
from model_p1.image_dataset import ImageDataset
from parse_config import create_parser
from utils import save_checkpoint, load_checkpoint, progress_bar, experiment_record

# step 0: fix random seed for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if __name__ == '__main__':
    # init constants:
    parser = create_parser()
    configs = parser.parse_args()

    uid = str(uuid.uuid1())
    best_epoch = 0
    pre_val_acc = 0.0

    # step 1: prepare dataset
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomRotation(25),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = ImageDataset('./p1_data/train_50/', transform=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size,
                                  shuffle=True)
    val_dataset = ImageDataset('./p1_data/val_50/', transform=val_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size,
                                shuffle=True)

    total_steps = len(train_dataloader)

    # step 2: init network
    net = PretrainedResnet()

    # step 3: define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=configs.lr, momentum=0.9)

    # step 4: check if resume training
    start_epoch = 0
    if configs.resume:
        ckpt = load_checkpoint(configs.ckpt)
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optim'])
        uid = ckpt['uid']
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

        # print Valid Accuracy per epoch
        correct = 0
        total = 0
        for val_data in val_dataloader:
            images, labels = val_data[0].to(device), val_data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('\nValid ACC: {:.4f}'
              .format(correct / total))

        # step 6: save checkpoint if better than previous
        if pre_val_acc < (correct / total):
            checkpoint = {
                'net': net.state_dict(),
                'epoch': epoch,
                'optim': optimizer.state_dict(),
                'uid': uid
            }
            save_checkpoint(checkpoint, configs.ckpt_path+"vgg16-{}.pt".format(uid[:8]))
            pre_val_acc = correct / total
            best_epoch = epoch

    experiment_record(uid, time.ctime(), best_epoch, pre_val_acc)
