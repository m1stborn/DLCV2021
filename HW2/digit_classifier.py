import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable


def load_classifier_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    
    # load digit classifier
    net = Classifier()
    path = "Classifier.pth"
    load_classifier_checkpoint(path, net)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        net = net.to(device)

    # print(net)

    from torchsummary import summary
    # summary(net, (3, 28, 28))

    fixed_noise = []
    fixed_noise_label = []
    size_per_classes = 100
    num_classes = 10
    latent_size = 100

    for i in range(10):
        label = np.full((size_per_classes,), i)
        # label = np.random.randint(0, num_classes, configs.batch_size)
        noise = np.random.normal(0, 1, (size_per_classes, latent_size))
        label_onehot = np.zeros((size_per_classes, num_classes))
        label_onehot[np.arange(size_per_classes), label] = 1
        noise[:, :num_classes] = label_onehot
        noise = torch.from_numpy(noise)
        noise = noise.view(size_per_classes, latent_size, 1, 1)

        fixed_noise.append(noise)
        fixed_noise_label.append(torch.from_numpy(label))
    fixed_noise = torch.cat(fixed_noise, 0).to(device)
    fixed_noise_label = torch.cat(fixed_noise_label, 0).to(device)

    # print(fixed_noise_label)

    fake_img = torch.randn((1000, 3, 28, 28), device=device)
    outputs = net(fake_img)
    _, predicted = torch.max(outputs.data, 1)

    correct = (predicted == fixed_noise_label).sum().item()
    print('\nValid ACC: {:.4f}'
          .format(correct / 1000))

    c = 0

    for i in range(0, 1000, 100):
        # fake_img = netG(fixed_noise[i:i + 100])
        outputs = net(fake_img[i:i + 100])
        _, predicted = torch.max(outputs.data, 1)
        c += (predicted == fixed_noise_label[i:i + 100]).sum().item()

    print('\nValid ACC: {:.4f}'
          .format(c / 1000))


