import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: require modify
class VGG16(nn.Module):
    def __init__(self, num_classes=50):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out

# class VGG16(nn.Module):
#     def __init__(self, n_classes):
#         super(VGG16, self).__init__()
#         # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#
#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#
#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#
#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#
#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#
#         # max pooling (kernel_size, stride)
#         self.pool = nn.MaxPool2d(2, 2)
#
#         # fully connected layers:
#         self.fc6 = nn.Linear(7*7*512, 4096)
#         self.fc7 = nn.Linear(4096, 4096)
#         self.fc8 = nn.Linear(4096, 1000)
#
#     def forward(self, x, training=True):
#         x = F.relu(self.conv1_1(x))
#         x = F.relu(self.conv1_2(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2_1(x))
#         x = F.relu(self.conv2_2(x))
#         x = self.pool(x)
#         x = F.relu(self.conv3_1(x))
#         x = F.relu(self.conv3_2(x))
#         x = F.relu(self.conv3_3(x))
#         x = self.pool(x)
#         x = F.relu(self.conv4_1(x))
#         x = F.relu(self.conv4_2(x))
#         x = F.relu(self.conv4_3(x))
#         x = self.pool(x)
#         x = F.relu(self.conv5_1(x))
#         x = F.relu(self.conv5_2(x))
#         x = F.relu(self.conv5_3(x))
#         x = self.pool(x)
#         x = x.view(-1, 7 * 7 * 512)
#         x = F.relu(self.fc6(x))
#         x = F.dropout(x, 0.5, training=training)
#         x = F.relu(self.fc7(x))
#         x = F.dropout(x, 0.5, training=training)
#         x = self.fc8(x)
#         return x
#
#     def predict(self, x):
#         # a function to predict the labels of a batch of inputs
#         x = F.softmax(self.forward(x, training=False))
#         return x
#
#     def accuracy(self, x, y):
#         # a function to calculate the accuracy of label prediction for a batch of inputs
#         #   x: a batch of inputs
#         #   y: the true labels associated with x
#         prediction = self.predict(x)
#         maxs, indices = torch.max(prediction, 1)
#         acc = 100 * torch.sum(torch.eq(indices.float(), y.float()).float())/y.size()[0]
#         return acc.cpu().data[0]
