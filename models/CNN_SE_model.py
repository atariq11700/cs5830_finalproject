import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y



class CNN_SE(nn.Module):
    def __init__(self, num_of_classes):
        super(CNN_SE, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=12, stride=4, padding=1)
        self.bn1_2d = nn.BatchNorm2d(32)
        self.seblock1 = SEBlock(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, stride=1, padding=1)
        self.bn2_2d = nn.BatchNorm2d(64)
        self.seblock2 = SEBlock(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1)
        self.bn3_2d = nn.BatchNorm2d(128)
        self.seblock3 = SEBlock(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_of_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1_2d(self.conv1(x)))
        x = self.seblock1(x)
        x = self.pool1(x)
        x = F.relu(self.bn2_2d(self.conv2(x)))
        x = self.seblock2(x)
        x = self.pool2(x)
        x = F.relu(self.bn3_2d(self.conv3(x)))
        x = self.seblock3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(self.fc1(x))
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)