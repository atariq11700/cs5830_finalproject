import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_of_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=12, stride=4, padding=1)
        self.bn1_2d = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, stride=1, padding=1)
        self.bn2_2d = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1)
        self.bn3_2d = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_of_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1_2d(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2_2d(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3_2d(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0),-1)
        x = F.dropout(self.fc1(x))
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)