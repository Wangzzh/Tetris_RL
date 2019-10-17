import torch.nn as nn
import torch.nn.functional as F
from replay import ReplayMemory

class TetrisDQN(nn.Module):

    def __init__(self, height=20, width=10, outChannel=6):
        super(TetrisDQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear((height - 6) * (width - 6) * 16, 16)
        self.fc2 = nn.Linear(16, outChannel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return x

