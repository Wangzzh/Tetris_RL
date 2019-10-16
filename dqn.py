import torch.nn as nn
import torch.nn.functional as F
from replay import ReplayMemory

class TetrisDQN(nn.Module):

    def __init__(self, height=20, width=10, outChannel=6):
        super(TetrisDQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=4, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc = nn.Linear((height - 9) * (width - 9) * 32, outChannel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.fc(x.view(x.size(0), -1))
        return x

