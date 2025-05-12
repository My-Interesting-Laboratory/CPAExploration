import numpy as np
import torch

from torchays import nn
from torchays.nn.modules import Tensor

GPU_ID = 0
device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(5)
torch.cuda.manual_seed_all(5)
np.random.seed(5)


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.avg = nn.AvgPool2d(2, 1)
        self.linear = nn.Linear(16, 2)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.avg(x)

        x = self.flatten(x)

        x = self.linear(x)
        return x


if __name__ == "__main__":
    net = TestNet().to(device)
    data = torch.randn(2, 3, 8, 8).to(device)

    net.graph()
    with torch.no_grad():
        output: Tensor = net(data)
        weight, bias = output.weight_graph, output.bias_graph
        print(output)
        for i in range(output.size(0)):
            output = (weight[i] * data[i]).sum(dim=(1, 2, 3)) + bias[i]
            print(output)
