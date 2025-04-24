from .. import nn
from ..cpa.model import Model


class CIFARNet(Model):
    def __init__(
        self,
        norm_layer=nn.BatchNorm2d,
        name: str = "CIFARNet",
    ):
        super().__init__()
        self.name = name
        self.n_relu = 3

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.norm1 = norm_layer(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.norm2 = norm_layer(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.norm3 = norm_layer(64)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x

    def forward_layer(self, x, depth):
        x = self.conv1(x)
        x = self.norm1(x)
        if depth == 0:
            return x
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        if depth == 1:
            return x
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        if depth == 2:
            return x
        x = self.relu(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        if depth == 3:
            return x
        x = self.relu(x)

        x = self.fc2(x)
        return x

    def _change_norm(self, _norm: nn.BatchNorm2d | nn.Norm2d):
        norm_dict = {}
        for k, v in self._modules.items():
            if not isinstance(v, nn.BatchNorm2d) and not isinstance(v, nn.Norm2d):
                continue
            norm = _norm(v.num_features)
            norm_dict[k] = norm
        self._modules.update(norm_dict)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        try:
            self._change_norm(nn.BatchNorm2d)
            return super().load_state_dict(state_dict, strict, assign)
        except Exception as _:
            self._change_norm(nn.Norm2d)
            return super().load_state_dict(state_dict, strict, assign)


class CIFARLinearNet(Model):
    def __init__(
        self,
        norm_layer=nn.BatchNorm1d,
        name: str = "CIFARLinearNet",
    ):
        super().__init__()
        self.name = name
        self.n_relu = 2
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.norm1 = norm_layer(512)

        self.fc2 = nn.Linear(512, 256)
        self.norm2 = norm_layer(256)

        self.fc3 = nn.Linear(256, 128)
        self.norm3 = norm_layer(128)

        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.fc4(x)
        return x

    def forward_layer(self, x, depth):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.norm1(x)
        if depth == 0:
            return x
        x = self.relu(x)

        x = self.fc2(x)
        x = self.norm2(x)
        if depth == 1:
            return x
        x = self.relu(x)

        x = self.fc3(x)
        x = self.norm3(x)
        if depth == 2:
            return x
        x = self.relu(x)

        x = self.fc4(x)
        return x
