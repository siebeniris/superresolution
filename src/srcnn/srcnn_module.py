import torch.nn as nn
from src.pytorch_train import PyTorchTrainerModel


class SRCNN(PyTorchTrainerModel):

    def __init__(self):
        super(SRCNN, self).__init__("srcnn")
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, img):
        out = self.conv1(img)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out



class SRCNNR(PyTorchTrainerModel):

    def __init__(self):
        super(SRCNNR, self).__init__("srcnn-residual")
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.bnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.bnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.bnorm3 = nn.BatchNorm2d(3)
        self.relu3 = nn.ReLU()

    def forward(self, img):
        residual = img
        out = self.conv1(img)
        out = self.bnorm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bnorm2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out += residual
        out = self.bnorm3(out)
        out = self.relu3(out)
        return out

class SRCNNBatchNorm(PyTorchTrainerModel):

    def __init__(self):
        super(SRCNNBatchNorm, self).__init__("srcnn-bnorm")
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.bnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.bnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.bnorm3 = nn.BatchNorm2d(3)

    def forward(self, img):
        out = self.conv1(img)
        out = self.bnorm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bnorm2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bnorm3(out)
        return out


