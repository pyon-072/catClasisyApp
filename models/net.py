import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision.models import resnet50


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #学習済モデルと全結合層の定義
        #self.feature = resnet50(pretrained=False)
        self.feature = resnet50(weights=None)
        self.fc1 = nn.Linear(1000,100)              # 出力数100へ変換
        self.bn  = nn.BatchNorm1d(100)              # バッチノーマライゼーション層をかましてみる
        self.fc2 = nn.Linear(100,5)                 # 出力数5へ変換

    #学習済モデル→全結合層への順伝播の流れを定義
    def forward(self,x):
        h = self.feature(x)
        h = self.fc1(h)
        h = self.bn(h)
        h = F.relu(h)
        h = self.fc2(h)
        return h