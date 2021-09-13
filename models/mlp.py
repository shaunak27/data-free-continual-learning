import torch
import torch.nn as nn
from .layers import CosineScaling
from torch.nn import functional as F

KL = False

class MLP(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256, cosine = False, bn = False):
        super(MLP, self).__init__()
        self.BN = bn
        self.in_dim = in_channel*img_sz*img_sz
        self.feature_layer_idx = [1,2,-1]
        if self.BN:
            self.linear1 = nn.Sequential(
                nn.Linear(self.in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(self.in_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
        if cosine:
            self.last = CosineScaling(hidden_dim, out_dim)  # Subject to be replaced dependent on task
        else:
            self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear1(x.view(-1,self.in_dim))
        x = self.linear2(x)
        if KL: x = x.softmax(dim=-1)
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def penultimate(self, x):
        x = self.features(x)
        return x

    def feature_forward(self, x):
        out1 = self.linear1(x.view(-1,self.in_dim))
        out2 = self.linear2(out1)
        if KL: out2 = out2.softmax(dim=-1)
        out3 = self.logits(out2)
        return {1:out1, 2:out2, -1:out3}

class MLPHead(nn.Module):

    def __init__(self, in_dim, out_dim=10, hidden_dim=256):
        super(MLPHead, self).__init__()
        self.in_dim = in_dim
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1, self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

def MLP400_MNIST(bn = False):
    return MLP(in_channel=1, img_sz=28, hidden_dim=400)
def MLP1000_MNIST(bn = False):
    return MLP(in_channel=1, img_sz=28, hidden_dim=1000)

def MLP400_MNIST_COSINE(bn = False):
    return MLP(in_channel=1, img_sz=28, hidden_dim=400, cosine = True)
def MLP1000_MNIST_COSINE(bn = False):
    return MLP(in_channel=1, img_sz=28, hidden_dim=1000, cosine = True)