import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=28, hidden_dim=256):
        super(MLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x, pen=False):
        x = self.features(x)
        if pen:
            return x
        else:
            x = self.logits(x)
            return x

    def penultimate(self, x):
        x = self.features(x)
        return x

def MLP100(out_dim):
    return MLP(hidden_dim=100)

def MLP200(out_dim):
    return MLP(hidden_dim=200)
    
def MLP300(out_dim):
    return MLP(hidden_dim=300)

def MLP400(out_dim):
    return MLP(hidden_dim=400)

def MLP500(out_dim):
    return MLP(hidden_dim=500)

def MLP600(out_dim):
    return MLP(hidden_dim=600)

def MLP800(out_dim):
    return MLP(hidden_dim=800)

def MLP1000(out_dim):
    return MLP(hidden_dim=1000)


def MLP2000(out_dim):
    return MLP(hidden_dim=2000)


def MLP5000(out_dim):
    return MLP(hidden_dim=5000)