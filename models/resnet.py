import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable

class ResNetZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, mode=1):
        super(ResNetZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)

        # get feature encoder
        if mode == 1:
            if pt:
                zoo_model = models.resnet18(pretrained=True)
            else:
                zoo_model = models.resnet18(pretrained=False)
        elif mode == 2:
            if pt:
                zoo_model = models.resnet34(pretrained=True)
            else:
                zoo_model = models.resnet34(pretrained=False)
        elif mode == 3:
            self.last = nn.Linear(2048, num_classes)
            if pt:
                zoo_model = models.wide_resnet50_2(pretrained=True)
            else:
                zoo_model = models.wide_resnet50_2(pretrained=False)

        self.feat = torch.nn.Sequential(*(list(zoo_model.children())[:-1]))

    def forward(self, x, pen=False, l=None):
        if l is None:
            out = self.feat(x)
            out = out.view(out.size(0), -1)
            if pen:
                return out
            else:
                out = self.last(out)
                return out
        else:
            if l == 0:
                out = self.feat(x)
                out = out.view(out.size(0), -1)
                out = self.last(out)
                return out
            elif l == 1:
                out = self.feat(x)
                return out.view(out.size(0), -1)
            elif l == 2:
                feat = torch.nn.Sequential(*(list(self.feat.children())[:-2]))
            elif l == 3:
                feat = torch.nn.Sequential(*(list(self.feat.children())[:-3]))
            elif l == 4:
                feat = torch.nn.Sequential(*(list(self.feat.children())[:-4]))
            elif l == 5:
                feat = torch.nn.Sequential(*(list(self.feat.children())[:-5]))
            else:
                raise ValueError('Value of layer is not valid')
            out = feat(x)
            return out.view(out.size(0), -1)

 
def resnet18(out_dim, block_division = None):
    return ResNetZoo(num_classes=out_dim, pt=False, mode=1)

def resnet18_pt(out_dim, block_division = None):
    return ResNetZoo(num_classes=out_dim, pt=True, mode=1)

def resnet34(out_dim, block_division = None):
    return ResNetZoo(num_classes=out_dim, pt=False, mode=2)

def resnet34_pt(out_dim, block_division = None):
    return ResNetZoo(num_classes=out_dim, pt=True, mode=2)

def WRN50_2(out_dim, block_division = None):
    return ResNetZoo(num_classes=out_dim, pt=False, mode=3)

def WRN50_2_pt(out_dim, block_division = None):
    return ResNetZoo(num_classes=out_dim, pt=True, mode=3)