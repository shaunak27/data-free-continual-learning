import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable

class ResNetZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, mode=1):
        super(ResNetZoo, self).__init__()
        self.vit_flag = False

        # get last layer
        self.last = nn.Linear(512, num_classes)

        # get feature encoder
        if mode == 0:
            if pt:
                from transformers import  ViTForImageClassification
                zoo_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
                self.last = nn.Linear(768, num_classes)
                self.vit_flag = True
        elif mode == 1:
            zoo_model = models.resnet18(pretrained=pt)
        elif mode == 2:
            zoo_model = models.resnet34(pretrained=pt)
        elif mode == 3:
            self.last = nn.Linear(2048, num_classes)
            zoo_model = models.wide_resnet50_2(pretrained=pt)

        self.feat = torch.nn.Sequential(*(list(zoo_model.children())[:-1]))

    def forward(self, x, pen=False):
        if self.vit_flag:
            out = self.feat(x).last_hidden_state[:,0,:]
        else:
            out = self.feat(x)
        out = out.view(out.size(0), -1)
        if pen:
            return out
        else:
            out = self.last(out)
            return out

 

def vit_pt_imnet(out_dim, block_division = None):
    return ResNetZoo(num_classes=out_dim, pt=True, mode=0)

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