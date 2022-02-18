
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from . import modified_linear
import copy


from torch.autograd import Variable

__all__ = ['ResNet', 'resnet18', 'resnet32']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class View(nn.Module):
    def __init__(self, dim):
        super(View, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = F.avg_pool2d(x, self.dim)
        return torch.flatten(x, start_dim=1)

class LBlock(nn.Module):
    def __init__(self, block, num_blocks, num_classes, planes):
        super(LBlock, self).__init__()
        self.in_planes = planes
        self.l1 = self._make_layer(block, planes, num_blocks[0], stride=1)
        self.l2 = self._make_layer(block, planes*2, num_blocks[1], stride=2)
        self.l3 = self._make_layer(block, planes*4, num_blocks[2], stride=2)
        self.last = nn.Linear(planes*4, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return self.last(out)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, block_division=16, cc=False):
        super(ResNet, self).__init__()
        self.block_division = block_division

        in_planes = 16*4
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        
        self.last = 1
        model_blocks = []
        for b in range(block_division):
            layer_blocks = LBlock(block, num_blocks, num_classes, int(in_planes / block_division))
            model_blocks.append(layer_blocks)

        self.layer1 = nn.ModuleList(model_blocks)
        self.apply(_weights_init)


    def features(self, x, div):
        out1 = F.relu(self.bn1(self.conv1(x)))

        # layer 1
        bd = out1.size()[1]
        bd_i = 0
        out2 = self.layer1[0](out1[:,bd_i:bd_i+int(bd/self.block_division)])
        out2 = out2
        for l in range(1,self.block_division):
            bd_i += int(bd/self.block_division)
            bout = self.layer1[l](out1[:,bd_i:bd_i+int(bd/self.block_division)])
            out2 = out2 + bout

        self.size_array = [out1.size(), out2.size()]

        if div:
            return [out1.view(out1.size(0), -1)], out2
        else:
            return out2

    def forward(self, x, div=False, pen=False):
        x = self.features(x, div)
        if div:
            return x[0], x[1], self.block_division
        else:
            return x


    def penultimate(self, x):
        x = self.features(x)
        return x

    def get_layer(self, l):

        if l == 1:
            return self.layer1

    def get_layer_forward(self, l, x):
        if l == 1:
            out = self.layer1[0](x)
            for l in range(1,self.block_division):
                out = torch.cat([out, self.layer1[l](x)], dim=1)
            return out


def resnet32(out_dim, block_division=8):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=out_dim, block_division=block_division)

def resnet32_cc(out_dim, block_division=8):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=out_dim, cc=True, block_division=block_division)

