
'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from . import modified_linear


from torch.autograd import Variable

__all__ = ['ResNet', 'resnet18', 'resnet32']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
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

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 =  nn.Identity(planes) # nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.Identity(planes) #nn.BatchNorm2d(planes)
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))

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
                     # nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = F.relu(out) 
        out = torch.tanh(out)
        out = self.alpha * out + self.beta
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, block_division=16, cc=False):
        super(ResNet, self).__init__()
        self.block_division = block_division

        in_planes = 16

        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.Identity(in_planes) #nn.BatchNorm2d(in_planes)
        self.in_planes = in_planes
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1, bd=block_division)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2, bd=block_division)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2, bd=block_division)
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))
        
        # final layer
        if cc:
            self.last = modified_linear.CosineLinear(in_planes*4, num_classes)
        else:
            self.last = nn.Linear(in_planes*4, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, bd):
        layer_blocks = []
        in_planes_hold = self.in_planes
        stride_hold = stride
        for bi in range(bd):
            self.in_planes = in_planes_hold
            stride = stride_hold
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, int(planes/bd), stride))
                self.in_planes = int(planes/bd) * block.expansion

            layer_blocks.append(nn.Sequential(*layers))
        self.in_planes = self.in_planes * bd
        return nn.ModuleList(layer_blocks)

    def features(self, x):
        # out1 = F.relu(self.bn1(self.conv1(x)))
        out1 = torch.tanh(self.bn1(self.conv1(x)))
        out1 = self.alpha * out1 + self.beta

        # layer 1
        out2 = self.layer1[0](out1)
        for l in range(1,self.block_division):
            out2 = torch.cat([out2, self.layer1[l](out1)], dim=1)

        # layer 2
        out3 = self.layer2[0](out2)
        for l in range(1,self.block_division):
            a = self.layer2[l](out2)
            out3 = torch.cat([out3, self.layer2[l](out2)], dim=1)

        # layer 3
        out4 = self.layer3[0](out3)
        for l in range(1,self.block_division):
            out4 = torch.cat([out4, self.layer3[l](out3)], dim=1)
        out4 = F.avg_pool2d(out4, out4.size()[3])

        self.size_array = [out1.size(), out2.size(), out3.size(), out4.size()]

        return [out1.view(out1.size(0), -1), out2.view(out2.size(0), -1), out3.view(out3.size(0), -1), out4.view(out4.size(0), -1)]

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x, pen=False, div=False):
        x = self.features(x)
        if pen and div:
            return x, self.block_division
        elif pen:
            return x[-1]
        else:
            x = self.logits(x[-1])
            return x

    def penultimate(self, x):
        x = self.features(x)
        return x

    def get_layer(self, l):

        if l == 1:
            return self.layer1

        elif l == 2:
            return self.layer2

        elif l == 3:
            return self.layer3

    def get_layer_forward(self, l, x):
        if l == 1:
            out = self.layer1[0](x)
            for l in range(1,self.block_division):
                out = torch.cat([out, self.layer1[l](x)], dim=1)
            return out

        elif l == 2:
            out = self.layer2[0](x)
            for l in range(1,self.block_division):
                out = torch.cat([out, self.layer2[l](x)], dim=1)
            return out

        elif l == 3:
            out = self.layer3[0](x)
            for l in range(1,self.block_division):
                out = torch.cat([out, self.layer3[l](x)], dim=1)
            out = F.avg_pool2d(out, out.size()[3])
            return out

def resnet32(out_dim):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=out_dim)

def resnet18(out_dim):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=out_dim)

def resnet32_cc(out_dim):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=out_dim, cc=True)

def resnet18_cc(out_dim):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=out_dim, cc=True)