
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

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, block_division=16, cc=False):
        super(ResNet, self).__init__()
        self.block_division = block_division

        in_planes = 16*2
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.in_planes = in_planes
        self.layer1_pre = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1, bd=block_division)
        self.layer2_pre = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2, bd=block_division)
        self.layer3_pre = nn.Conv2d(in_planes*2, in_planes*2, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2, bd=block_division)
        
        # final layer
        if cc:
            self.last = modified_linear.CosineLinear(in_planes*4, num_classes)
        else:
            self.last = nn.Linear(in_planes*4, num_classes)

        self.last_0 = nn.Linear(32, num_classes)
        self.last_1 = nn.Linear(32, num_classes)
        self.last_2 = nn.Linear(64, num_classes)

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

    def forward_h(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out1_ = F.avg_pool2d(out1, out1.size()[3])
        out1 = self.layer1_pre(out1)
        

        # layer 1
        out2 = self.layer1[0](out1)
        for l in range(1,self.block_division):
            out2 = torch.cat([out2, self.layer1[l](out1)], dim=1)
        out2_ = F.avg_pool2d(out2, out2.size()[3])
        out2 = self.layer2_pre(out2)
        

        # layer 2
        out3 = self.layer2[0](out2)
        for l in range(1,self.block_division):
            a = self.layer2[l](out2)
            out3 = torch.cat([out3, self.layer2[l](out2)], dim=1)
        out3_ = F.avg_pool2d(out3, out3.size()[3])
        out3 = self.layer3_pre(out3)
        

        # layer 3
        out4 = self.layer3[0](out3)
        for l in range(1,self.block_division):
            out4 = torch.cat([out4, self.layer3[l](out3)], dim=1)
        out4_ = F.avg_pool2d(out4, out4.size()[3])

        out_0 = out1_.view(out1_.size(0), -1)
        out_1 = out2_.view(out2_.size(0), -1)
        out_2 = out3_.view(out3_.size(0), -1)
        out = out4_.view(out4_.size(0), -1)
        return [out_0, out_1, out_2, out]

    def features(self, x, div):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1_pre(out1)

        # layer 1
        out2 = self.layer1[0](out1)
        for l in range(1,self.block_division):
            out2 = torch.cat([out2, self.layer1[l](out1)], dim=1)
        out2 = self.layer2_pre(out2)

        # layer 2
        out3 = self.layer2[0](out2)
        for l in range(1,self.block_division):
            a = self.layer2[l](out2)
            out3 = torch.cat([out3, self.layer2[l](out2)], dim=1)
        out3 = self.layer3_pre(out3)

        # layer 3
        out4 = self.layer3[0](out3)
        for l in range(1,self.block_division):
            out4 = torch.cat([out4, self.layer3[l](out3)], dim=1)
        out4 = F.avg_pool2d(out4, out4.size()[3])

        self.size_array = [out1.size(), out2.size(), out3.size()]

        if div:
            return [out1.view(out1.size(0), -1), out2.view(out2.size(0), -1), out3.view(out3.size(0), -1)], out4.view(out4.size(0), -1)
        else:
            return out4.view(out4.size(0), -1)

    def logits_h(self, x, detach = True):
        if detach:
            return [self.last_0(x[0].detach()), self.last_1(x[1].detach()), self.last_2(x[2].detach()), self.last(x[3])]
        else:
            return [self.last_0(x[0]), self.last_1(x[1]), self.last_2(x[2]), self.last(x[3])]

    def pen_forward_helped(self, x, helper):
        out1 = F.relu(helper.bn1(helper.conv1(x)))
        out1 = self.layer1_pre(out1)

        # layer 1
        out2 = helper.layer1[0](out1)
        for l in range(1,helper.block_division):
            out2 = torch.cat([out2, helper.layer1[l](out1)], dim=1)
        out2 = self.layer2_pre(out2)

        # layer 2
        out3 = helper.layer2[0](out2)
        for l in range(1,helper.block_division):
            a = helper.layer2[l](out2)
            out3 = torch.cat([out3, helper.layer2[l](out2)], dim=1)
        out3 = self.layer3_pre(out3)

        # layer 3
        out4 = helper.layer3[0](out3)
        for l in range(1,helper.block_division):
            out4 = torch.cat([out4, helper.layer3[l](out3)], dim=1)
        out4 = F.avg_pool2d(out4, out4.size()[3])

        return out4.view(out4.size(0), -1)

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x, pen=False, div=False, l_freeze=-1):
        x = self.features(x, div)
        if pen and div:
            return x[0], x[1], self.block_division
        elif pen:
            return x
        else:
            x = self.logits(x)
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

    def get_linear_layer(self, l):

        if l == 0:
            return self.last_0
        elif l == 1:
            return self.last_1

        elif l == 2:
            return self.last_2

        elif l == 3:
            return self.last

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

def resnet32(out_dim, block_division=8):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=out_dim, block_division=block_division)

def resnet32_cc(out_dim, block_division=8):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=out_dim, cc=True, block_division=block_division)
