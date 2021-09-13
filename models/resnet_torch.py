import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import CosineScaling, CosineSimilarity
import math
from torch.nn import init
from torch.nn import functional as F
from . import modified_linear

KL = False

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, droprate=0):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.drop = nn.Dropout(p=droprate) if droprate>0 else None
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        if self.drop is not None:
            out = self.drop(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, droprate=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3, bn = False):
        super(PreActResNet, self).__init__()
        self.BN = bn
        self.in_planes = 64
        last_planes = 512*block.expansion

        self.conv1 = conv3x3(in_channels, 64)
        self.stage1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.stage2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.stage3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.stage4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if self.BN: self.bn_last = nn.BatchNorm2d(last_planes)
        self.last = nn.Linear(last_planes, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        return out

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        out = self.logits(self.penultimate(x))
        return x

    def penultimate(self, x):
        x = self.features(x)
        if self.BN:
            x = F.relu(self.bn_last(x))
        else:
            x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return x

class PreActResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, filters, num_classes=10, droprate=0, bn = False, cosine=False):
        super(PreActResNet_cifar, self).__init__()
        self.BN = bn
        self.in_planes = 16
        last_planes = filters[2]*block.expansion
        self.feature_layer_idx = [1,2,3,4,-1]

        self.conv1 = conv3x3(3, self.in_planes)
        self.stage1 = self._make_layer(block, filters[0], num_blocks[0], stride=1, droprate=droprate)
        self.stage2 = self._make_layer(block, filters[1], num_blocks[1], stride=2, droprate=droprate)
        self.stage3 = self._make_layer(block, filters[2], num_blocks[2], stride=2, droprate=droprate)
        if self.BN: self.bn_last = nn.BatchNorm2d(last_planes)

        if cosine:
            self.last = CosineScaling(last_planes, num_classes)
        else:
            self.last = nn.Linear(last_planes, num_classes)

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()
        """

    def _make_layer(self, block, planes, num_blocks, stride, droprate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, droprate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return out

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        out = self.logits(self.penultimate(x))
        return out

    def penultimate(self, x):
        out = self.features(x)
        if self.BN:
            out = F.relu(self.bn_last(out))
        else:
            out = F.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        if KL: out = out.softmax(dim=-1)
        return out

    def feature_forward(self, x):
        out1 = self.conv1(x)
        out2 = self.stage1(out1)
        out3 = self.stage2(out2)
        out4 = self.stage3(out3)
        if self.BN:
            out4 = F.relu(self.bn_last(out4))
        else:
            out4 = F.relu(out4)
        out4 = F.avg_pool2d(out4, 8)
        out5 = out4.view(out4.size(0), -1)
        if KL: out5 =out5.softmax(dim=-1)
        out5 = self.logits(out5)
        return {1:out1.view(out1.size(0), -1), 2:out2.view(out2.size(0), -1), 3:out3.view(out3.size(0), -1), 4:out4.view(out4.size(0), -1), -1:out5.view(out5.size(0), -1)}

class PreActResNet_imnet(nn.Module):
    def __init__(self, block, num_blocks, filters, num_classes=10, droprate=0, bn = False, cosine=False):
        super(PreActResNet_imnet, self).__init__()
        self.BN = bn
        self.in_planes = 16
        last_planes = 512
        self.feature_layer_idx = [1,2,3,4,-1]

        self.conv1 = conv3x3(3, self.in_planes)
        self.stage1 = self._make_layer(block, filters[0], num_blocks[0], stride=1, droprate=droprate)
        self.stage2 = self._make_layer(block, filters[1], num_blocks[1], stride=2, droprate=droprate)
        self.stage3 = self._make_layer(block, filters[2], num_blocks[2], stride=2, droprate=droprate)
        if self.BN: self.bn_last = nn.BatchNorm2d(last_planes)

        if cosine:
            self.last = CosineScaling(last_planes, num_classes)
        else:
            self.last = nn.Linear(last_planes, num_classes)

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()
        """

    def _make_layer(self, block, planes, num_blocks, stride, droprate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, droprate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return out

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x, pen=False):
        out = self.penultimate(x)
        if pen:
            return out
        else:
            out = self.logits(out)
            return out

    def penultimate(self, x):
        out = self.features(x)
        if self.BN:
            out = F.relu(self.bn_last(out))
        else:
            out = F.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        if KL: out = out.softmax(dim=-1)
        return out

    def feature_forward(self, x):
        out1 = self.conv1(x)
        out2 = self.stage1(out1)
        out3 = self.stage2(out2)
        out4 = self.stage3(out3)
        if self.BN:
            out4 = F.relu(self.bn_last(out4))
        else:
            out4 = F.relu(out4)
        out4 = F.avg_pool2d(out4, 8)
        out5 = out4.view(out4.size(0), -1)
        if KL: out5 =out5.softmax(dim=-1)
        out5 = self.logits(out5)
        return {1:out1.view(out1.size(0), -1), 2:out2.view(out2.size(0), -1), 3:out3.view(out3.size(0), -1), 4:out4.view(out4.size(0), -1), -1:out5.view(out5.size(0), -1)}

# ResNet for Cifar10/100 or the dataset with image size 32x32
def WideResNet_28_2_cifar(out_dim=10, bn = False):
    return PreActResNet_cifar(PreActBlock, [4, 4, 4], [32, 64, 128], num_classes=out_dim)

def WideResNet_28_2_imagenet(out_dim=100, bn = False):
    return PreActResNet_imnet(PreActBlock, [4, 4, 4], [32, 64, 128], num_classes=out_dim)

def WideResNet_28_2_drop_cifar(out_dim=10, bn = False):
    return PreActResNet_cifar(PreActBlock, [4, 4, 4], [32, 64, 128], num_classes=out_dim, droprate=0.3)

def WideResNet_28_2_cifar_cs(out_dim=10, bn = False, cosine = True):
    return PreActResNet_cifar(PreActBlock, [4, 4, 4], [32, 64, 128], num_classes=out_dim)

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1, last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.last = last
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # if not self.last: #remove ReLU in the last layer
        #     out = F.relu(out)
        out = F.relu(out)
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cosine=False, zdim=64):
        super(ResNet, self).__init__()
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, last_phase=cosine)

        zdim = 512*block.expansion
        if cosine:
            self.last = modified_linear.CosineLinear(zdim, num_classes)
        else:
            self.last = nn.Linear(zdim, num_classes)

        # self.almost_last = nn.Linear(512*block.expansion, zdim)
        # self.almost_bn = nn.BatchNorm1d(zdim)
        # if cosine:
        #     self.last = modified_linear.CosineLinear(zdim, num_classes)
        # else:
        #     self.last = nn.Linear(zdim, num_classes)

        
        # deep inversion scale
        self.deep_inv_scale = {64:1, 128:.5, 256:.05, 512:.005}
        self.deep_inv_scale_key = {64:0, 128:1, 256:2, 512:3}


 
    def _make_layer(self, block, planes, num_blocks, stride, last_phase=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        if last_phase:
            for i in range(len(strides)-1):
                stride = strides[i]
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            stride = strides[-1]
            layers.append(block(self.in_planes, planes, stride, last=True))
            self.in_planes = planes * block.expansion
        else:
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, out_feature=False):
        x = self.conv1(x)

        x = self.bn1(x)
        out = F.relu(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        # feature = F.relu(self.almost_bn(self.almost_last(feature)))
        out = self.last(feature)
        if out_feature == False:
            return out
        else:
            return out,feature

    def pre_scale_forward(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        out = F.relu(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        # feature = F.relu(self.almost_bn(self.almost_last(feature)))
        return self.last.fc_only(feature)

    def penultimate(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        out = F.relu(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # out = F.relu(self.almost_bn(self.almost_last(out)))
        if KL: out = out.softmax(dim=-1)
        return out

# def ResNet32(out_dim=10, bn = False):
#     return ResNet(BasicBlock, [5,5,5], out_dim)

# def ResNet34(out_dim=10, bn = False):
#     return ResNet(BasicBlock, [3,4,6,3], out_dim)

# def ResNet34_balanced(out_dim=10, bn = False):
#     return ResNet(BasicBlock, [3,4,6,3], out_dim, cosine = True)

# def ResNet18(out_dim=10, bn = False):
#     return ResNet(BasicBlock, [2,2,2,2], out_dim)

# def ResNet18_balanced(out_dim=10, bn = False):
#     return ResNet(BasicBlock, [2,2,2,2], out_dim, cosine = True)