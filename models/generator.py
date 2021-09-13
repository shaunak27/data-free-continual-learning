import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Generator(nn.Module):

    def __init__(self, in_channel=1, img_sz=32, hidden_dim=256, z_dim=100, bn = False, outdim=10):
        super(Generator, self).__init__()
        
        
        self.BN = bn
        self.in_dim = in_channel*img_sz*img_sz
        self.z_dim=z_dim
        enc_mlp_output_size = hidden_dim
        # reparametrization ("to Z and back")
        out_nl = True
        dec_mlp_input_size = hidden_dim
        self.fromZ = fc_layer(out_dim=dec_mlp_input_size, in_dim=z_dim+outdim, bn=self.BN)
        self.fcD = nn.Sequential(
            fc_layer(out_dim=hidden_dim, in_dim=hidden_dim, bn=self.BN),
            fc_layer(out_dim=self.in_dim, in_dim=hidden_dim, rl=False, bn=self.BN)
        )
        self.reshapeD = ToImage(image_channels=in_channel)

        self.n_classes = outdim
        self.eye = torch.eye(self.n_classes).to('cuda')

    def decode(self, z):
        '''Pass latent variable activations through feedback connections, to give reconstructed image [image_recon].'''
        hD = self.fromZ(z)
        image_features = self.fcD(hD)
        image_recon = self.reshapeD(image_features)
        return image_recon

    def sample(self, size, labels, kdim, rcode):
        
        # sample z
        z = torch.randn(size, self.z_dim)
        z = z.cuda()
        # X = self.decode(z)
        z = [z]
        code = self.eye[labels]
        # k_needed = self.n_classes - len(kdim)
        # code = code[:,kdim] + 0.1 * torch.randn(code.size(0), len(kdim)).cuda()
        # code = F.softmax(code, dim=1)
        # code = [rcode]
        # code.append(torch.zeros((len(labels),k_needed), requires_grad=True).cuda())
        # code = torch.cat(code, dim=1)
        z.append(code)
        X = self.decode(torch.cat(z, dim=1))

        # return samples as [batch_size]x[channels]x[image_size]x[image_size] tensor, plus classes-labels
        return X

class Reshape(nn.Module):
    '''A nn-module to reshape a tensor to a 4-dim "image"-tensor with [image_channels] channels.'''
    def __init__(self, image_channels):
        super().__init__()
        self.image_channels = image_channels

    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        image_size = int(np.sqrt(x.nelement() / (batch_size*self.image_channels)))
        return x.view(batch_size, self.image_channels, image_size, image_size)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(channels = {})'.format(self.image_channels)
        return tmpstr

class ToImage(nn.Module):
    '''Reshape input units to image with pixel-values between 0 and 1.

    Input:  [batch_size] x [in_units] tensor
    Output: [batch_size] x [image_channels] x [image_size] x [image_size] tensor'''

    def __init__(self, image_channels=1):
        super().__init__()
        # reshape to 4D-tensor
        self.reshape = Reshape(image_channels=image_channels)
        # put through sigmoid-nonlinearity
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.reshape(x)
        x = self.sigmoid(x)
        return x

    def image_size(self, in_units):
        '''Given the number of units fed in, return the size of the target image.'''
        image_size = np.sqrt(in_units/self.image_channels)
        return image_size

class fc_layer(nn.Module):
    def __init__(self, out_dim=10, in_dim = 1024, rl=True, bn=True):
        super().__init__()
        self.BN = bn
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear =  nn.Linear(in_dim, out_dim)
        if self.BN: self.bn = nn.BatchNorm1d(out_dim)
        if rl: self.rl =  nn.ReLU()
        else: self. rl = None

    def forward(self, x):
        x = self.linear(x)
        if self.BN:
            x = self.bn(x)
        if self.rl is not None:
            x = self.rl(x)
        return x

class GeneratorSimple(Generator):

    def __init__(self, in_channel, img_sz, kernel_num, z_size, bn = False, outdim=10):
        super(GeneratorSimple, self).__init__()
        
        self.BN = bn
        self.image_size = img_sz
        self.channel_num = in_channel
        self.kernel_num = kernel_num
        self.z_dim = z_size
        self.n_classes = outdim
        self.eye = torch.eye(self.n_classes).to('cuda')

        # Training related components that should be set before training
        # -criterion for reconstruction
        self.recon_criterion = None

        self.decoder = nn.Sequential(
            self._deconv(256, 128),
            self._deconv(128, 128, s=1),
            self._deconv(128, 128, s=1),
            self._deconv(128, 64),
            self._deconv(64, 64, s=1),
            self._deconv(64, 64, s=1),
            self._deconv(64, in_channel, ReLU=False),
            nn.Sigmoid()
        )

        self.feature_size = img_sz // 8
        self.feature_volume = self.kernel_num * (self.feature_size ** 2)

        # layers
        # encoder
        # self.encoder = nn.Sequential(
        #     self._conv(in_channel, kernel_num // 4),
        #     # self._conv(kernel_num // 8, kernel_num // 4),
        #     self._conv(kernel_num // 4, kernel_num // 2),
        #     self._conv(kernel_num // 2, kernel_num),
        # )

        # # encoded feature's size and volume
        # self.feature_size = img_sz // 16
        # self.feature_volume = kernel_num * (self.feature_size ** 2)

        # # decoder
        # self.decoder = nn.Sequential(
        #     self._deconv(kernel_num, kernel_num // 2),
        #     self._deconv(kernel_num // 2, kernel_num // 4),
        #     # self._deconv(kernel_num // 4, kernel_num // 8),
        #     self._deconv(kernel_num // 4, in_channel, ReLU=False),
        #     nn.Sigmoid()
        # )

        # q
        self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
        self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)

        # projection
        self.project = self._linear(z_size+self.n_classes, self.feature_volume, relu=False)

    def decode(self, z):
        '''Pass latent variable activations through feedback connections, to give reconstructed image [image_recon].'''
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected)

    def _conv(self, channel_size, kernel_num):
        if self.BN:
            return nn.Sequential(
                nn.Conv2d(
                    channel_size, kernel_num,
                    kernel_size=4, stride=2, padding=1,
                ),
                nn.BatchNorm2d(kernel_num),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    channel_size, kernel_num,
                    kernel_size=4, stride=2, padding=1,
                ),
                nn.ReLU(),
            )

    def _deconv(self, channel_num, kernel_num, ReLU=True, s=2):
        if ReLU:
            if self.BN:
                return nn.Sequential(
                    nn.ConvTranspose2d(
                        channel_num, kernel_num,
                        kernel_size=4, stride=s, padding=1,
                    ),
                    nn.BatchNorm2d(kernel_num),
                    nn.ReLU(),
                )
            else:
                return nn.Sequential(
                    nn.ConvTranspose2d(
                        channel_num, kernel_num,
                        kernel_size=4, stride=s, padding=1,
                    ),
                    nn.ReLU(),
                )
        else:
            if self.BN:
                return nn.Sequential(
                    nn.ConvTranspose2d(
                        channel_num, kernel_num,
                        kernel_size=4, stride=s, padding=1,
                    ),
                    nn.BatchNorm2d(kernel_num),
                )
            else:
                return nn.Sequential(
                    nn.ConvTranspose2d(
                        channel_num, kernel_num,
                        kernel_size=4, stride=s, padding=1,
                    ),
                )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)

def MLPAutoEncoder100(temp = 1, bn = False, outdim=10):
    return Generator(in_channel=1, img_sz=28, hidden_dim=100, outdim=outdim)


def MLPAutoEncoder400(temp = 1, bn = False, outdim=10):
    return Generator(in_channel=1, img_sz=28, hidden_dim=400, outdim=outdim)


def MLPAutoEncoder1000(temp = 1, bn = False, outdim=10):
    return Generator(in_channel=1, img_sz=28, hidden_dim=1000, outdim=outdim)


def MLPAutoEncoder2000(temp = 1, bn = False, outdim=10):
    return Generator(in_channel=1, img_sz=28, hidden_dim=2000, outdim=outdim)


def MLPAutoEncoder5000(temp = 1, bn = False, outdim=10):
    return Generator(in_channel=1, img_sz=28, hidden_dim=5000, outdim=outdim)

# def Autoencoder_cifar(temp = 1, bn = False, outdim=10):
#     return GeneratorSimple(in_channel=3, img_sz=32, kernel_num=256, z_size=128, outdim=outdim)

class GeneratorBig(nn.Module):
    def __init__(self, zdim, in_channel, img_sz):
        super(GeneratorBig, self).__init__()
        self.z_dim = zdim

        self.init_size = img_sz // (2**5)
        self.l1 = nn.Sequential(nn.Linear(zdim, 64*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(64),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks6 = nn.Sequential(
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 64, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks3(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks4(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks5(img)
        img = self.conv_blocks6(img)
        return img

    def sample(self, size):
        
        # sample z
        z = torch.randn(size, self.z_dim)
        z = z.cuda()
        X = self.forward(z)
        return X
class Generator(nn.Module):
    def __init__(self, zdim, in_channel, img_sz):
        super(Generator, self).__init__()
        self.z_dim = zdim

        self.init_size = img_sz // 4
        self.l1 = nn.Sequential(nn.Linear(zdim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def sample(self, size):
        
        # sample z
        z = torch.randn(size, self.z_dim)
        z = z.cuda()
        X = self.forward(z)
        return X

class GeneratorMed(nn.Module):
    def __init__(self, zdim, in_channel, img_sz):
        super(GeneratorMed, self).__init__()
        self.z_dim = zdim

        self.init_size = img_sz // 8
        self.l1 = nn.Sequential(nn.Linear(zdim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks3(img)
        return img

    def sample(self, size):
        
        # sample z
        z = torch.randn(size, self.z_dim)
        z = z.cuda()
        X = self.forward(z)
        return X

def Autoencoder_cifar(temp = 1, bn = False, outdim=10):
    return Generator(zdim=1000, in_channel=3, img_sz=32)

def Autoencoder_tinyimnet(temp = 1, bn = False, outdim=10):
    return GeneratorMed(zdim=1000, in_channel=3, img_sz=64)

def Autoencoder_imnet(temp = 1, bn = False, outdim=10):
    return GeneratorBig(zdim=1000, in_channel=3, img_sz=224)