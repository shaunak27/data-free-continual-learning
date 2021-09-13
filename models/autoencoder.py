import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):

    def __init__(self, hallucinate=False, in_channel=1, img_sz=32, hidden_dim=256, z_dim=100, bn = False):
        super(AutoEncoder, self).__init__()
        self.BN = bn
        self.in_dim = in_channel*img_sz*img_sz
        self.z_dim=z_dim
        # Training related components that should be set before training
        # -criterion for reconstruction
        self.recon_criterion = None
        # -weigths of different components of the loss function
        self.lamda_rcl = 1.
        self.lamda_vl = 1.
        self.flatten = Flatten()
        self.fcE = nn.Sequential(fc_layer(out_dim=hidden_dim, in_dim=self.in_dim, bn=self.BN),
        fc_layer(out_dim=hidden_dim, in_dim=hidden_dim, bn=self.BN)
        )
        enc_mlp_output_size = hidden_dim
        # reparametrization ("to Z and back")
        out_nl = True
        dec_mlp_input_size = hidden_dim
        self.toZ = nn.Linear(enc_mlp_output_size, z_dim)       # estimating mean
        self.toZlogvar = nn.Linear(enc_mlp_output_size, z_dim) # estimating log(SD**2)
        self.fromZ = fc_layer(out_dim=dec_mlp_input_size, in_dim=z_dim, bn=self.BN)
        self.fcD = nn.Sequential(
            fc_layer(out_dim=hidden_dim, in_dim=hidden_dim, bn=self.BN),
            fc_layer(out_dim=self.in_dim, in_dim=hidden_dim, rl=False, bn=self.BN)
        )
        self.reshapeD = ToImage(image_channels=in_channel)
        self.hallucinate = hallucinate


    def reparameterize(self, mu, logvar):
        '''Perform "reparametrization trick" to make these stochastic variables differentiable.'''
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def decode(self, z):
        '''Pass latent variable activations through feedback connections, to give reconstructed image [image_recon].'''
        hD = self.fromZ(z)
        image_features = self.fcD(hD)
        image_recon = self.reshapeD(image_features)
        return image_recon

    def encode(self, x):
            '''Pass input through feed-forward connections, to get [hE], [z_mean] and [z_logvar].'''
            # extract final hidden features (forward-pass)
            hE = self.fcE(self.flatten(x))
            # get parameters for reparametrization
            z_mean = self.toZ(hE)
            z_logvar = self.toZlogvar(hE)
            return z_mean, z_logvar, hE

    def forward(self, x):
        '''Forward function to propagate [x] through the encoder, reparametrization and decoder.

        Input:  - [x]   <4D-tensor> of shape [batch_size]x[channels]x[image_size]x[image_size]

        Output:
        - [x_recon]     <4D-tensor> reconstructed image (features) in same shape as [x]
        - [mu]          <2D-tensor> with either [z] or the estimated mean of [z]
        - [logvar]      None or <2D-tensor> estimated log(SD^2) of [z]
        - [z]           <2D-tensor> reparameterized [z] used for reconstruction'''
        # encode (forward), reparameterize and decode (backward)
        mu, logvar, hE = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        x_recon = self.decode(z)
        return (x_recon, mu, logvar, z)

    def sample(self, size):
            '''Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device as <self>.

            OUTPUT: - [X]     <4D-tensor> generated images'''

            # set model to eval()-mode
            mode = self.training
            self.eval()
            # sample z
            z = torch.randn(size, self.z_dim)
            z = z.cuda()
            with torch.no_grad():
                X = self.decode(z)
            # set model back to its initial mode
            self.train(mode=mode)
            # return samples as [batch_size]x[channels]x[image_size]x[image_size] tensor, plus classes-labels
            return X

    def loss_function(self, recon_x, x, dw, mu=None, logvar=None, y_hat=None, targets=None):
        '''Calculate and return various losses that could be used for training and/or evaluating the model.

        INPUT:  - [x_recon]         <4D-tensor> reconstructed image in same shape as [x]
                - [x]               <4D-tensor> original image
                - [y_hat]           <2D-tensor> with predicted "logits" for each class
                - [y_target]        <1D-tensor> with target-classes (as integers)
                - [scores]          <2D-tensor> with target "logits" for each class
                - [mu]              <2D-tensor> with either [z] or the estimated mean of [z]
                - [logvar]          None or <2D-tensor> with estimated log(SD^2) of [z]

        OUTPUT: - [reconL]       reconstruction loss indicating how well [x] and [x_recon] match
                - [variatL]      variational (KL-divergence) loss "indicating how normally distributed [z] is"'''

        batch_size = x.size(0)


        # debug version
        # average = True
        # reconL = F.binary_cross_entropy(input=recon_x.view(batch_size, -1), target=x.view(batch_size, -1),
        #                                 reduction='none')
        # reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)
        # reconL = torch.mean(reconL * dw)
        # variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        # variatL = torch.mean(variatL)                             #-> average over batch
        # if average:
        #     variatL /= (self.in_dim )

        ###-----Reconstruction loss-----###
        reconL = (self.recon_criterion(input=recon_x.view(batch_size, -1), target=x.view(batch_size, -1))).mean(dim=1)
        if self.hallucinate:
            reconL += self.recon_criterion_b(y_hat, targets.long()) * 0.5
        reconL = torch.mean(reconL * dw)

        ###-----Variational loss-----###
        if logvar is not None:
            #---- see Appendix B from: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 ----#
            variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            # -normalise by same number of elements as in reconstruction
            variatL /= self.in_dim
            # --> because self.recon_criterion averages over batch-size but also over all pixels/elements in recon!!

        else:
            variatL = torch.tensor(0.)
            variatL = variatL.cuda()
        
        # Return a tuple of the calculated losses
        return reconL, variatL

    def train_batch(self, x, y, data_weights, model, allowed_predictions):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)'''

        # Set model to training-mode
        self.train()

        ##--(1)-- CURRENT DATA --##
        precision = 0.
        # Run the model
        recon_batch,  mu, logvar, z = self.forward(x)

        # get predictions from classifier if hallucinating
        if self.hallucinate:
            y_hat = model.forward(recon_batch)
            y_hat = y_hat[:, allowed_predictions]
        else:
            y_hat = None

        # Calculate all losses
        reconL, variatL = self.loss_function(recon_x=recon_batch, x=x, dw = data_weights, mu=mu, logvar=logvar, y_hat=y_hat, targets=y)

        # Weigh losses as requested
        loss_total = self.lamda_rcl*reconL + self.lamda_vl*variatL

        # perform update
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        return loss_total.detach()

class Flatten(nn.Module):
    '''A nn-module to flatten a multi-dimensional tensor to 2-dim tensor.'''
    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr

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

class AutoEncoderSimple(AutoEncoder):

    def __init__(self, in_channel, img_sz, kernel_num, z_size, bn = False):
        super(AutoEncoderSimple, self).__init__()
        self.BN = bn
        self.image_size = img_sz
        self.channel_num = in_channel
        self.kernel_num = kernel_num
        self.z_dim = z_size

        # Training related components that should be set before training
        # -criterion for reconstruction
        self.recon_criterion = None

        if img_sz == 84:
            self.encoder = nn.Sequential(
                self._conv(in_channel, 64, kernel_size_=4, stride_=3),
                self._conv(64, 128),
                self._conv(128, 512),
            )

            self.decoder = nn.Sequential(
                self._deconv(512, 256),
                self._deconv(256, 64),
                self._deconv(64, in_channel, ReLU=False, kernel_size_=5, stride_=3),
                nn.Sigmoid()
            )
            self.feature_size = img_sz // 12
        else:
            self.encoder = nn.Sequential(
                self._conv(in_channel, 64),
                self._conv(64, 128),
                self._conv(128, 512),
            )

            self.decoder = nn.Sequential(
                self._deconv(512, 256),
                self._deconv(256, 64),
                self._deconv(64, in_channel, ReLU=False),
                nn.Sigmoid()
            )
            self.feature_size = img_sz // 8

        
        self.kernel_num = 512
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
        self.project = self._linear(z_size, self.feature_volume, relu=False)

        

        
        

    def decode(self, z):
        '''Pass latent variable activations through feedback connections, to give reconstructed image [image_recon].'''
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected)

    def encode(self, x):
        '''Pass input through feed-forward connections, to get [hE], [z_mean] and [z_logvar].'''
        # encode x
        encoded = self.encoder(x)
        # sample latent code z from q given x.
        z_mean, z_logvar = self.q(encoded)
        return z_mean, z_logvar, encoded

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def _conv(self, channel_size, kernel_num, kernel_size_=4, stride_=2):
        if self.BN:
            return nn.Sequential(
                nn.Conv2d(
                    channel_size, kernel_num,
                    kernel_size=kernel_size_, stride=stride_, padding=1,
                ),
                nn.BatchNorm2d(kernel_num),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    channel_size, kernel_num,
                    kernel_size=kernel_size_, stride=stride_, padding=1,
                ),
                nn.ReLU(),
            )

    def _deconv(self, channel_num, kernel_num,ReLU=True, kernel_size_=4, stride_=2):
        if ReLU:
            if self.BN:
                return nn.Sequential(
                    nn.ConvTranspose2d(
                        channel_num, kernel_num,
                        kernel_size=kernel_size_, stride=stride_, padding=1,
                    ),
                    nn.BatchNorm2d(kernel_num),
                    nn.ReLU(),
                )
            else:
                return nn.Sequential(
                    nn.ConvTranspose2d(
                        channel_num, kernel_num,
                        kernel_size=kernel_size_, stride=stride_, padding=1,
                    ),
                    nn.ReLU(),
                )
        else:
            if self.BN:
                return nn.Sequential(
                    nn.ConvTranspose2d(
                        channel_num, kernel_num,
                        kernel_size=kernel_size_, stride=stride_, padding=1,
                    ),
                    nn.BatchNorm2d(kernel_num),
                )
            else:
                return nn.Sequential(
                    nn.ConvTranspose2d(
                        channel_num, kernel_num,
                        kernel_size=kernel_size_, stride=stride_, padding=1,
                    ),
                )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)

def MLPAutoEncoder100(hallucinate=False, bn = False):
    return AutoEncoder(hallucinate, in_channel=1, img_sz=28, hidden_dim=100)


def MLPAutoEncoder400(hallucinate=False, bn = False):
    return AutoEncoder(hallucinate, in_channel=1, img_sz=28, hidden_dim=400)


def MLPAutoEncoder1000(hallucinate=False, bn = False):
    return AutoEncoder(hallucinate, in_channel=1, img_sz=28, hidden_dim=1000)


def MLPAutoEncoder2000(hallucinate=False, bn = False):
    return AutoEncoder(hallucinate, in_channel=1, img_sz=28, hidden_dim=2000)


def MLPAutoEncoder5000(hallucinate=False, bn = False):
    return AutoEncoder(hallucinate, in_channel=1, img_sz=28, hidden_dim=5000)

def Autoencoder_cifar(bn = False):
    return AutoEncoderSimple(in_channel=3, img_sz=32, kernel_num=512, z_size=1024)

def Autoencoder_imnet(bn = False):
    return AutoEncoderSimple(in_channel=3, img_sz=84, kernel_num=512, z_size=1024)