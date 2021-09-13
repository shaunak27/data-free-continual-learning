import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms as transforms

import numpy as np
import math
import random
import collections
import gc

import copy

VOCAL = False

class Teacher(nn.Module):
    '''Teacher module for Deep Generative Replay (with two separate models).'''

    def __init__(self, solver, generator, gen_opt, img_shape, iters, class_idx, deep_inv_params, var, mean, student = True, train = True, config=None):
        '''Instantiate a new Teacher-object.

        [solver]:      <Solver> for classifying images'''

        super().__init__()
        self.solver = solver
        self.generator = generator
        self.gen_opt = gen_opt
        self.solver.eval()
        self.generator.eval()
        self.img_shape = img_shape
        self.iters = iters
        self.invert_with_student = student
        self.k_var = var
        self.k_mean = mean
        self.track_step = 0
        self.config = config

        # hyperparameters
        self.sig_scale = deep_inv_params[0] # 10.0 for adaptive deep inversion, 0 else
        self.di_lr = deep_inv_params[1]
        self.r_feature_weight = deep_inv_params[2]
        self.di_var_scale = deep_inv_params[3]
        self.di_l2_scale = deep_inv_params[4]
        self.content_weight = deep_inv_params[5]
        self.layer_scale = deep_inv_params[6]
        self.epoch_iter = int(deep_inv_params[8])

        self.backup_gen = copy.deepcopy(self.generator)

        # get class keys
        self.class_idx = list(class_idx)
        self.num_k = len(self.class_idx)

        # first time?
        self.first_time = train

        # set up criteria for optimization
        self.criterion = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.mse_loss = nn.MSELoss(reduction='none').cuda()
        self.smoothing = Gaussiansmoothing(3,5,1)

        # # All version
        # Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in self.solver.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                loss_r_feature_layers.append(DeepInversionFeatureHookC(module, 0, self.r_feature_weight)) # * (self.layer_scale ** self.solver.deep_inv_scale_key[module.num_features])))
        self.loss_r_feature_layers = loss_r_feature_layers


    def sample(self, size, device, student, student_head, num_k_new, return_scores=False, step=False):
        '''Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device as <self>.

        INPUT:  - [allowed_predictions] <list> of [class_ids] which are allowed to be predicted
                - [return_scores]       <bool>; if True, [y_hat] is also returned

        OUTPUT: - [X]     <4D-tensor> generated images
                - [y]     <1D-tensor> predicted corresponding labels
                - [y_hat] <2D-tensor> predicted "logits"/"scores" for all [allowed_predictions]'''
        
        # debug
        self.student_head = student_head

        # make sure solver is eval mode
        self.solver.eval()

        # use student?
        if not self.invert_with_student: student = None

        # step
        self.generator.train()
        if self.first_time:
            self.first_time = False
            self.get_images(bs=size, epochs=self.iters, idx=-1, net_student=student, num_k_new=num_k_new)
            self.track_step = 0

        # if updating
        if step and (self.sig_scale > 0 or self.sig_scale < 0 or self.di_l2_scale > 0) and self.track_step % self.epoch_iter == 0:
        # if step and self.track_step % self.epoch_iter == 0:
            self.generator.load_state_dict(self.backup_gen.state_dict())
            self.get_images(bs=size, epochs=self.epoch_iter, idx=-1, net_student=student, num_k_new=num_k_new)
        self.track_step += 1
        
        # sample images
        self.generator.eval()
        with torch.no_grad():
            x_i = self.generator.sample(size)
            # x_i = torch.clamp(x_i, 0.0, 1.0)

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x_i)
        y_hat = y_hat[:, self.class_idx]

        # get predicted class-labels (indexed according to each class' position in [self.class_idx]!)
        _, y = torch.max(y_hat, dim=1)
        # print(np.unique(y.cpu().detach().numpy()))

        return (x_i, y, y_hat) if return_scores else (x_i, y)

    def generate_scores(self, x, allowed_predictions=None, return_label=False):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        return (y, y_hat) if return_label else y_hat


    def generate_scores_pen(self, x):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        return y_hat

    def get_images(self, bs=256, epochs=1000, idx=-1, net_student=None, num_k_new=None):

        # clear cuda cache
        torch.cuda.empty_cache()
        
        # preventing backpropagation through student for Adaptive DeepInversion
        if net_student is not None:
            net_mode = net_student.training
            net_student.eval()

        self.generator.train()
        for epoch in range(epochs):

            # self.vocal = VOCAL and ((epoch+1) % 100 == 0 or epoch == 0)
            self.vocal = VOCAL and ((self.track_step +1) % 100 == 0 or self.track_step == 0)

            if self.vocal: print('***************')
            if self.vocal: print(epoch+1)

            inputs = self.generator.sample(bs)

            # foward with images
            self.gen_opt.zero_grad()
            self.solver.zero_grad()

            # clamp image
            # inputs_clamped = torch.clamp(inputs, 0.0, 1.0)
            inputs_clamped = inputs

            # content - new
            outputs = self.solver(inputs_clamped)[:,:self.num_k]
            loss = self.criterion(outputs / self.layer_scale, torch.argmax(outputs, dim=1)) * self.content_weight
            loss_target = loss.item()

            # class balance
            softmax_o_T = F.softmax(outputs, dim = 1).mean(dim = 0)
            loss += (1.0 + (softmax_o_T * torch.log(softmax_o_T) / math.log(self.num_k)).sum())
            if self.vocal: 
                print('class balance loss: ' + str(loss.item() - loss_target))

            # R_feature loss
            for mod in self.loss_r_feature_layers: 
                loss_distr = mod.r_feature * self.r_feature_weight / len(self.loss_r_feature_layers)
                if len(self.config['gpuid']) > 1:
                    loss_distr = loss_distr.to(device=torch.device('cuda:'+str(self.config['gpuid'][0])))
                loss = loss + loss_distr
            if self.vocal: 
                print('layer loss: ' + str(loss.item() - loss_target))

            # # ambiguity loss
            if net_student is not None:
                out_student = self.student_head(net_student.forward(x=inputs_clamped, pen=True))[:,:num_k_new]
                loss_verifier_amb = self.criterion(out_student, torch.argmax(out_student[:,self.num_k:num_k_new], dim=1))
                loss = loss + self.di_l2_scale * loss_verifier_amb
                if self.vocal: 
                    print('amg loss: ' + str(self.di_l2_scale * loss_verifier_amb.item()))

            # competition loss, Adaptive DeepInvesrion
            if self.sig_scale > 0.0 and net_student is not None:
                net_student.zero_grad()
                T = 3.0
                outputs_student = net_student(inputs_clamped)[:,:self.num_k]

                # jensen shanon divergence:
                # another way to force KL between negative probabilities
                P = F.softmax(outputs_student / T, dim=1)
                Q = F.softmax(outputs / T, dim=1)
                M = 0.5 * (P + Q)
                P = torch.clamp(P, 0.01, 0.99)
                Q = torch.clamp(Q, 0.01, 0.99)
                M = torch.clamp(M, 0.01, 0.99)
                eps = 0.0
                # loss_verifier_cig = 0.5 * self.kl_loss(F.log_softmax(outputs_verifier / T, dim=1), M) +  0.5 * self.kl_loss(F.log_softmax(outputs/T, dim=1), M)
                loss_verifier_cig = 0.5 * self.kl_loss(torch.log(P + eps), M) + 0.5 * self.kl_loss(torch.log(Q + eps), M)
                # JS criteria - 0 means full correlation, 1 - means completely different
                loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

                # update loss
                loss = loss + self.sig_scale * loss_verifier_cig

            # competition loss, Adaptive DeepInvesrion
            elif self.sig_scale < 0.0 and net_student is not None:

                net_student.zero_grad()
                T = 3.0
                outputs_student = net_student.forward(x=inputs_clamped, pen=True)
                loss_verifier_cig = torch.exp(-1 * self.mse_loss(outputs_student, self.solver.forward(x=inputs_clamped, pen=True)).sum(dim=1)/(out_student.size(1))).mean()
                loss_verifier_cig = loss_verifier_cig * -1.0
                loss = loss + self.sig_scale * loss_verifier_cig

            # # image prior - old
            # diff1 = inputs_clamped[:,:,:,:-1] - inputs_clamped[:,:,:,1:]
            # diff2 = inputs_clamped[:,:,:-1,:] - inputs_clamped[:,:,1:,:]
            # diff3 = inputs_clamped[:,:,1:,:-1] - inputs_clamped[:,:,:-1,1:]
            # diff4 = inputs_clamped[:,:,:-1,:-1] - inputs_clamped[:,:,1:,1:]
            # loss_var = (torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4))
            # loss_var /= (inputs_clamped.shape[1] * inputs_clamped.shape[2] * inputs_clamped.shape[3])
            # loss = loss + self.di_var_scale * loss_var

            # image prior - new
            inputs_clamped_smooth = self.smoothing(F.pad(inputs_clamped, (2, 2, 2, 2), mode='reflect'))
            loss_var = self.mse_loss(inputs_clamped, inputs_clamped_smooth).mean()
            loss = loss + self.di_var_scale * loss_var

            # backward pass
            loss.backward()
            
            if self.vocal: 
                # print('other-losses')
                print('loss target: ' + str(loss_target))
                print('loss var: ' + str(loss_var.item() * self.di_var_scale))
                if self.sig_scale != 0.0 and net_student is not None:
                    print('loss comp: ' + str(self.sig_scale * loss_verifier_cig.item()))
                print('loss total: ' + str(loss.item()))

            if self.vocal: print('***************')

            self.gen_opt.step()

        # clear cuda cache
        torch.cuda.empty_cache()
        self.generator.eval()
        if net_student is not None: net_student.train(mode=net_mode)

class DeepInversionFeatureHookC():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, gram_matrix_weight, layer_weight):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.target = None
        self.gram_matrix_weight = gram_matrix_weight
        self.layer_weight = layer_weight

    def hook_fn(self, module, input, output):

        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8
        r_feature = torch.log(var**(0.5) / (module.running_var.data.type(var.type()) + 1e-8)**(0.5)).mean() - 0.5 * (1.0 - (module.running_var.data.type(var.type()) + 1e-8 + (module.running_mean.data.type(var.type())-mean)**2)/var).mean()

        self.r_feature = r_feature

            
    def close(self):
        self.hook.remove()

class Gaussiansmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussiansmoothing, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).cuda()

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
