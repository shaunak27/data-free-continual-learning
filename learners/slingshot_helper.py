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

VOCAL = True

class Sampler(nn.Module):

    def __init__(self, solver, img_shape, iters, class_idx, deep_inv_params):
        '''Instantiate a new Teacher-object.
        [solver]:      <Solver> for classifying images'''

        super().__init__()
        self.solver = solver
        self.optimizer = None
        self.solver.eval()
        self.img_shape = img_shape
        self.iters = iters

        # hyperparameters
        self.r_feature_weight = deep_inv_params[0]
        self.sling_weight = deep_inv_params[1]
        self.di_lr = deep_inv_params[2]
        self.batch_size = deep_inv_params[3]
        self.inner_steps = deep_inv_params[4]
        self.var_scale = deep_inv_params[5]

        # get class keys
        self.class_idx = list(class_idx)
        self.num_k = len(self.class_idx)

        # set up criteria for optimization
        self.criterion = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.mse_loss = nn.MSELoss(reduction='none').cuda()
        self.smoothing = Gaussiansmoothing(3,5,1)

        # solver parameters
        teacher_params = {n: p for n, p in self.solver.named_parameters() if p.requires_grad}
        self.teacher_params = {}
        for n, p in teacher_params.items():
            self.teacher_params[n] = p.clone().detach()

        # Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in self.solver.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module, 0, self.r_feature_weight)) # * (self.layer_scale ** self.solver.deep_inv_scale_key[module.num_features])))
        self.loss_r_feature_layers = loss_r_feature_layers
            


    def train_samples(self, student, num_k_new):
        '''Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device as <self>.
        INPUT:  - [allowed_predictions] <list> of [class_ids] which are allowed to be predicted
                - [return_scores]       <bool>; if True, [y_hat] is also returned
        OUTPUT: - [X]     <4D-tensor> generated images
                - [y]     <1D-tensor> predicted corresponding labels
                - [y_hat] <2D-tensor> predicted "logits"/"scores" for all [allowed_predictions]'''
        
        # make sure solver is eval mode
        self.solver.eval()

        # init
        x_i = torch.randn((int(self.batch_size), self.img_shape[1], self.img_shape[2], self.img_shape[3]), requires_grad=True, device='cuda', dtype=torch.float)
        self.optimizer = Adam([x_i], lr=self.di_lr)
        
        # get inputs
        x_i = self.get_images(bs=self.batch_size, epochs=self.iters, net_student=student, num_k_new=num_k_new, inputs=x_i)

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x_i)
        y_hat = y_hat[:, self.class_idx]

        # get predicted class-labels (indexed according to each class' position in [self.class_idx]!)
        _, y = torch.max(y_hat, dim=1)

        return (x_i, y)

    def get_images(self, bs=256, epochs=1000, net_student=None, num_k_new=None, inputs=None, replay_t=None):
        
        # start
        self.optimizer.state = collections.defaultdict(dict)  # Reset state of optimizer

        # clear cuda cache
        torch.cuda.empty_cache()

        # preventing backpropagation through student for Adaptive DeepInversion
        if net_student is not None:
            net_mode = net_student.training
            net_student.eval()

        best_cost = 1e12
        
        for epoch in range(epochs):

            self.vocal = VOCAL and ((epoch+1) % 10 == 0 or epoch == 0)

            if self.vocal: print('***************')
            if self.vocal: print(epoch+1)

            # foward with images
            self.optimizer.zero_grad()
            self.solver.zero_grad()

            # clamp image
            # inputs_clamped = torch.clamp(inputs, 0.0, 1.0)
            inputs_clamped = inputs

            # content - new
            outputs = self.solver(inputs_clamped)[:,:self.num_k]
            
            # class balance
            softmax_o_T = F.softmax(outputs, dim = 1).mean(dim = 0)
            loss = (1.0 + (softmax_o_T * torch.log(softmax_o_T) / math.log(self.num_k)).sum())
            loss_bal = loss.item()

            # content - new
            loss_target = self.criterion(outputs, torch.argmax(outputs, dim=1))
            loss = loss + loss_target
            loss_target = loss_target.item()

            # R_feature loss
            for mod in self.loss_r_feature_layers: 
                loss_distr = mod.r_feature * self.r_feature_weight / len(self.loss_r_feature_layers)
                loss = loss + loss_distr
            loss_feat = loss.item() - loss_bal

            # prior
            inputs_clamped_smooth = self.smoothing(F.pad(inputs_clamped, (2, 2, 2, 2), mode='reflect'))
            loss_var = self.mse_loss(inputs_clamped, inputs_clamped_smooth).mean()
            loss = loss + self.var_scale * loss_var

            # slingshot loss
            if self.sling_weight > 0:

                # set up slingshot
                student_copy = copy.deepcopy(net_student)

                # perform slingshot
                student_params = {n: p for n, p in net_student.named_parameters() if p.requires_grad}
                for i in range(int(self.inner_steps)):

                    # get predicted logit-scores
                    with torch.no_grad():
                        y_hat = self.solver.forward(inputs_clamped)
                    y_hat = y_hat[:, self.class_idx]
                    _, y = torch.max(y_hat, dim=1)

                    # update model
                    if i == 0:
                        loss_inner, fast_weights = self.inner_forward(inputs_clamped, y, list(student_copy.named_parameters()), {n: p for n, p in student_copy.named_parameters()})
                    else:
                        loss_inner, fast_weights = self.inner_forward(inputs_clamped, y, fast_weights, {n: p for n, p in student_copy.named_parameters()})

                    # # update model
                    # # grads = torch.autograd.grad(loss_inner, student_copy.parameters(), create_graph=True)
                    # if i == 0:
                    #     grads = torch.autograd.grad(loss_inner, student_copy.parameters(), create_graph=True, allow_unused=True)
                    #     # fast_weights = {(name, param - self.di_lr*grad) for ((name, param), grad) in zip(student_params, grads)}
                    #     # fast_weights = list(map(lambda p: p[1] - self.di_lr * p[0], zip(grads, student_params)))
                    #     fast_weights = []
                    #     i_ = 0
                    #     for n,p in student_params.items():
                    #         fast_weights.append(p - self.di_lr[n] * grads[i_])
                    #         i_ += 1
                    # else:
                    if True:
                        grads = torch.autograd.grad(loss_inner, fast_weights, create_graph = True)
                        # fast_weights = {(name, param - self.di_lr*grad) for ((name, param), grad) in zip(fast_weights.items(), grads)}
                        # fast_weights = list(map(lambda p: p[1] - self.di_lr * p[0], zip(grads, fast_weights)))
                        for i_ in range(len(fast_weights)):
                            fast_weights[i_] = fast_weights[i_] - self.di_lr * grads[i_]

                loss_sling = None
                i_ = 0
                for n, p in self.teacher_params.items():
                    if loss_sling is None:
                        loss_sling = ((p - fast_weights[i_]) ** 2).sum()
                    else:
                        loss_sling += ((p - fast_weights[i_]) ** 2).sum()
                    i_ += 1
                
                # update loss
                loss = loss + loss_sling
                loss_sling = loss_sling.item()

            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs_clamped.data

            # backward pass
            loss.backward()
            
            if self.vocal: 
                print('loss target: ' + str(loss_target))
                print('loss balance: ' + str(loss_bal))
                print('loss features: ' + str(loss_feat))
                if self.sling_weight > 0:
                    print('loss slingshot: ' + str(loss_sling))
                print('loss total: ' + str(loss.item()))
            if self.vocal: print('***************')
            self.optimizer.step()

        # clear cuda cache
        torch.cuda.empty_cache()
        if net_student is not None: net_student.train(mode=net_mode)
        return best_inputs

    def inner_forward(self, in_, target, weights, params):
        ''' Run data through net, return loss and output '''
        input_var = in_
        target_var = torch.autograd.Variable(target).cuda(async=True)
        
        # Hard way to implement
        # out = self.net_forward(input_var, weights, params)[:, self.class_idx]
        
        # can this work?
        solver_copy = copy.deepcopy(self.solver)
        i_ = 0
        for n,p in solver_copy.named_parameters():
            p = weights[i_]
            i_ += 1
        out = solver_copy.forward(input_var)[:, self.class_idx]
        
        loss = self.criterion(out, target_var)

        fast_weights = []
        for n,p in solver_copy.named_parameters():
            fast_weights.append(p)
        return loss, fast_weights

    def net_forward(self, x, vars, params, bn_training=False):

        idx = 0
        bn_idx = 0

        for name, param in params.items():
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError
        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

class DeepInversionFeatureHook():
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
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False) # + 1e-8

        # kl divergence
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