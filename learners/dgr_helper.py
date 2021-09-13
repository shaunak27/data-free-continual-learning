import torch
from torch import nn
import random
import math
import numpy as np
import torch.nn.functional as F

class Scholar(nn.Module):
    '''Scholar module for Deep Generative Replay (with two separate models).'''

    def __init__(self, generator, solver, stats = None, class_idx = None, temp = None):
        '''Instantiate a new Scholar-object.

        [generator]:   <Generator> for generating images from previous tasks
        [solver]:      <Solver> for classifying images'''

        super().__init__()
        self.generator = generator
        self.solver = solver
        self.stats = stats
        self.Dtemp = temp

        # get class keys
        if class_idx is not None:
            self.class_idx = list(class_idx)
            self.layer_idx = list(self.stats.keys())
            self.num_k = len(self.class_idx)


    def sample(self, size, allowed_predictions=None, return_scores=False):
        '''Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device as <self>.

        INPUT:  - [allowed_predictions] <list> of [class_ids] which are allowed to be predicted
                - [return_scores]       <bool>; if True, [y_hat] is also returned

        OUTPUT: - [X]     <4D-tensor> generated images
                - [y]     <1D-tensor> predicted corresponding labels
                - [y_hat] <2D-tensor> predicted "logits"/"scores" for all [allowed_predictions]'''

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # sample images
        x = self.generator.sample(size)

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        # set model back to its initial mode
        self.train(mode=mode)

        return (x, y, y_hat) if return_scores else (x, y)

    def generate_scores(self, x, allowed_predictions=None):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat

    def generate_scores_pen(self, x):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat

    def sample_stats(self, size, device):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # images per class
        random.shuffle(self.class_idx)
        num_per_class = math.ceil(size / self.num_k)
        start = 0
        end = num_per_class

        # sample targets
        logits_targets = {}
        for l in self.layer_idx:
            logits_targets[l] = []

            for k in self.class_idx:

                if k in list(self.stats[l]['mean'].keys()):

                    mean, stddev, cov = self.stats[l]['mean'][k], self.stats[l]['stdev'][k], self.stats[l]['cov'][k]
                    gauss = np.random.normal(size=(num_per_class, mean.shape[0]))
                    
                    # covariance or std
                    # pre_sftmx = np.multiply(gauss, stddev) + mean[None,:]
                    if l == -1:
                        pre_sftmx = np.matmul(gauss, cov) + mean[None,:]
                    else:
                        pre_sftmx = np.multiply(gauss, stddev) + mean[None,:]

                # # sample targets directly
                # sample_ind = np.random.choice(len(self.stats[l]['mem'][k]), num_per_class)
                # pre_sftmx = self.stats[l]['mem'][k][sample_ind]

                else:
                    lsize = self.stats[l]['mean'][list(self.stats[l]['mean'].keys())[0]].shape[0]
                    gauss = np.random.normal(size=(num_per_class, lsize))
                    one_hot = np.zeros((lsize,))
                    one_hot[k] = 1.0
                    pre_sftmx = gauss + one_hot


                logits_targets_k = torch.from_numpy(pre_sftmx).float().to(device)
                if l == -1:
                    logits_targets_k = F.relu(logits_targets_k / self.Dtemp)
                logits_targets_k.requires_grad_(True)

                # append
                logits_targets[l].append(logits_targets_k)


            # collect all targets
            logits_targets[l] = torch.cat(logits_targets[l])[:size]

        return logits_targets
