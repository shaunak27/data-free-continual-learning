from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import libmr
import numpy as np
from models.layers import CosineScaling
from utils.visualization import tsne_eval, confusion_matrix_vis, pca_eval
from torch.optim import Optimizer
import contextlib
import os
from .dgr_helper import Scholar
from .default import NormalNN, weight_reset, accumulate_acc, loss_fn_kd, Teacher
import copy
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.schedulers import CosineSchedule

KFDTTEMP = 8
POWER_ITERS = 10
NUMBER_STORED_EXAMPLES = 5000
STATS_ONLINE_DECAY = 0.99
class Generative_Replay(NormalNN):

    def __init__(self, learner_config):
        super(Generative_Replay, self).__init__(learner_config)
        self.generator = self.create_generator()
        self.generative_replay = False
        self.previous_scholar = None
        self.generator.recon_criterion = nn.BCELoss(reduction='none')
        self.dw = self.config['DW']
        self.debug_dir = None

        # generator optimizor
        self.generator.optimizer, self.generator_scheduler = self.new_optimizer(self.generator)

        # repeat call for generator network
        if self.gpu:
            self.cuda_gen()
        
    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()
                self.generator.optimizer, self.generator_scheduler = self.new_optimizer(self.generator)
            
            # data weighting
            self.data_weighting(train_dataset, num_seen=self.estimate_class_distribution(train_loader))

            # Evaluate the performance of current task
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = AverageMeter()
            acc = AverageMeter()
            gen_losses = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0:
                    self.scheduler.step()
                    self.generator_scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task)  in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()
                    self.generator.train()

                    # send data to gpu
                    if self.gpu:
                        x = [x[k].cuda() for k in range(len(x))]
                        y = y.cuda()

                    # data replay
                    if not self.generative_replay:
                        x_replay = None   #-> if no replay
                    else:
                        allowed_predictions = list(range(self.last_valid_out_dim))
                        x_replay, y_replay, y_replay_hat = self.previous_scholar.sample(len(x[0]), allowed_predictions=allowed_predictions,
                                                            return_scores=True)
                    
                    # if KD
                    if self.KD and self.generative_replay:
                        y_hat = self.previous_scholar.generate_scores(x[0], allowed_predictions=allowed_predictions)
                        _, y_hat_com = self.combine_data(((x[0], y_hat),(x_replay, y_replay_hat)))
                    else:
                        y_hat_com = None

                    # combine inputs and generated samples for classification
                    if self.generative_replay:
                        x_com, y_com = self.combine_data(((x[0], y),(x_replay, y_replay)))
                    else:
                        x_com, y_com = x[0], y

                    

                    # dgr data weighting (NOT online learning compatible)
                    if self.dw:
                        dw_cls = self.dw_k[y_com.long()]
                    else:
                        #dw_cls = self.dw_k[-1 * torch.ones(y_com.size()).long()]

                        mappings = torch.ones(y_com.size(), dtype=torch.float32)
                        if self.gpu:
                            mappings = mappings.cuda()
                        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
                        mappings[:self.last_valid_out_dim] = rnt
                        mappings[self.last_valid_out_dim:] = 1-rnt
                        dw_cls = mappings[y_com.long()]

                    # model update
                    loss, output= self.update_model(x_com, y_com, y_hat_com, dw_force = dw_cls, kd_index = np.arange(len(x[0]), len(x_com)))

                    # generator update
                    loss_gen = self.generator.train_batch(x_com, y_com, dw_cls, self.model, list(range(self.valid_out_dim)))

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y_com = y_com.detach()
                    accumulate_acc(output, y_com, task, acc, topk=(self.top_k,))
                    losses.update(loss,  y_com.size(0)) 
                    gen_losses.update(loss_gen, y_com.size(0))
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))
                self.log(' * Gen Loss {loss.avg:.3f}'.format(loss=gen_losses))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = AverageMeter()
                acc = AverageMeter()
                gen_losses = AverageMeter()

        self.model.eval()
        self.generator.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        # new scholar
        scholar = Scholar(generator=self.generator, solver=self.model)
        self.previous_scholar = copy.deepcopy(scholar)
        self.generative_replay = True

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)
        if len(self.config['gpuid']) > 1:
            self.previous_linear = copy.deepcopy(self.model.module.last)
        else:
            self.previous_linear = copy.deepcopy(self.model.last)

        try:
            return batch_time.avg
        except:
            return None

    ##########################################
    #             MODEL EVAL                 #
    ##########################################

    def visualization(self, dataloader, topdir, name, task, embedding):

        # save generated images
        self.debug_dir = topdir
        if self.generative_replay: self.visualize_gen(dataloader, topdir, name)

        return super(Generative_Replay, self).visualization(dataloader, topdir, name, task, embedding)

    def visualize_gen(self, dataloader, topdir, name, model=None, gen_text = "", gen_dim_cur = True):

        if model is None: model = self.previous_scholar

        if gen_dim_cur:
            gen_dim = self.valid_out_dim
        else:
            gen_dim = self.last_valid_out_dim

        num_show = 50
        cols = 10
        input_gen = []
        x_real = []
        y_real = []
        for i, (input, target, _) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            input_gen.append(input)
            x_real.extend(input.detach().cpu().numpy())
            y_real.extend(target.detach().cpu().numpy())
        sample_ind = np.random.choice(len(y_real), num_show+self.valid_out_dim)
        input_gen = torch.cat(input_gen)
        input_gen = input_gen[sample_ind]
        x_real, y_real = np.asarray(x_real), np.asarray(y_real)
        x_real, y_real = x_real[sample_ind], y_real[sample_ind]
        min_x, max_x = np.amin(x_real), np.amax(x_real)
        x_gen, y_gen = model.sample(num_show, allowed_predictions=np.arange(gen_dim),
                                                    return_scores=False)
        x_gen, y_gen = x_gen.detach().cpu().numpy(), y_gen.detach().cpu().numpy()                                       

        # print(min_x)
        # print(max_x)
        # print(np.amin(x_gen))
        # print(np.amax(x_gen))
        
        savedir = topdir + name + '/'
        if not os.path.exists(savedir): os.makedirs(savedir)  
        rows = math.ceil(num_show/cols)
        if len(gen_text) > 0:
            data_loop = [[(x_gen, y_gen), 'gen']]
        else:
            data_loop = [[(x_real, y_real), 'real'], [(x_gen, y_gen), 'gen']]
        for data in data_loop:
            x,y = data[0]
            for j in range(num_show):
                plt.subplot(rows, cols, j + 1)
                im_show = np.squeeze(x[j].transpose((1,2,0)))
                im_show = (im_show - min_x) / ((max_x - min_x))
                plt.imshow((im_show * 255).astype(np.uint8)); plt.axis('off')
                plt.title(str(y[j]))
            save_name = savedir + data[1] + '_images' + gen_text + '.png'
            plt.savefig(save_name) 
            plt.close()


    ##########################################
    #             MODEL UTILS                #
    ##########################################

    def save_model(self, filename):
        
        model_state = self.generator.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving generator model to:', filename)
        torch.save(model_state, filename + 'generator.pth')
        super(Generative_Replay, self).save_model(filename)

    def load_model(self, filename):
        
        self.generator.load_state_dict(torch.load(filename + 'generator.pth'))
        if self.gpu:
            self.generator = self.generator.cuda()
        self.generator.eval()
        super(Generative_Replay, self).load_model(filename)

    def create_generator(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        generator = models.__dict__[cfg['gen_model_type']].__dict__[cfg['gen_model_name']]()
        return generator

    def print_model(self):
        super(Generative_Replay, self).print_model()
        self.log(self.generator)
        self.log('#parameter of generator:', self.count_parameter_gen())
    
    def reset_model(self):
        super(Generative_Replay, self).reset_model()
        self.generator.apply(weight_reset)

    def count_parameter_gen(self):
        return sum(p.numel() for p in self.generator.parameters())

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.count_parameter_gen() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

    def cuda_gen(self):
        self.generator = self.generator.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.generator= torch.nn.DataParallel(self.generator, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def combine_data(self, data):
        x, y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(data[i][1])
        x, y = torch.cat(x), torch.cat(y)
        return x, y

    # estimate number of classes present for data weighting
    def estimate_class_distribution(self, train_loader):

        labels=[]
        for i, (x, y, task) in enumerate(train_loader):

            # send data to gpu
            if self.gpu:
                x = [x[k].cuda() for k in range(len(x))]
                y = y.cuda()

            # data replay
            if not self.generative_replay:
                x_replay = None   #-> if no replay
            else:
                allowed_predictions = list(range(self.last_valid_out_dim))
                x_replay, y_replay = self.previous_scholar.sample(len(x[0]), allowed_predictions=allowed_predictions,
                                                    return_scores=False)
            
            # combine inputs and generated samples for classification
            if self.generative_replay:
                x_com, y_com = self.combine_data(((x[0], y),(x_replay, y_replay)))
            else:
                x_com, y_com = x[0], y

            labels.extend(y_com.cpu().detach().numpy())
        
        labels = np.asarray(labels, dtype=np.int64)
        return np.asarray([len(labels[labels==k]) for k in range(self.valid_out_dim)], dtype=np.float32)