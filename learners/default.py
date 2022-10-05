from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from utils.visualization import tsne_eval, confusion_matrix_vis, pca_eval, calculate_cka
from torch.optim import Optimizer
import contextlib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
from utils.schedulers import CosineSchedule

class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, learner_config):

        super(NormalNN, self).__init__()
        self.log = print
        self.config = learner_config
        self.out_dim = learner_config['out_dim']
        self.model = self.create_model()
        self.reset_optimizer = True
        self.overwrite = learner_config['overwrite']
        self.batch_size = learner_config['batch_size']
        self.previous_teacher = None
        self.tasks = learner_config['tasks']
        self.top_k = learner_config['top_k']

        # replay memory parameters
        self.memory_size = self.config['memory']
        self.task_count = 0

        # class balancing
        self.dw = self.config['DW']
        if self.memory_size <= 0:
            self.dw = False

        # distillation
        self.DTemp = learner_config['temp']
        self.mu = learner_config['mu']
        self.beta = learner_config['beta']
        self.eps = learner_config['eps']

        # supervised criterion
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none')
        
        # cuda gpu
        if learner_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        
        # highest class index from past task
        self.last_valid_out_dim = 0 

        # highest class index from current task
        self.valid_out_dim = 0

        # set up schedules
        self.schedule_type = self.config['schedule_type']
        self.schedule = self.config['schedule']

        # initialize optimizer
        self.init_optimizer()

    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        # try to load model
        need_train = True
        # if not self.overwrite:
        if True:
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

            # data weighting
            self.data_weighting(train_dataset)
            
            # Evaluate the performance of current task
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task)  in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                    
                    # model update
                    loss, output= self.update_model(x, y)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = AverageMeter()
                acc = AverageMeter()
                
        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)

        try:
            return batch_time.avg
        except:
            return None

    def criterion(self, logits, targets, data_weights):
        """The loss criterion with any additional regularizations added
        The inputs and targets could come from single task 
        The network always makes the predictions with all its heads
        The criterion will match the head and task to calculate the loss.
        Parameters
        ----------
        logits : dict(torch.Tensor)
            Dictionary of predictions, e.g. outs from `forward`
        targets : torch.Tensor
            target labels
        Returns
        -------
        torch._Loss :
            the loss function with any modifications added
        """
        # print('*************')
        # print(data_weights)
        # print('*************')
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised 

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        if dw_force is not None:
            dw_cls = dw_force
        elif self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits

    ##########################################
    #             MODEL EVAL                 #
    ##########################################

    def visualization(self, dataloader, topdir, name, task, embedding):

        #
        # tsne
        #
        # setup files
        savedir = topdir + name + '/'
        if not os.path.exists(savedir): os.makedirs(savedir)
        savename = savedir 

        # gather data
        X = []
        y_pred = []
        y_true = []
        y_act = [[] for k in range(self.valid_out_dim)]
        logit_out = [[] for k in range(self.valid_out_dim)]
        orig_mode = self.training
        self.eval()
        for i, (input, target, _) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            output = self.predict(input)
            penultimate = self.model.forward(x=input, pen=True)
            X.extend(penultimate.cpu().detach().tolist())
            y_pred.extend(np.argmax(output.cpu().detach().numpy(), axis = 1).tolist())
            y_true.extend(target.cpu().detach())
            for k in range(self.valid_out_dim):
                y_act[k].extend(output[:,k].cpu().detach().numpy().tolist())
                y_ind = np.where(target.cpu().detach().numpy() == k)[0]
                logit_out[k].extend(output[y_ind,:].cpu().detach().numpy().tolist())
        self.train(orig_mode)

        # # debug - print activation stats
        # logit_out = [np.asarray(logit_out[k]) for k in range(self.valid_out_dim)]
        # print('acutal')
        # for k in range(self.valid_out_dim):
        #     print(np.mean(logit_out[k], axis=0))
        #     print(np.std(logit_out[k], axis=0))
        #     print(np.cov(logit_out[k].transpose()))

        # save activations
        y_act = np.asarray(y_act)
        cm_array = np.zeros((self.valid_out_dim,2))
        cm_array[:,0] = np.mean(y_act, axis=1)
        cm_array[:,1] = np.std(y_act, axis=1)
        np.savetxt(savename+'mean_std_act_per_task.csv', cm_array, delimiter=",", fmt='%.0f')

        # convert to arrays
        X = np.asarray(X)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # confusion matrix
        title = 'CM - Task ' + str(task+1)
        ood_cuttoff = self.valid_out_dim
        confusion_matrix_vis(y_pred[y_true < ood_cuttoff], y_true[y_true < ood_cuttoff], savename, title)

        # tsne for in and out of distribution data
        title = 'TSNE - Task ' + str(task+1)
        tsne_eval(X[y_true < ood_cuttoff], y_true[y_true < ood_cuttoff], savename, title, self.out_dim)

        # pca for in and out of distribution data
        title = 'PCA - Task ' + str(task+1)
        embedding = pca_eval(X[y_true < ood_cuttoff], y_true[y_true < ood_cuttoff], savename, title, self.out_dim, embedding)
        
        return embedding

    def data_visualization(self, dataloader, topdir, name, task):

        #
        # dataset vis
        #
        # setup files
        savedir = topdir + name + '/'
        if not os.path.exists(savedir): os.makedirs(savedir)
        num_show = 50
        cols = 10

        dataloader_list = [dataloader]
        savename_list = ['training', 'external']
        for dl in range(len(dataloader_list)):
            datal = dataloader_list[dl]
            savel = savename_list[dl]

            # gather data
            X = []
            for i, (input, target, _) in enumerate(datal):
                if self.gpu:
                    with torch.no_grad():
                        input = input.cuda()
                X.extend(input.cpu().detach().tolist())

            x = np.asarray(X) 
            np.random.shuffle(x)                               
            if dl == 0: min_x, max_x = np.amin(x), np.amax(x) 
            rows = math.ceil(num_show/cols)
            for j in range(num_show):
                plt.subplot(rows, cols, j + 1)
                im_show = np.squeeze(x[j].transpose((1,2,0)))
                im_show = (im_show - min_x) / ((max_x - min_x))
                plt.imshow((im_show * 255).astype(np.uint8)); plt.axis('off')
            save_name = savedir + savel + '_images.png'
            plt.savefig(save_name) 
            plt.close()

    def cka_eval(self, dataloader):

        try:
            # deep CKA analysis
            if True and self.task_count > 1:
                vis_dir = self.debug_dir + 'cka_plots' + '/'
                if not os.path.exists(vis_dir): os.makedirs(vis_dir)
                l_values = [0,1,2,3,4,5]

                for t_anchor in range(self.task_count):

                    # get file name
                    filename = vis_dir + str(t_anchor+1) + '.png'
                    cka_array = []
                    acc_array = []

                    # load model anchor
                    m_a = self.create_model()
                    m_a = self.load_model_other(self.debug_model_dir+str(t_anchor+1)+'/',m_a)
                    teacher_a = Teacher(m_a)

                    # get task
                    task_in = self.tasks[t_anchor]
                    cka_array.append([1.0 for l in range(len(l_values))])
                    acc_array.append(self.validation(dataloader, model=m_a, task_in = task_in, cka_flag=task_in[-1]))

                    for t_drift in range(t_anchor+1,self.task_count):
                        # load model drift
                        m_b = self.create_model()
                        m_b = self.load_model_other(self.debug_model_dir+str(t_drift+1)+'/',m_b)
                        teacher_b = Teacher(m_b)

                        # get cka similarity between two models
                        layer_array = []
                        for l in l_values:
                            X_anchor = []
                            X_drift = []
                            for i, (input, target, _) in enumerate(dataloader):
                                if self.gpu:
                                    with torch.no_grad():
                                        input = input.cuda()
                                        target = target.cuda()

                                mask = target >= task_in[0]
                                mask_ind = mask.nonzero().view(-1) 
                                input, target = input[mask_ind], target[mask_ind]

                                mask = target < task_in[-1]
                                mask_ind = mask.nonzero().view(-1) 
                                input, target = input[mask_ind], target[mask_ind]

                                if len(target) > 0:

                                    # current
                                    penultimate_anchor = teacher_a.generate_scores_layer(input,l)
                                    X_anchor.extend(penultimate_anchor.cpu().detach().tolist())

                                    # past
                                    penultimate_drift = teacher_b.generate_scores_layer(input,l)
                                    X_drift.extend(penultimate_drift.cpu().detach().tolist())

                            # convert to arrays
                            X_anchor = np.asarray(X_anchor)
                            X_drift = np.asarray(X_drift)

                            # return cka score
                            cka = calculate_cka(X_anchor, X_drift)
                            layer_array.append(cka)
                        cka_array.append(layer_array)
                        acc_array.append(self.validation(dataloader, model=m_b, task_in = task_in, cka_flag=self.tasks[t_drift][-1]))

                    # save plot
                    if len(cka_array) > 0:
                        cmap = plt.get_cmap('jet')
                        colors = cmap(np.linspace(0, 1.0, len(l_values)+1))
                        plt.figure(figsize=(8,4))
                        l_legend = ['Linear', 'Pen','L-2','L-3','L-4','L-5']
                        x = np.arange(len(cka_array)) + 1 + t_anchor
                        final_acc = np.asarray([cka_array[-1][l] for l in range(len(l_values))])
                        for s in range(len(l_values)):
                            i = np.argsort(final_acc)[-s-1]
                            y = np.asarray([cka_array[j][i] for j in range(len(x))])
                            plt.plot(x,y,lw=2, color = colors[i], linestyle = 'solid', label = l_legend[i])
                            plt.scatter(x,y,s=50, color = colors[i])
                        y = np.asarray(acc_array) / 100.0
                        plt.plot(x,y,lw=2, color = colors[-1], linestyle = 'dashed', label = 'acc')
                        plt.scatter(x,y,s=50, color = colors[-1])
                        tick_x = np.arange(self.task_count) + 1
                        tick_x_s = []
                        for tick in tick_x:
                            tick_x_s.append(str(int(tick)))
                        plt.xticks(tick_x, tick_x_s,fontsize=14)
                        plt.ylim(0,1.1)
                        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],fontsize=14)
                        plt.ylabel('CKA Score', fontweight='bold', fontsize=18)
                        plt.xlabel('Tasks', fontweight='bold', fontsize=18)
                        plt.title('CKA Forgetting Analysis for Task = '+str(t_anchor+1), fontsize=18)
                        plt.legend(loc='lower left', prop={'weight': 'bold', 'size': 10})
                        plt.grid()
                        plt.tight_layout()
                        plt.grid()
                        plt.savefig(filename,format='png')  
                        plt.close()

            # gather data
            X_cur = []
            X_past = []
            for i, (input, target, _) in enumerate(dataloader):
                if self.gpu:
                    with torch.no_grad():
                        input = input.cuda()
                        target = target.cuda()

                # current
                penultimate_cur = self.previous_teacher.generate_scores_pen(input)
                X_cur.extend(penultimate_cur.cpu().detach().tolist())

                # past
                penultimate_past = self.previous_previous_teacher.generate_scores_pen(input)
                X_past.extend(penultimate_past.cpu().detach().tolist())

            # convert to arrays
            X_cur = np.asarray(X_cur)
            X_past = np.asarray(X_past)

            # return cka score
            return calculate_cka(X_cur, X_past)
        except:
            return -1


    def validation(self, dataloader, model=None, task_in = None, task_metric='acc', relabel_clusters = True, verbal = True, cka_flag = -1, task_global=False):

        if model is None:
            if task_metric == 'acc':
                model = self.model
            else:
                return -1

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                output = model.forward(input)[:, :self.valid_out_dim]
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]

                mask = target < task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    if cka_flag > -1:
                        output = model.forward(input)[:, :cka_flag]
                        acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                    else:
                        if task_global:
                            output = model.forward(input)[:, :self.valid_out_dim]
                            acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                        else:
                            output = model.forward(input)[:, task_in]
                            acc = accumulate_acc(output, target-task_in[0], task, acc, topk=(self.top_k,))
            
        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    def label_clusters(self, dataloader):
        pass

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    # data weighting
    def data_weighting(self, dataset, num_seen=None):

        self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    def save_model(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        self.log('=> Save Done')

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename + 'class.pth'))
        self.log('=> Load Done')
        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

    def load_model_other(self, filename, model):
        model.load_state_dict(torch.load(filename + 'class.pth'))
        if self.gpu:
            model = model.cuda()
        return model.eval()


    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        print('*****************************************')
        print('*****************************************')
        print('*****************************************')
        print('Num param opt: ' + str(count_parameters(self.model.parameters())))
        print('*****************************************')
        print('*****************************************')
        print('*****************************************')
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim)

        return model

    def print_model(self):
        self.log(self.model)
        self.log('#parameter of model:', self.count_parameter())
    
    def reset_model(self):
        self.model.apply(weight_reset)

    def forward(self, x):
        return self.model.forward(x)[:, :self.valid_out_dim]

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        return out
    
    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())   

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log("Running on:", device)
        return device

    def pre_steps(self):
        pass

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def accumulate_acc(output, target, task, meter, topk):
    meter.update(accuracy(output, target, topk), len(target))
    return meter

def loss_fn_kd(scores, target_scores, data_weights, allowed_predictions, T=2., soft_t = False):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].
    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""


    log_scores_norm = F.log_softmax(scores[:, allowed_predictions] / T, dim=1)
    if soft_t:
        targets_norm = target_scores
    else:
        targets_norm = F.softmax(target_scores[:, allowed_predictions] / T, dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    # KD_loss_unnorm = KD_loss_unnorm * data_weights                  # data weighting
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm # * T**2

    return KD_loss

##########################################
#            TEACHER CLASS               #
##########################################

class Teacher(nn.Module):

    def __init__(self, solver):

        super().__init__()
        self.solver = solver

    def generate_scores(self, x, allowed_predictions=None, threshold=None):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # set model back to its initial mode
        self.train(mode=mode)

        # threshold if desired
        if threshold is not None:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            y_hat = F.softmax(y_hat, dim=1)
            ymax, y = torch.max(y_hat, dim=1)
            thresh_mask = ymax > (threshold)
            thresh_idx = thresh_mask.nonzero().view(-1)
            y_hat = y_hat[thresh_idx]
            y = y[thresh_idx]
            return y_hat, y, x[thresh_idx]

        else:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            ymax, y = torch.max(y_hat, dim=1)

            return y_hat, y

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

    def generate_scores_layer(self, x, layer):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True, l = layer)

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat

def count_parameters(params):
    return sum(p.numel() for p in params if p.requires_grad)