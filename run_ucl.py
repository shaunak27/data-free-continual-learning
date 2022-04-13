from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import argparse
import torch
import numpy as np
import yaml
import json
import random
from trainer import Trainer

def create_args():
    
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                         help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="outputs/out",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='MLP', help="The name of actual model for the backbone")
    parser.add_argument('--gen_model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of generator network")
    parser.add_argument('--gen_model_name', type=str, default='MLP', help="The name of actual model for the generator")
    parser.add_argument('--learner_type', type=str, default='default', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='NormalNN', help="The class name of learner")
    parser.add_argument('--optimizer', type=str, default='SGD', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="CIFAR10|MNIST")
    parser.add_argument('--dataset_ext', type=str, default='SAME', help="SAME|CIFAR10|MNIST|KMNIST|FashionMNIST|SVHN|FakeData")
    parser.add_argument('--dataset_swap', default=False, action='store_true', help='swap half of tasks in two datasets')
    parser.add_argument('--load_model_dir', type=str, default=None, help="try loading from external model directory")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    parser.add_argument('--KD', default=False, action='store_true', help='knowledge distillation')
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--schedule_type', type=str, default='decay',
                        help="decay, cosine")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--debug_mode', type=int, default=0, metavar='N',
                    help="activate learner specific settings for debug_mode")
    parser.add_argument('--workers', type=int, default=8, help="#Thread for dataloader")
    parser.add_argument('--vis_flag', type=int, default=0, metavar='N',
                    help="activate visualizations")
    parser.add_argument('--validation', default=False, action='store_true', help='Evaluate on fold of training dataset rather than testing data')
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--overwrite', type=int, default=0, metavar='N', help='Train regardless of whether saved model exists')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='')
    parser.add_argument('--playground_flag', default=False, action='store_true', help='')
    parser.add_argument('--balanced_bce', default=False, action='store_true', help='')
    
    

    # layer statistic matching
    parser.add_argument('--stat_layers', nargs="+", type=int, default=[-1],
                         help="The list of layers to match statistics")
    
    # cluster parameters
    parser.add_argument('--online_cluster', default=False, action='store_true', help='clustering takes place with feature learning')

    # data free knoweldge distilation
    parser.add_argument('--power_iters', type=int, default=10, help="backprop power iterations for producing images")
    parser.add_argument('--refresh_iters', type=int, default=10, help="update gen images")
    parser.add_argument('--deep_inv_params', nargs="+", type=float, default=[-1],
                         help="")

    # CL Args
    parser.add_argument('--first_split_size', type=int, default=2)
    parser.add_argument('--other_split_size', type=int, default=2)              
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--max_task', type=int, default=-1, help="number tasks to perform; if -1, then all tasks; for debug")
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")

    # Arch params
    parser.add_argument('--block_size', type=int, default=1)
    parser.add_argument('--layer_freeze', type=int, default=1)
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--eps', type=float, default=1.0)


    return parser

def get_args(argv):
    parser=create_args()
    args = parser.parse_args(argv)
    return args

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    # determinstic backend
    torch.backends.cudnn.deterministic=True

    # duplicate output stream to output file
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_out = args.log_dir + '/output.log'
    sys.stdout = Logger(log_out)

    # save args
    with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)
    
    metric_keys = ['acc','aux_task','mem','time','plastic','til','cka']
    save_keys = ['global', 'pt', 'pt-local']
    global_only = ['mem','time','plastic','til','cka']
    avg_metrics = {}
    for mkey in metric_keys: 
        avg_metrics[mkey] = {}
        for skey in save_keys: avg_metrics[mkey][skey] = []

    # load results
    if args.overwrite:
        start_r = 0
    else:
        try:
            for mkey in metric_keys: 
                for skey in save_keys:
                    if (not (mkey in global_only)) or (skey == 'global'):
                        save_file = args.log_dir+'/results-'+mkey+'/'+skey+'.yaml'
                        if os.path.exists(save_file):
                            with open(save_file, 'r') as yaml_file:
                                yaml_result = yaml.safe_load(yaml_file)
                                avg_metrics[mkey][skey] = np.asarray(yaml_result['history'])

            # next repeat needed
            start_r = avg_metrics[metric_keys[0]][save_keys[0]].shape[-1]

            # extend if more repeats left
            if start_r < args.repeat:
                max_task = avg_metrics['acc']['global'].shape[0]
                for mkey in metric_keys: 
                    avg_metrics[mkey]['global'] = np.append(avg_metrics[mkey]['global'], np.zeros((max_task,args.repeat-start_r)), axis=-1)
                    if (not (mkey in global_only)):
                        avg_metrics[mkey]['pt'] = np.append(avg_metrics[mkey]['pt'], np.zeros((max_task,max_task,args.repeat-start_r)), axis=-1)
                        avg_metrics[mkey]['pt-local'] = np.append(avg_metrics[mkey]['pt-local'], np.zeros((max_task,max_task,args.repeat-start_r)), axis=-1)

        except:
            start_r = 0
    # start_r = 0
    for r in range(start_r, args.repeat):

        print('************************************')
        print('* STARTING TRIAL ' + str(r+1))
        print('************************************')

        # set random seeds
        seed = r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # set up a trainer
        trainer = Trainer(args, seed, metric_keys, save_keys)

        # init total run metrics storage
        max_task = trainer.max_task
        if r == 0: 
            for mkey in metric_keys: 
                avg_metrics[mkey]['global'] = np.zeros((max_task,args.repeat))
                if (not (mkey in global_only)):
                    avg_metrics[mkey]['pt'] = np.zeros((max_task,max_task,args.repeat))
                    avg_metrics[mkey]['pt-local'] = np.zeros((max_task,max_task,args.repeat))

        # train model
        avg_metrics = trainer.train(avg_metrics)  

        # evaluate model
        avg_metrics = trainer.evaluate(avg_metrics)    

        # save results
        for mkey in metric_keys: 
            m_dir = args.log_dir+'/results-'+mkey+'/'
            if not os.path.exists(m_dir): os.makedirs(m_dir)
            for skey in save_keys:
                if (not (mkey in global_only)) or (skey == 'global'):
                    save_file = m_dir+skey+'.yaml'
                    result=avg_metrics[mkey][skey]
                    yaml_results = {}
                    if len(result.shape) > 2:
                        yaml_results['mean'] = result[:,:,:r+1].mean(axis=2).tolist()
                        if r>1: yaml_results['std'] = result[:,:,:r+1].std(axis=2).tolist()
                        yaml_results['history'] = result[:,:,:r+1].tolist()
                    else:
                        yaml_results['mean'] = result[:,:r+1].mean(axis=1).tolist()
                        if r>1: yaml_results['std'] = result[:,:r+1].std(axis=1).tolist()
                        yaml_results['history'] = result[:,:r+1].tolist()
                    with open(save_file, 'w') as yaml_file:
                        yaml.dump(yaml_results, yaml_file, default_flow_style=False)

                    if mkey == 'acc' and skey == 'global':
                        with open(args.log_dir + '/_global-acc', 'w') as yaml_file:
                            yaml.dump(yaml_results, yaml_file, default_flow_style=False)

        # Print the summary so far
        print('===Summary of experiment repeats:',r+1,'/',args.repeat,'===')
        for mkey in metric_keys: 
            print(mkey, ' | mean:', avg_metrics[mkey]['global'][-1,:r+1].mean(), 'std:', avg_metrics[mkey]['global'][-1,:r+1].std())
    
    


