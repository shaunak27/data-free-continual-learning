import torch
import torch.utils.data as data
from torch.utils.data import TensorDataset
import numpy as np

class ModuleHelper(object):

    def __init__(self, batch_shape, num_blocks):
        super(ModuleHelper, self).__init__()
        self.current_task = 0
        self.memory_size = 250
        self.task_memory = {}
        self.current_task_memory = []
        self.batch_shape = batch_shape
        self.num_blocks = num_blocks
        self.bs = batch_shape[0]

    def update_memory(self, x):

        # add data to current memory, if space is available
        if len(self.current_task_memory) < self.memory_size:
            self.current_task_memory.extend(x.detach().cpu().tolist())

    def update_loader(self, t):

        # update task memory
        task_count = t
        num_sample_per_task = self.memory_size // task_count
        for storage in self.task_memory.values():
            storage = storage[:num_sample_per_task]
        randind = np.random.permutation(len(self.current_task_memory))[:num_sample_per_task]
        
        self.task_memory[task_count] = [self.current_task_memory[i] for i in randind]
        self.current_task_memory = []

        # form new dataloader
        dataset_list = []
        for storage in self.task_memory.values():
            dataset_list.append(TensorDataset(torch.from_numpy(np.asarray(storage))))
        dataset = torch.utils.data.ConcatDataset(dataset_list)
        self.new_train_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=self.bs,
                                                    shuffle=True,
                                                    num_workers=1)
        self.module_iter = iter(self.new_train_loader)

        # update task id
        self.current_task = t

    def generate_samples(self):
        
        try:
            x = next(self.module_iter)
        except:
            self.module_iter = iter(self.new_train_loader)
            x = next(self.module_iter)
        return x[0].cuda().reshape(-1, int(self.batch_shape[1]/self.num_blocks), self.batch_shape[2], self.batch_shape[3]).float()

    def unflatten(self, x):
        return x.reshape(-1, int(self.batch_shape[1]/self.num_blocks), self.batch_shape[2], self.batch_shape[3])