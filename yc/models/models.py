#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from select import select
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTaskModel(nn.Module):
    """ Single-task baseline model with encoder + decoder """

    def __init__(self, backbone: nn.Module, decoder: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.task = task

    def forward(self, x):

        out_size = x.size()[2:]
        out = self.decoder(self.backbone(x))
        return {self.task: F.interpolate(out, out_size, mode='bilinear')}


class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """

    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(MultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks

    def forward(self, x):
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        return {task: F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear') for task in self.tasks}


class IterativeMultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """

    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(IterativeMultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks

    def forward(self, x, selected_tasks=None):
        out_size = x.size()[2:]

        preds_tasks = selected_tasks if selected_tasks is not None else self.tasks
        shared_representation = {}
        for task in preds_tasks:
            shared_representation[task] = self.backbone(x, task)

        return {task: F.interpolate(self.decoders[task](shared_representation[task]), out_size, mode='bilinear') for task in preds_tasks}
