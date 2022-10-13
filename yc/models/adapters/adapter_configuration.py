# This file is mainly from rabeehk/hyperformer and rabeehk/compacter
# https://github.com/rabeehk/hyperformer
# https://github.com/rabeehk/compacter
"""Implements the adapters' configurations."""

from collections import OrderedDict

import torch.nn as nn
from dataclasses import dataclass


@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751.
    We additionally pass all the configuration of parameter-efficient finetuning
    methods with this config."""
    # standard adapters
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = True
    non_linearity: str = "swish"
    reduction_factor: int = 16
    weight_init_range = 1e-2
    conditional_layer_norm = False
    hidden_dim = 128
    train_adapters_blocks = True

    # add_adapter_in_feed_forward = True
    # add_adapter_in_self_attention = False
    # task_adapter_layers_encoder = None
    # task_adapter_layers_decoder = None
    intrinsic_dim = 100
    normalize_intrinsic_projections = False
    # This can be either random, or fastfood.
    intrinsic_projection = "random"

    # Hypercomplex adapters parameters
    hypercomplex_adapters = False
    hypercomplex_division = 8
    learn_phm = True
    hypercomplex_nonlinearity = "glorot-uniform"
    shared_phm_rule = False
    factorized_phm = False
    shared_W_phm = False
    factorized_phm_rule = False
    phm_c_init = "normal"
    phm_rank = 1
    phm_init_range = 0.01

    # BitFit configuration.
    bitfit = False

    # Low-rank adapters.
    low_rank_adapters = False
    low_rank_w_init = "glorot-uniform"
    low_rank_rank = 1

    kronecker_prod = False

# This is from hyperformer
# @dataclass
# class AdapterConfig(object):
#     """Implements the adapter configuration proposed by Houlsby et. al, 2019
#     in https://arxiv.org/abs/1902.00751."""
#     add_layer_norm_before_adapter: bool = False
#     add_layer_norm_after_adapter: bool = True
#     non_linearity: str = "swish"
#     reduction_factor: int = 16
#     weight_init_range = 1e-2
#     # Whether to use conditional layer norms for adapters.
#     conditional_layer_norm = False
#     hidden_dim = 128
#     # Whether to add adapter blocks, this is used in case we need
#     # to tune only layer norms.
#     train_adapters_blocks = True


class MetaAdapterConfig(AdapterConfig):
    """Implements Meta adapter in which a hyper-network generates the parameters of
     adapter layers. In this case we have a task embeddings which is feed to the
     hyper-network to allow it generate the weights for the adapter layers."""
    task_embedding_dim = 512
    task_embedding_dir = None
    hidden_dim = 128
    train_task_embeddings = False
    projected_task_embedding_dim = 64
    task_hidden_dim = 128
    parametric_task_embedding = False
    # If Specified, uses one hypernet to generates the adapters weights.
    unique_hyper_net = False
    unique_hyper_net_layer_norm = True
    # We consider only one hyper-net for all the blocks of transformer.
    efficient_unique_hyper_net = False


ADAPTER_CONFIG_MAPPING = OrderedDict(
    [("adapter", AdapterConfig),
     ("meta-adapter", MetaAdapterConfig)])


class AutoAdapterConfig(nn.Module):
    """Generic Adapter config class to instantiate different adapter configs."""

    @classmethod
    def get(cls, config_name: str):
        if config_name in ADAPTER_CONFIG_MAPPING:
            return ADAPTER_CONFIG_MAPPING[config_name]()
        raise ValueError(
            "Unrecognized adapter config type identifier: {}. Should contain one of {}"
            .format(config_name, ", ".join(ADAPTER_CONFIG_MAPPING.keys())))
