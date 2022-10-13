"""Implements an Adapter, Low-rank adapters and Hyper-adapter Layers."""
import torch.nn as nn
from .adapter_utils import Activations
from models.hypercomplex.layers import PHMLinear
from .low_rank_layer import LowRankLinear

# config = {
#                 input_dim: self.e_pool_size,
#                 reduction_factor: 2,
#                 non_linearity: 'relu',
#                 phm_c_init: "normal",
#                 hypercomplex_division: 2,
#                 learn_phm: True,
#                 shared_phm_rule: True,
#                 factorized_phm: True,
#                 phm_rank: 1,
#                 phm_init_range: 0.01,
#                 kronecker_prod: True,
#             }

class LowRankAdapter(nn.Module):
    """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.down_sample_size = self.input_dim // config.reduction_factor

        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = LowRankLinear(self.input_dim, self.down_sample_size,
                                          w_init=config.low_rank_w_init,
                                          rank=config.low_rank_rank)
        self.up_sampler = LowRankLinear(self.down_sample_size, self.input_dim,
                                        w_init=config.low_rank_w_init,
                                        rank=config.low_rank_rank)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output


class HyperComplexAdapter(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.down_sample_size = self.input_dim // config.reduction_factor


        input_dim = self.input_dim

        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = PHMLinear(in_features=input_dim,
                                      out_features=self.down_sample_size,
                                      bias=True,
                                      c_init=config.phm_c_init,
                                      phm_dim=config.hypercomplex_division,
                                      learn_phm=config.learn_phm,
                                      w_init=config.hypercomplex_nonlinearity,
                                      shared_phm_rule=config.shared_phm_rule,
                                      factorized_phm=config.factorized_phm,
                                      shared_W_phm=config.shared_W_phm,
                                      factorized_phm_rule=config.factorized_phm_rule,
                                      phm_rank=config.phm_rank,
                                      phm_init_range=config.phm_init_range,
                                      kronecker_prod=config.kronecker_prod)
        self.up_sampler = PHMLinear(in_features=self.down_sample_size,
                                    out_features=self.output_dim,
                                    bias=True,
                                    c_init=config.phm_c_init,
                                    phm_dim=config.hypercomplex_division,
                                    learn_phm=config.learn_phm,
                                    w_init=config.hypercomplex_nonlinearity,
                                    shared_phm_rule=config.shared_phm_rule,
                                    factorized_phm=config.factorized_phm,
                                    shared_W_phm=config.shared_W_phm,
                                    factorized_phm_rule=config.factorized_phm_rule,
                                    phm_rank=config.phm_rank,
                                    phm_init_range=config.phm_init_range,
                                    kronecker_prod=config.kronecker_prod)

    def set_phm_rule(self, phm_rule=None, phm_rule_left=None, phm_rule_right=None):
        """If factorized_phm_rules is set, phm_rule is a tuple, showing the left and right
        phm rules, and if this is not set, this is showing  the phm_rule."""
        self.down_sampler.set_phm_rule(phm_rule, phm_rule_left, phm_rule_right)
        self.up_sampler.set_phm_rule(phm_rule, phm_rule_left, phm_rule_right)

    def forward(self, x, phm_rule=None):
        z = self.down_sampler(x, phm_rule)
        z = self.activation(z)
        return self.up_sampler(z, phm_rule)
