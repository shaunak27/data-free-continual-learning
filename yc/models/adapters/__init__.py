# This file is mainly from rabeehk/hyperformer
# https://github.com/rabeehk/hyperformer


from .adapter_configuration import ADAPTER_CONFIG_MAPPING, AutoAdapterConfig
from .adapter_configuration import AdapterConfig, MetaAdapterConfig
from .adapter_controller import (AdapterController,
                                 AutoAdapterController, MetaLayersAdapterController)

# MetaAdapterController,
from .adapter_hypernetwork import Adapter, AdapterHyperNet, AdapterLayersHyperNetController, \
    AdapterLayersOneHyperNetController, AdapterLayersOneKroneckerHyperNetController
from .adapter_compacter import LowRankAdapter, HyperComplexAdapter
from .adapter_utils import TaskEmbeddingController, TaskPHMRULEController, LayerNormHyperNet
