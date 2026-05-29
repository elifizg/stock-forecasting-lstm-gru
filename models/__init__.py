# models/__init__.py
# Tüm modelleri tek yerden import etmeyi sağlar.
# Kullanım:
#   from models import StockLSTM, get_model

from .lstm       import StockLSTM
from .gru        import StockGRU
from .bidir_lstm import BidirSignalLSTM
from .bidir_gru  import BidirSignalGRU

import torch.nn as nn


def get_model(name: str) -> nn.Module:
    """
    Model factory.

    name seçenekleri:
        'lstm'       → StockLSTM       (Part b/c, regresyon)
        'gru'        → StockGRU        (Part b/c, regresyon)
        'bidir_lstm' → BidirSignalLSTM (Part d, sınıflandırma)
        'bidir_gru'  → BidirSignalGRU  (Part d, sınıflandırma)
    """
    registry = {
        "lstm"      : StockLSTM,
        "gru"       : StockGRU,
        "bidir_lstm": BidirSignalLSTM,
        "bidir_gru" : BidirSignalGRU,
    }
    assert name in registry, (
        f"Bilinmeyen model: '{name}'. Seçenekler: {list(registry)}"
    )
    return registry[name]()


__all__ = [
    "StockLSTM",
    "StockGRU",
    "BidirSignalLSTM",
    "BidirSignalGRU",
    "get_model",
]
