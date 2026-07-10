from __future__ import annotations
from .base import Carrier, CarrierError
from .lsb import LSB
from .lsb_matching import LSBMatching
from .dct import DCT
from .adaptive import AdaptiveLSBM
from .adaptive_matrix import MatrixAdaptive
from .robust_dct import RobustDCT
_REGISTRY = {'lsb': LSB, 'lsbm': LSBMatching, 'dct': DCT, 'adaptive': AdaptiveLSBM, 'amx': MatrixAdaptive, 'rdct': RobustDCT}

def list_methods() -> list[str]:
    return sorted(_REGISTRY)

def get_method(name: str, **kwargs) -> Carrier:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f'unknown method {name!r}; choose from {list_methods()}') from None
    return cls(**kwargs)
__all__ = ['Carrier', 'CarrierError', 'LSB', 'LSBMatching', 'DCT', 'AdaptiveLSBM', 'MatrixAdaptive', 'RobustDCT', 'get_method', 'list_methods']
