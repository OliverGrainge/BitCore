"""
bitcore package entry point.
"""
try: 
    import bitops
    HAS_BITOPS = True
except ImportError:
    HAS_BITOPS = False

from .bitlinear import BitLinear
from .bitlinear import get_quantizers, QUANTIZERS

__all__ = [
    "BitLinear", 
    "get_quantizers", 
    "QUANTIZERS",
]

