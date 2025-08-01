"""Cross‑modal fusion modules.

This subpackage provides classes for fusing pairs or triplets of
modalities.  Cross attention is implemented in `cross_attention`,
while the staged three‑modal fusion and classifier live in
`multi_modal_fusion`.
"""

from .cross_attention import CrossAttentionFusion
from .multi_modal_fusion import MultiModalFusion4, MultiModalClassifier4

__all__ = [
    "CrossAttentionFusion",
    "MultiModalFusion4",
    "MultiModalClassifier4",
]
