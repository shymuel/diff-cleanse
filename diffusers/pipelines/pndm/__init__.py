from typing import TYPE_CHECKING

from ...utils import _LazyModule


_import_structure = {"pipeline_pndm": ["PNDMPipeline"]}

if TYPE_CHECKING:
    from .pipeline_pndm import PNDMPipeline, PNDMPipeline_ACT, PNDMPipeline_PRUNE
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
