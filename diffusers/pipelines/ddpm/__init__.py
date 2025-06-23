from typing import TYPE_CHECKING

from ...utils import (
    _LazyModule,
)


_import_structure = {"pipeline_ddpm": ["DDPMPipeline", "DDPMPipeline_ACT", "DDPMPipeline_PRUNE"]}

if TYPE_CHECKING:
    from .pipeline_ddpm import DDPMPipeline, DDPMPipeline_ACT, DDPMPipeline_PRUNE

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
