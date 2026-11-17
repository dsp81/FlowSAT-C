"""FlowSAT-C: metadata conditioning on FLUX.2-klein-base-4B.

Deliberately imports NOTHING at package level. `data.metadata` must stay usable
in dataset-build scripts without torch installed, and importing torch here would
pull it in through every submodule. Import what you need from the submodule:

    from flowsat_C.models.encoder import SatMetadataEncoder
    from flowsat_C.models.transformer import FlowSatC, build_flowsat_c
    from flowsat_C.data.dataset import FMoWCaptionDataset
"""
