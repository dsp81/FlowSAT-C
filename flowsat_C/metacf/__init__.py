"""
MetaCF v1 -- a controllability evaluation protocol for conditional generators.

Import as `metacf` with this repo's root (flowsat_C/) on sys.path; the reused
FlowSAT-C code is imported as `flowsat_C.*`, so the directory above it (DJ_Sat/)
must be on sys.path too. `metacf.paths.ensure()` sets both.

Layout
  config.py      YAML config, canonical hash (P4)
  stats.py       bootstrap, Benjamini-Hochberg, Fisher-z, circular statistics, OLS (P6)
  mi.py          Ross (2014) k-NN mixed MI, Miller-Madow plug-in MI
  metrics.py     per-field-type metrics (Pearson, circular mean cos, balanced accuracy, geo km)
  physical.py    physical image measures (shadow direction, sharpness, scale, ExG, cloud mask)
  solar.py       CONSISTENT-mode recomputation of derived solar fields
  fields.py      field specs, value grids, support, anchors, derangements, neighbourhoods,
                 natural experiments
  data.py        dataset records and the fMoW manifest reader
  backbones.py   frozen feature backbones (CLIP-L, DINOv2 via eval_metrics; handcrafted for tests)
  readers.py     Reader A / Reader B heads, grouped-CV fitting, reader validation (P2, P3)
  requests.py    generation requests, content-addressed cache keys, image store, planner
  adapters/      ModelAdapter interface, FlowSatCAdapter, DummyAdapter, DiffusionSatAdapter stub
  context.py     shared state a run needs (records, readers, grids, anchors, sweeps)
  base.py        MetaCFTest base class and result rows (P1, P5)
  tier1/         T1-T10 universal tests
  tier2/         T11-T18 Earth-observation tests
  runner.py      plan / generate / run orchestration
  card.py        results.json -> control_card.md / .html
  cli.py         `python -m metacf <command> --config ...`
"""

__version__ = "1.0.0"
