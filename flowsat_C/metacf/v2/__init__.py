"""
MetaCF v2 (final protocol). Code lives under metacf/v2/ and reuses v1 components by import:
records / manifests (metacf.data), generation requests / cache / planner (metacf.requests), statistics
(metacf.stats: bootstrap helpers, BH, circular stats, OLS), physical measures (metacf.physical),
CONSISTENT solar recomputation (metacf.solar), support map / derangement / seeding (metacf.fields),
reader heads (metacf.readers._ridge, _logreg, location_folds) and the FlowSAT-C adapter.

Layout
  core/       features (dino / clip / stats spaces + feature store), fields (regression transforms),
              pairs (same-site real pairs, VIF, joint groups), regression (pair OLS on per-site sufficient
              statistics, split-half reliability), select (feature-space selection), heads (readers),
              norm (normalisation helper, result rows)
  measures/   C0 coverage, C1 counterfactual fidelity, C2 synthetic-to-real transfer,
              C3 conditional information, C4 conditional distribution fidelity, C5 memorisation
  physics/    E1 resolution, E2 season, E3 sun, E4 year
  meta/       adapter wrappers + meta-evaluation harness
  run.py      the Run object (config, data, spaces, feature store, selection, image store)
  runner.py   build-pairs | select-spaces | fit-readers | plan | generate | score | meta | card
  summary.py  MetaCF summary, Benjamini-Hochberg, results.json, control_card.md / .html
  adapters.py DummyAdapterV2 (CPU tests) and the batched FlowSAT-C adapter
  cli.py      `python -m metacf.v2 <command> --config ...` (also the `metacf2` wrapper script)
"""
__version__ = "2.0.0"
