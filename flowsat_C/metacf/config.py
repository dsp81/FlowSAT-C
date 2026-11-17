"""P4 frozen config: a run is one YAML file; its hash is sha256 of the canonical JSON."""
from __future__ import annotations

import copy
import hashlib
import json

import yaml


def load_config(src) -> dict:
    """Path to a YAML file, or a dict (deep-copied)."""
    if isinstance(src, dict):
        return copy.deepcopy(src)
    with open(src) as f:
        return yaml.safe_load(f)


def apply_model(cfg: dict, model_src) -> dict:
    """Swap in a model overlay (configs/metacf_models/<name>.yaml): its `model` section REPLACES
    cfg["model"]; its `plan` section is merged over cfg["plan"] (seconds_per_image is per model).
    The protocol (fields, tests, readers) is untouched. The hash is taken on the merged config, so
    the same protocol on two models lands in two result directories (P4 still holds: the merged
    config is written next to the results as config.json)."""
    over = load_config(model_src)
    out = copy.deepcopy(cfg)
    out["model"] = copy.deepcopy(over["model"])
    if over.get("plan"):
        out["plan"] = dict(out.get("plan", {}), **over["plan"])
    return out


def canonical_json(cfg: dict) -> str:
    """Sorted keys, no whitespace, so the same config always serialises identically."""
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: dict, n: int = 12) -> str:
    return hashlib.sha256(canonical_json(cfg).encode()).hexdigest()[:n]


def get(cfg: dict, path: str, default=None):
    """get(cfg, "tests.T1.n_records", 1000)"""
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def test_cfg(cfg: dict, test_id: str) -> dict:
    """Per-test section merged over `defaults`: tests.<id> wins."""
    out = dict(cfg.get("defaults", {}))
    out.update(cfg.get("tests", {}).get(test_id, {}) or {})
    return out
