"""YAML config loading with `base:` inheritance and dotted overrides."""
import copy
import hashlib
import json
import os

import yaml


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load_config(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


def set_dotted(cfg, key, value):
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def apply_overrides(cfg, overrides):
    out = copy.deepcopy(cfg)
    for k, v in (overrides or {}).items():
        set_dotted(out, k, v)
    return out


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
