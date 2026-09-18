import json
import os

import numpy as np


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dumps(obj):
    return json.dumps(obj, default=_default)


class JsonlWriter:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "w")

    def write(self, obj):
        self._f.write(dumps(obj) + "\n")

    def close(self):
        self._f.close()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(obj, default=_default, indent=2))
