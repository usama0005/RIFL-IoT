"""Reference ranges for the LLM prompt, computed from the calibration pool only.

Never touches the episode pool used during runs, so there is no leakage between
"what the LLM is told about typical values" and "what it is evaluated on".
"""
import numpy as np

from rifl_iot.environment.dataset import SENSORS


def compute_calibration_summary(data, calib_idx):
    ctx = data.context[calib_idx]
    summary = {}
    for c, name in enumerate(["healthy", "degraded", "critical"]):
        idx = calib_idx[ctx == c]
        summary[name] = {
            s: {"mean": round(float(data.feat[s][idx, 0].mean()), 2),
                "min": round(float(data.feat[s][idx, 0].min()), 2),
                "max": round(float(data.feat[s][idx, 0].max()), 2)}
            for s in SENSORS
        }
    return summary


def format_reference_ranges(summary):
    lines = []
    for name in ["healthy", "degraded", "critical"]:
        parts = [f"{s}≈{summary[name][s]['mean']} (range {summary[name][s]['min']}–{summary[name][s]['max']})"
                 for s in SENSORS]
        lines.append(f"- {name}: " + ", ".join(parts))
    return "\n".join(lines)
