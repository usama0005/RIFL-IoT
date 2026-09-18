"""Prompts shared by all LLM-based methods. Site costs/breakdown rates are NOT disclosed."""
from rifl_iot.environment.dataset import SENSORS, STATS

SYSTEM_PROMPT_TEMPLATE = """You are a maintenance decision assistant for a fleet of hydraulic test rigs. Each decision concerns one unit, based on a 60-second sensor window (1 Hz) from its cooling circuit.

Sensors: temperature_C = mean of four circuit temperature sensors; cooling_efficiency_pct = cooler efficiency; cooling_power_kW = cooler power. For each sensor you get mean, std, min, max and slope (units per second) over the window.

Reference ranges observed on this fleet (from historical calibration data, mean values per sensor):
{reference_ranges}

Task:
1. Assess the cooler condition as exactly one of: healthy, degraded, critical, using the reference ranges above as your primary evidence.
2. Rank ALL four actions from best to worst for this unit:
A0 CONTINUE: keep normal operation.
A1 DERATE: reduce load; lowers breakdown risk, loses some throughput.
A2 SCHEDULE_MAINTENANCE: service the cooler at the next window; moderate downtime.
A3 SHUTDOWN_REPAIR: stop immediately and repair; highest downtime, removes breakdown risk.
Site-specific costs and breakdown rates are not provided.

Respond with ONLY a JSON object:
{{"condition": "healthy|degraded|critical", "ranking": ["A?", "A?", "A?", "A?"], "rationale": "<at most 25 words>"}}"""


def build_system_prompt(calib_summary):
    from rifl_iot.environment.calibration import format_reference_ranges
    return SYSTEM_PROMPT_TEMPLATE.format(reference_ranges=format_reference_ranges(calib_summary))


def observation_block(features):
    lines = ["Unit sensor window:"]
    for s in SENSORS:
        lines.append(f"{s}: " + ", ".join(f"{st}={features[s][st]}" for st in STATS))
    return "\n".join(lines)


ACTION_NAMES = ["A0_CONTINUE", "A1_DERATE", "A2_SCHEDULE_MAINTENANCE", "A3_SHUTDOWN_REPAIR"]


def compact_obs(features):
    return " ".join(f"{s}={features[s]['mean']}" for s in SENSORS)


def format_history(records):
    if not records:
        return "No prior decisions yet."
    lines = []
    for r in records:
        fb = "; ".join(f"fb(y_A={f['y_A']:.2f},b={f['b']})" for f in r["feedback"]) or "no feedback received yet"
        lines.append(f"- unit(perceived={r['llm_condition']}, {r['obs_compact']}) "
                     f"-> chose {ACTION_NAMES[r['action']]}; {fb}")
    return "\n".join(lines)


def history_messages(features, system_prompt, history_records):
    hist = "Recent decisions and their feedback, most recent last (up to last 20):\n" + format_history(history_records)
    user = hist + "\n\n" + observation_block(features)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def vanilla_messages(features, system_prompt):
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": observation_block(features)}]
