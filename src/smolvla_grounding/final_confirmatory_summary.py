"""Preregistered final Task-0 statistics; no policy or environment imports."""

import csv
import json
from pathlib import Path
import random

from .protocol import atomic_json


MODELS = ("original_language", "augmented_language", "paired_bc")
PROMPTS = ("original", "paraphrase_final_01", "paraphrase_final_02")
STATES = tuple(range(33, 50))
METRICS = ("O", "P1", "P2", "P", "Gap", "H")
EFFECTS = (
    "delta_para_original", "delta_para_augmented",
    "delta_orig_original", "delta_orig_augmented",
    "delta_H_original", "delta_H_augmented",
)


def _percentile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _one_sample(outcomes, sampled_states):
    n = len(sampled_states)
    rates = {}
    for model in MODELS:
        original = sum(int(outcomes[state][model]["original"]) for state in sampled_states)
        final_01 = sum(int(outcomes[state][model]["paraphrase_final_01"]) for state in sampled_states)
        final_02 = sum(int(outcomes[state][model]["paraphrase_final_02"]) for state in sampled_states)
        o = original / n
        p1 = final_01 / n
        p2 = final_02 / n
        p = (final_01 + final_02) / (2 * n)
        rates[model] = {
            "O": o, "P1": p1, "P2": p2, "P": p,
            "Gap": p - o,
            "H": 2 * o * p / (o + p) if o + p else 0.0,
            "counts": {
                "original": original,
                "paraphrase_final_01": final_01,
                "paraphrase_final_02": final_02,
                "pooled_paraphrases": final_01 + final_02,
            },
        }
    bc = rates["paired_bc"]
    original = rates["original_language"]
    augmented = rates["augmented_language"]
    effects = {
        "delta_para_original": bc["P"] - original["P"],
        "delta_para_augmented": bc["P"] - augmented["P"],
        "delta_orig_original": bc["O"] - original["O"],
        "delta_orig_augmented": bc["O"] - augmented["O"],
        "delta_H_original": bc["H"] - original["H"],
        "delta_H_augmented": bc["H"] - augmented["H"],
    }
    return rates, effects


def compute_statistics(outcomes, states, replicates=10000, seed=42):
    """Paired cluster bootstrap: each resampled state retains all nine outcomes."""
    states = tuple(states)
    if len(states) != 17 or len(set(states)) != 17:
        raise ValueError("Exactly 17 unique state clusters are required")
    if replicates != 10000 or seed != 42:
        raise ValueError("Final bootstrap settings are frozen at B=10000, seed=42")
    if set(outcomes) != set(states):
        raise ValueError("State outcome matrix is incomplete or contains extra states")
    for state in states:
        if set(outcomes[state]) != set(MODELS):
            raise ValueError(f"Missing or extra model outcome for state {state}")
        for model in MODELS:
            if set(outcomes[state][model]) != set(PROMPTS):
                raise ValueError(f"Missing or extra prompt outcome for {model} state {state}")
            if any(type(outcomes[state][model][prompt]) is not bool for prompt in PROMPTS):
                raise ValueError("Every goal_ever outcome must be a boolean")

    point, effects = _one_sample(outcomes, states)
    draws = {f"{model}.{metric}": [] for model in MODELS for metric in METRICS}
    draws.update({effect: [] for effect in EFFECTS})
    rng = random.Random(seed)
    for _ in range(replicates):
        sample = [states[rng.randrange(17)] for _ in range(17)]
        sampled, sampled_effects = _one_sample(outcomes, sample)
        for model in MODELS:
            for metric in METRICS:
                draws[f"{model}.{metric}"].append(sampled[model][metric])
        for effect in EFFECTS:
            draws[effect].append(sampled_effects[effect])
    ci = {key: [_percentile(values, 0.025), _percentile(values, 0.975)]
          for key, values in draws.items()}
    return {
        "bootstrap": {
            "cluster_unit": "state_id", "replicates": replicates,
            "seed": seed, "confidence_level": 0.95,
            "interval": "percentile",
            "paired_resampling": "all three models and all three prompts retained per sampled state",
        },
        "models": {
            model: {
                "counts": point[model]["counts"],
                "rates": {metric: point[model][metric] for metric in METRICS},
                "ci95": {metric: ci[f"{model}.{metric}"] for metric in METRICS},
            }
            for model in MODELS
        },
        "paired_effects": {
            effect: {"estimate": effects[effect], "ci95": ci[effect]}
            for effect in EFFECTS
        },
    }


def summarize_final_run(run_dir):
    """Require every committed outcome before revealing any aggregate success rate."""
    run_dir = Path(run_dir)
    meta_file = run_dir / "experiment_meta_v4.json"
    if not meta_file.exists():
        raise RuntimeError("Missing v4 final-run metadata")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    matrix = meta["matrix_config"]
    if matrix["models"] != list(MODELS) or matrix["tasks"][0]["initial_states"] != list(STATES):
        raise RuntimeError("Final matrix differs from preregistered model/state set")
    outcomes = {}
    raw_rows = []
    for state in STATES:
        outcomes[state] = {}
        row = {"state_index": state}
        for model in MODELS:
            outcomes[state][model] = {}
            for prompt in PROMPTS:
                trial = run_dir / model / "task_00" / f"state_{state:04d}_seed_0000_{prompt}"
                commits = []
                if trial.exists():
                    commits = sorted(trial.glob("attempt_*/complete.json"))
                if len(commits) != 1:
                    raise RuntimeError(f"Expected one committed final rollout: {trial}")
                complete = json.loads(commits[0].read_text(encoding="utf-8"))
                episode_path = commits[0].parent / "episode.json"
                if not episode_path.exists():
                    raise RuntimeError(f"Missing episode record: {episode_path}")
                episode = json.loads(episode_path.read_text(encoding="utf-8"))
                expected = {
                    "status": "COMPLETED", "model": model, "task_id": 0,
                    "initial_state_index": state, "instruction_id": prompt,
                    "policy_steps": 280,
                }
                if any(complete.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(f"Invalid completion marker: {commits[0]}")
                for key in ("model", "task_id", "initial_state_index", "instruction_id",
                            "policy_steps", "goal_ever", "goal_final"):
                    if episode.get(key) != complete.get(key):
                        raise RuntimeError(f"Episode/commit disagreement: {episode_path}")
                if type(complete.get("goal_ever")) is not bool or type(complete.get("goal_final")) is not bool:
                    raise RuntimeError(f"Invalid success indicator: {commits[0]}")
                outcomes[state][model][prompt] = complete["goal_ever"]
                row[f"{model}_{prompt}_ever"] = int(complete["goal_ever"])
                row[f"{model}_{prompt}_final"] = int(complete["goal_final"])
        raw_rows.append(row)
    stats = compute_statistics(
        outcomes, STATES,
        matrix["statistical_rules"]["bootstrap_replicates"],
        matrix["statistical_rules"]["bootstrap_seed"],
    )
    report = {
        "run_name": matrix["run_name"],
        "status": "COMPLETED",
        "completed_rollouts": 153,
        "states": list(STATES),
        "models": list(MODELS),
        "prompts": list(PROMPTS),
        "primary_metric": "goal_ever",
        "secondary_metric": "goal_final",
        "matrix_sha256": meta["matrix_sha256"],
        "manifest_sha256": meta["manifest_sha256"],
        "catalog_sha256": meta["catalog_sha256"],
        **stats,
    }
    csv_path = run_dir / "per_state_raw_outcomes.csv"
    temp = csv_path.with_name(csv_path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    temp.replace(csv_path)
    atomic_json(run_dir / "summary.json", report)
    return report