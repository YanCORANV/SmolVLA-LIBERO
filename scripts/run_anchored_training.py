"""Train anchored_language arm matching task0_longer_training with P(original)=0.50 and P(paraphrases)=0.125."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.adaptation_records import (
    build_anchored_schedule, compare_anchored_training_logs, read_jsonl, schedule_hash,
)
from smolvla_grounding.benchmark_records import validate_run_name
from smolvla_grounding.protocol import atomic_json, file_sha256, run_lock


def ask_string(label, default):
    try:
        val = input(f"{label} [{default}]: ").strip()
        return val or default
    except EOFError:
        print(f"{label} [default {default}] (non-interactive)", flush=True)
        return default


def ask_integer(label, default, minimum):
    while True:
        try:
            val_str = input(f"{label} [{default}]: ").strip()
            value = int(val_str or str(default))
            if not minimum <= value < 2**32:
                raise ValueError(f"Enter an integer in [{minimum}, {2**32 - 1}]")
            return value
        except EOFError:
            print(f"{label} [default {default}] (non-interactive)", flush=True)
            return default
        except ValueError as exc:
            print(str(exc), flush=True)


def main():
    config_path = "configs/anchored_training.json"
    config = json.loads((ROOT / config_path).read_text(encoding="utf-8"))
    reference = ROOT / config["reference_run"]
    ref_experiment = json.loads((reference / "experiment.json").read_text(encoding="utf-8"))
    ref_schedule = json.loads((reference / "schedule.json").read_text(encoding="utf-8"))
    settings = deepcopy(ref_experiment["provenance"]["settings"])
    settings["arms"] = [config["arm"]]

    profile = json.loads((ROOT / settings["profile_config"]).read_text(encoding="utf-8"))
    ready = json.loads((ROOT / profile["data_ready"]).read_text(encoding="utf-8"))
    if file_sha256(ROOT / profile["data_ready"]) != ref_experiment["provenance"]["data_ready"]:
        raise RuntimeError("Demonstration split/data evidence changed relative to reference")

    training = [r for r in ready["episodes"] if r["episode_index"] in ready["selection"]["split"]["train"]]
    if len(training) != 36:
        raise RuntimeError(f"Expected 36 training demonstrations, got {len(training)}")

    print(f"Anchored language training arm at peak LR {profile['learning_rate']}; budget {config['default_updates']}.", flush=True)
    print("Hypothesis: Increasing canonical-language exposure (P=0.50) recovers original-instruction performance while retaining paraphrase robustness.", flush=True)

    while True:
        try:
            name = validate_run_name(ask_string("Training run name", config["default_run_name"]))
            break
        except ValueError as exc:
            print(str(exc), flush=True)

    directory = ROOT / settings["output_root"] / "runs" / name
    if directory == reference:
        raise ValueError("Choose a new run name")

    sources = {
        config_path: file_sha256(ROOT / config_path),
        "scripts/run_anchored_training.py": file_sha256(__file__),
        "scripts/train_adaptation_arm.py": file_sha256(ROOT / "scripts/train_adaptation_arm.py"),
        "src/smolvla_grounding/adaptation_records.py": file_sha256(ROOT / "src/smolvla_grounding/adaptation_records.py"),
        settings["profile_config"]: file_sha256(ROOT / settings["profile_config"]),
        settings["training_catalog"]: file_sha256(ROOT / settings["training_catalog"]),
        settings["development_catalog"]: file_sha256(ROOT / settings["development_catalog"]),
    }

    registration = {
        "reference_run": config["reference_run"],
        "reference_experiment_sha256": file_sha256(reference / "experiment.json"),
        "config_sha256": file_sha256(ROOT / config_path),
        "driver_sha256": file_sha256(__file__),
    }

    with run_lock(directory):
        path = directory / "experiment.json"
        if path.exists():
            experiment = json.loads(path.read_text(encoding="utf-8"))
            print(f"Existing run found: {experiment['updates']} updates; arm {settings['arms']}.", flush=True)
            if ask_string("Continue from verified checkpoint? [y/N]", "y").strip().lower() != "y":
                return
        else:
            experiment = {
                "provenance": {
                    "settings": settings,
                    "upstream_sources": ref_experiment["provenance"]["upstream_sources"],
                    "profile": ref_experiment["provenance"]["profile"],
                    "data_ready": ref_experiment["provenance"]["data_ready"],
                    "sources": {**ref_experiment["provenance"]["sources"], **sources},
                },
                "updates": ask_integer("Optimizer updates", config["default_updates"], 2),
                "seed": ask_integer("Training seed", config["default_seed"], 0),
            }
            interval = ask_integer("Checkpoint interval in updates", config["default_checkpoint_every"], 1)
            experiment["provenance"]["settings"]["checkpoint_every"] = interval
            atomic_json(directory / "anchored_reference.json", registration)
            atomic_json(path, experiment)

        # Build anchored schedule: P(0)=0.50, P(1..4)=0.125
        schedule = build_anchored_schedule(training, experiment["updates"], experiment["seed"], settings["language_seed"], 5)
        # Verify that frame sampling is 100% identical to reference
        ref_frames = [(r["step"], r["episode_index"], r["frame_index"]) for r in ref_schedule[:experiment["updates"]]]
        anc_frames = [(r["step"], r["episode_index"], r["frame_index"]) for r in schedule]
        if ref_frames != anc_frames:
            raise RuntimeError("Frame sampling sequence diverged from reference task0_longer_training schedule!")

        schedule_path = directory / "schedule.json"
        if schedule_path.exists() and json.loads(schedule_path.read_text(encoding="utf-8")) != schedule:
            raise RuntimeError("Saved frame/language schedule changed")
        atomic_json(schedule_path, schedule)

        # Verify sampling probabilities
        counts = {i: sum(r["anchored_variant"] == i for r in schedule) for i in range(5)}
        print(f"Schedule language counts (updates={experiment['updates']}): {counts}", flush=True)
        print(f"  P(original instruction) = {counts[0] / experiment['updates']:.4f} (target 0.5000)", flush=True)
        for i in range(1, 5):
            print(f"  P(paraphrase {i:02d}) = {counts[i] / experiment['updates']:.4f} (target 0.1250)", flush=True)

        arm = "anchored_language"
        print(f"\nStart {arm} training ({experiment['updates']} updates)...", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts/train_adaptation_arm.py")], cwd=ROOT,
                       env=dict(os.environ, SMOLVLA_ADAPTATION_DIR=str(directory), SMOLVLA_ADAPTATION_ARM=arm), check=True)

        finished_path = directory / arm / "finished.json"
        if not finished_path.exists():
            raise RuntimeError(f"Training did not produce {finished_path}")
        finished = json.loads(finished_path.read_text(encoding="utf-8"))

        # Verify initial parameters match reference
        expected_init = json.loads((reference / "original_language/initial_parameters.json").read_text(encoding="utf-8"))
        actual_init = json.loads((directory / arm / "initial_parameters.json").read_text(encoding="utf-8"))
        if actual_init != expected_init:
            raise RuntimeError("Starting parameters differ from reference task0_longer_training!")

        # Compare training log with original_language log
        ref_orig_log = read_jsonl(reference / "original_language/step_001000/attempt_001/training_log.jsonl")
        anc_log = read_jsonl(directory / arm / finished["training_log"])
        controls = compare_anchored_training_logs(ref_orig_log, anc_log, schedule)
        controls["same_initial_parameter_bytes"] = True

        # Check empirical variant counts in committed training log
        log_counts = {i: sum(r["variant_index"] == i for r in anc_log) for i in range(5)}
        if log_counts != counts:
            raise RuntimeError(f"Training log variant counts {log_counts} differ from schedule {counts}!")

        comparison_report = {
            "passed": True,
            "arm": arm,
            "updates": experiment["updates"],
            "schedule_hash": schedule_hash(schedule),
            "language_counts": log_counts,
            "language_probabilities": {i: log_counts[i] / experiment["updates"] for i in range(5)},
            "controls": controls,
            "checkpoint": finished["checkpoint"],
            "parameter_update": finished["parameter_update"],
            "peak_reserved_bytes": finished["peak_reserved_bytes"],
        }
        atomic_json(directory / "comparison.json", comparison_report)
        print("\nPASS: anchored_language training completed successfully!", flush=True)
        print(f"Saved: {directory / 'comparison.json'}", flush=True)
        print(f"Checkpoint: {directory / arm / finished['checkpoint'] / 'pretrained_model'}", flush=True)


if __name__ == "__main__":
    main()
