"""Reuse the accepted paired trainer with a single changed optimization setting."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.adaptation_eval_records import read_training
from smolvla_grounding.adaptation_records import compare_training_logs, read_jsonl, schedule_hash
from smolvla_grounding.benchmark_records import validate_run_name
from smolvla_grounding.protocol import atomic_json, file_sha256, run_lock


def require_reference_controls(previous, history):
    fields = ("step", "episode_index", "frame_index", "variant_index", "data_hash", "rng_before", "noise_hash")
    if len(previous) != len(history) or any(any(a[k] != b[k] for k in fields) for a, b in zip(previous, history)):
        raise RuntimeError("Frames, language or randomness differ from the reference optimization experiment")


def main():
    config_path = "configs/learning_rate_comparison.json"
    config = json.loads((ROOT / config_path).read_text())
    reference = ROOT / config["reference_run"]
    accepted = read_training(ROOT, {"training_run": config["reference_run"],
                                   "models": ["untuned_fp32", "original_language", "augmented_language"]})
    original = json.loads((reference / "experiment.json").read_text())
    schedule = json.loads((reference / "schedule.json").read_text())
    original_settings = original["provenance"]["settings"]
    original_profile = json.loads((ROOT / original_settings["profile_config"]).read_text())
    for name, expected in {**original["provenance"]["sources"], **original["provenance"]["upstream_sources"]}.items():
        if file_sha256(ROOT / name) != expected:
            raise RuntimeError("Reference implementation changed: " + name)
    print(f"Matched repeat of {original['updates']} updates per arm, seed {original['seed']}; only peak learning rate changes.", flush=True)
    print(f"Reference peak: {original_profile['learning_rate']}; recommended next peak: {config['default_learning_rate']}.", flush=True)
    while True:
        try:
            name = validate_run_name(input(f"Training run name [{config['default_run_name']}]: ").strip() or config["default_run_name"])
            break
        except ValueError as exc:
            print(str(exc), flush=True)
    directory = ROOT / original_settings["output_root"] / "runs" / name
    if directory == reference:
        raise ValueError("Preserve the reference training run; choose a different name")
    with run_lock(directory):
        experiment_path = directory / "experiment.json"
        registration_path = directory / "learning_rate_comparison.json"
        registration = {"reference_comparison_sha256": accepted["comparison_sha256"],
                        "reference_experiment_sha256": file_sha256(reference / "experiment.json"),
                        "config_sha256": file_sha256(ROOT / config_path), "driver_sha256": file_sha256(__file__)}
        if experiment_path.exists():
            if not registration_path.exists() or json.loads(registration_path.read_text()) != registration:
                raise RuntimeError("Existing run belongs to another experiment or its source changed")
            experiment = json.loads(experiment_path.read_text())
            if input("Continue both arms from their last verified checkpoints? [y/N]: ").strip().lower() != "y":
                return
        else:
            while True:
                try:
                    rate = float(input(f"Peak learning rate [{config['default_learning_rate']}]: ").strip() or config["default_learning_rate"])
                    # The official preset retains its 2.5e-6 terminal floor.
                    if not math.isfinite(rate) or not 2.5e-6 < rate < original_profile["learning_rate"]:
                        raise ValueError("Choose a peak above the unchanged 2.5e-6 floor and below the reference 1e-4 peak")
                    break
                except ValueError as exc:
                    print(str(exc), flush=True)
            profile = {**original_profile, "learning_rate": rate}
            profile_path = directory / "profile_config.json"
            atomic_json(profile_path, profile)
            experiment = deepcopy(original)
            experiment["provenance"]["settings"]["profile_config"] = profile_path.relative_to(ROOT).as_posix()
            for path in (profile_path, ROOT / config_path, Path(__file__), ROOT / "scripts/run_learning_rate_comparison.sh",
                         ROOT / "src/smolvla_grounding/adaptation_eval_records.py"):
                experiment["provenance"]["sources"][path.relative_to(ROOT).as_posix()] = file_sha256(path)
            atomic_json(registration_path, registration)
            atomic_json(experiment_path, experiment)
        saved_schedule = directory / "schedule.json"
        if saved_schedule.exists() and json.loads(saved_schedule.read_text()) != schedule:
            raise RuntimeError("Saved sampling schedule changed")
        atomic_json(saved_schedule, schedule)
        finals = {}
        for arm in original_settings["arms"]:
            print("Start " + arm + "; using the unchanged official-training worker", flush=True)
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/train_adaptation_arm.py")], cwd=ROOT,
                env=dict(os.environ, SMOLVLA_ADAPTATION_DIR=str(directory), SMOLVLA_ADAPTATION_ARM=arm), check=True)
            finals[arm] = json.loads((directory / arm / "finished.json").read_text())
        logs = [read_jsonl(directory / a / finals[a]["training_log"]) for a in original_settings["arms"]]
        controls = compare_training_logs(*logs, schedule)
        initial = [json.loads((directory / a / "initial_parameters.json").read_text()) for a in original_settings["arms"]]
        expected = json.loads((reference / "original_language/initial_parameters.json").read_text())
        if initial[0] != expected or initial[1] != expected:
            raise RuntimeError("New runs did not start from the same original weights and precision")
        controls["same_initial_parameter_bytes"] = True
        reference_report = json.loads((reference / "comparison.json").read_text())
        for arm, history in zip(original_settings["arms"], logs, strict=True):
            previous = read_jsonl(reference / arm / reference_report["arms"][arm]["training_log"])
            require_reference_controls(previous, history)
        atomic_json(directory / "comparison.json", {"passed": True, "updates_per_arm": experiment["updates"],
            "schedule_hash": schedule_hash(schedule), "controls": controls, "arms": finals,
            "reference_training_controls_equal": True, "evaluation_status": "Not run; no robustness improvements claimed"})
        print("PASS: lower-learning-rate arms completed; controls matched within this run and against the reference.", flush=True)
        print("Saved " + (directory / "comparison.json").relative_to(ROOT).as_posix(), flush=True)
        print("Simulator evaluation is still required. Training loss alone is not a success-rate result.", flush=True)


if __name__ == "__main__":
    main()
