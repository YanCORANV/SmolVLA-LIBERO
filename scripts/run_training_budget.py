"""Longer paired training using the unchanged accepted worker and frozen data."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_adaptation import ask_integer
from run_learning_rate_comparison import require_reference_controls
from smolvla_grounding.adaptation_eval_records import read_training
from smolvla_grounding.adaptation_records import build_schedule, compare_training_logs, read_jsonl, schedule_hash
from smolvla_grounding.benchmark_records import validate_run_name
from smolvla_grounding.protocol import atomic_json, file_sha256, run_lock


def longer_schedule(episodes, updates, seed, language_seed, variants, previous):
    if updates <= len(previous):
        raise ValueError("Choose more updates than the reference run")
    schedule = build_schedule(episodes, updates, seed, language_seed, variants)
    if schedule[:len(previous)] != previous:
        raise RuntimeError("The reference sampling prefix changed")
    return schedule


def main():
    config_path = "configs/training_budget.json"
    config = json.loads((ROOT / config_path).read_text())
    reference = ROOT / config["reference_run"]
    accepted = read_training(ROOT, {"training_run": config["reference_run"],
        "models": ["untuned_fp32", "original_language", "augmented_language"]})
    original = json.loads((reference / "experiment.json").read_text())
    old_schedule = json.loads((reference / "schedule.json").read_text())
    settings = original["provenance"]["settings"]
    for name, expected in {**original["provenance"]["sources"], **original["provenance"]["upstream_sources"]}.items():
        if file_sha256(ROOT / name) != expected:
            raise RuntimeError("Accepted trainer source/config changed: " + name)
    profile = json.loads((ROOT / settings["profile_config"]).read_text())
    ready = json.loads((ROOT / profile["data_ready"]).read_text())
    if file_sha256(ROOT / profile["data_ready"]) != original["provenance"]["data_ready"]:
        raise RuntimeError("Accepted demonstration split/data evidence changed")
    print(f"Two fresh training arms at peak LR {profile['learning_rate']}; reference budget {original['updates']}.", flush=True)
    print("The cosine decay horizon grows with the budget. Earlier checkpoints/results remain unchanged.", flush=True)
    while True:
        try:
            name = validate_run_name(input(f"Training run name [{config['default_run_name']}]: ").strip() or config["default_run_name"])
            break
        except ValueError as exc:
            print(str(exc), flush=True)
    directory = ROOT / settings["output_root"] / "runs" / name
    if directory == reference:
        raise ValueError("Choose a new run name")
    registration = {"reference_comparison_sha256": accepted["comparison_sha256"],
                    "reference_experiment_sha256": file_sha256(reference / "experiment.json"),
                    "config_sha256": file_sha256(ROOT / config_path), "driver_sha256": file_sha256(__file__)}
    with run_lock(directory):
        path = directory / "experiment.json"
        if path.exists():
            saved_registration = directory / "budget_reference.json"
            if not saved_registration.exists() or json.loads(saved_registration.read_text()) != registration:
                raise RuntimeError("Existing run belongs to another experiment or its source changed")
            experiment = json.loads(path.read_text())
            if input("Continue from each arm's last verified checkpoint? [y/N]: ").strip().lower() != "y":
                return
        else:
            experiment = deepcopy(original)
            experiment["updates"] = ask_integer("Optimizer updates per arm", config["default_updates"], original["updates"] + 1)
            interval = ask_integer("Checkpoint interval in updates", config["default_checkpoint_every"], 1)
            experiment["provenance"]["settings"]["checkpoint_every"] = interval
            for source in (Path(__file__), ROOT / config_path, ROOT / "scripts/run_training_budget.sh"):
                experiment["provenance"]["sources"][source.relative_to(ROOT).as_posix()] = file_sha256(source)
            atomic_json(directory / "budget_reference.json", registration)
            atomic_json(path, experiment)
        training = [r for r in ready["episodes"] if r["episode_index"] in ready["selection"]["split"]["train"]]
        variants = len(read_jsonl(ROOT / settings["training_catalog"]))
        schedule = longer_schedule(training, experiment["updates"], experiment["seed"], settings["language_seed"], variants, old_schedule)
        schedule_path = directory / "schedule.json"
        if schedule_path.exists() and json.loads(schedule_path.read_text()) != schedule:
            raise RuntimeError("Saved frame/language schedule changed")
        atomic_json(schedule_path, schedule)
        interval = experiment["provenance"]["settings"]["checkpoint_every"]
        print(f"{experiment['updates']} updates per arm; checkpoint every {interval} and at completion; each checkpoint is about 2.2 GB.", flush=True)
        print("Default 1000 updates / interval 500 saves two checkpoints per arm, about 9 GB total. Live logs report measured speed.", flush=True)
        finals = {}
        for arm in settings["arms"]:
            print("Start " + arm, flush=True)
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/train_adaptation_arm.py")], cwd=ROOT,
                env=dict(os.environ, SMOLVLA_ADAPTATION_DIR=str(directory), SMOLVLA_ADAPTATION_ARM=arm), check=True)
            finals[arm] = json.loads((directory / arm / "finished.json").read_text())
        logs = [read_jsonl(directory / a / finals[a]["training_log"]) for a in settings["arms"]]
        controls = compare_training_logs(*logs, schedule)
        expected = json.loads((reference / "original_language/initial_parameters.json").read_text())
        for arm in settings["arms"]:
            if json.loads((directory / arm / "initial_parameters.json").read_text()) != expected:
                raise RuntimeError("Starting parameters/precision differ from the reference")
        controls["same_initial_parameter_bytes"] = True
        previous_report = json.loads((reference / "comparison.json").read_text())
        for arm, history in zip(settings["arms"], logs, strict=True):
            old_history = read_jsonl(reference / arm / previous_report["arms"][arm]["training_log"])
            require_reference_controls(old_history, history[:len(old_history)])
        atomic_json(directory / "comparison.json", {"passed": True, "updates_per_arm": experiment["updates"],
            "schedule_hash": schedule_hash(schedule), "controls": controls, "arms": finals,
            "reference_sampling_rng_prefix_equal": True,
            "schedule_note": "Longer budget and longer cosine decay horizon; LR prefix differs from 200-update reference.",
            "evaluation_status": "Not run; no robustness improvements claimed"})
        print("PASS: longer training completed for both arms; paired controls and reference sampling/RNG prefix checked.", flush=True)
        print("Saved " + (directory / "comparison.json").relative_to(ROOT).as_posix(), flush=True)
        print("Simulator evaluation is next; training completion is not evidence of success-rate improvement.", flush=True)


if __name__ == "__main__":
    main()
