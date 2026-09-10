"""Freeze an interactive paired training plan, then run each arm in its own process."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.adaptation_records import build_schedule, compare_training_logs, read_jsonl, schedule_hash, validate_training_catalog
from smolvla_grounding.benchmark_records import validate_run_name
from smolvla_grounding.protocol import atomic_json, file_sha256, fingerprint, run_lock


def ask_integer(label, default, minimum):
    while True:
        try:
            value = int(input(f"{label} [{default}]: ").strip() or str(default))
            if not minimum <= value < 2**32:
                raise ValueError(f"Enter an integer in [{minimum}, {2**32 - 1}]")
            return value
        except ValueError as exc:
            print(str(exc), flush=True)


def main():
    settings = json.loads((ROOT / "configs/adaptation.json").read_text())
    profile = ROOT / settings["profile_directory"]
    step_report = json.loads((profile / "step_report.json").read_text())
    reload_report = json.loads((profile / "reload_report.json").read_text())
    if not step_report["passed"] or not reload_report["passed"] or not reload_report["all_parameter_bytes_equal"]:
        raise RuntimeError("Complete the real optimizer-step/reload profile first")
    profile_config = json.loads((ROOT / settings["profile_config"]).read_text())
    ready = json.loads((ROOT / profile_config["data_ready"]).read_text())
    if not ready["passed"] or file_sha256(ROOT / profile_config["data_ready"]) != step_report["data_ready_sha256"]:
        raise RuntimeError("Data differs from the successful training profile")
    rows = read_jsonl(ROOT / settings["training_catalog"])
    validate_training_catalog(rows, read_jsonl(ROOT / settings["development_catalog"]), step_report["dataset"]["instruction"])
    print("Training language catalog (existing evaluation paraphrases are excluded):", flush=True)
    for row in rows:
        print("  " + row["id"] + ": " + row["text"], flush=True)
    while True:
        try:
            name = validate_run_name(input("Training run name [task0_adaptation]: ").strip() or "task0_adaptation")
            break
        except ValueError as exc:
            print(str(exc), flush=True)
    directory = ROOT / settings["output_root"] / "runs" / name
    names = ["configs/adaptation.json", settings["profile_config"], settings["training_catalog"], settings["development_catalog"],
             "scripts/run_adaptation.py", "scripts/train_adaptation_arm.py", "scripts/run_adaptation.sh",
             "src/smolvla_grounding/adaptation_records.py", "src/smolvla_grounding/training_policy.py",
             "src/smolvla_grounding/training_profile_records.py", "src/smolvla_grounding/training_data.py",
             "src/smolvla_grounding/benchmark_records.py", "src/smolvla_grounding/protocol.py",
             "src/smolvla_grounding/diagnostic_records.py", "src/smolvla_grounding/diagnostic_policy.py"]
    upstream = ["scripts/lerobot_train.py", "datasets/dataset_reader.py", "datasets/dataset_metadata.py",
                "datasets/lerobot_dataset.py", "datasets/io_utils.py", "common/train_utils.py",
                "policies/pretrained.py", "policies/smolvla/modeling_smolvla.py", "policies/smolvla/smolvlm_with_expert.py",
                "optim/optimizers.py", "optim/schedulers.py", "utils/random_utils.py"]
    upstream = [".venv/lib/python3.12/site-packages/lerobot/" + p for p in upstream]
    provenance = {"settings": settings, "upstream_sources": {p: file_sha256(ROOT / p) for p in upstream},
                  "profile": {p: file_sha256(profile / p) for p in ("step_report.json", "reload_report.json")},
                  "data_ready": step_report["data_ready_sha256"], "sources": {p: file_sha256(ROOT / p) for p in names}}
    with run_lock(directory):
        path = directory / "experiment.json"
        if path.exists():
            experiment = json.loads(path.read_text())
            if experiment["provenance"] != provenance:
                raise RuntimeError("Saved experiment code/config/data changed; preserve it and choose a new run name")
            print(f"Saved budget: {experiment['updates']} updates per arm; seed {experiment['seed']}.", flush=True)
            if input("Continue both arms from their last verified checkpoints? [y/N]: ").strip().lower() != "y":
                return
        else:
            experiment = {"provenance": provenance, "updates": ask_integer("Optimizer updates per arm", settings["default_updates"], 2),
                          "seed": ask_integer("Training seed", settings["default_seed"], 0)}
            atomic_json(path, experiment)
        training = [r for r in ready["episodes"] if r["episode_index"] in ready["selection"]["split"]["train"]]
        schedule = build_schedule(training, experiment["updates"], experiment["seed"], settings["language_seed"], len(rows))
        schedule_path = directory / "schedule.json"
        if schedule_path.exists() and json.loads(schedule_path.read_text()) != schedule:
            raise RuntimeError("Saved frame/language schedule changed")
        atomic_json(schedule_path, schedule)
        print(f"Two serial training arms, {experiment['updates']} updates each, batch 1. Checkpoints every {settings['checkpoint_every']} updates and at the end.", flush=True)
        print("Elapsed first-step profiling is not a reliable total-time estimate; live progress will report observed throughput.", flush=True)
        print("Each saved checkpoint keeps full policy/optimizer state (about 2.2 GB); no automatic deletion.", flush=True)
        env = dict(os.environ, SMOLVLA_ADAPTATION_DIR=str(directory))
        finals = {}
        for arm in settings["arms"]:
            print("Start " + arm, flush=True)
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/train_adaptation_arm.py")],
                           env=dict(env, SMOLVLA_ADAPTATION_ARM=arm), cwd=ROOT, check=True)
            finals[arm] = json.loads((directory / arm / "finished.json").read_text())
        logs = [read_jsonl(directory / arm / finals[arm]["training_log"]) for arm in settings["arms"]]
        initial = [json.loads((directory / arm / "initial_parameters.json").read_text()) for arm in settings["arms"]]
        if initial[0] != initial[1]:
            raise RuntimeError("Training arms did not start with identical parameters and precision")
        comparison = compare_training_logs(*logs, schedule)
        comparison["same_initial_parameter_bytes"] = True
        atomic_json(directory / "comparison.json", {"passed": True, "updates_per_arm": experiment["updates"],
            "schedule_hash": schedule_hash(schedule), "controls": comparison, "arms": finals,
            "evaluation_status": "Not run; no robustness improvements claimed"})
        print("PASS: both fine-tuning arms saved; paired training controls checked. Simulator evaluation is next.", flush=True)
        print("Saved " + (directory / "comparison.json").relative_to(ROOT).as_posix(), flush=True)


if __name__ == "__main__":
    main()
