"""Execute both frozen paired arms against one committed data/language schedule."""

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.adaptation_records import (
    build_paired_schedule, compare_paired_training_logs, latest_checkpoint, read_jsonl, schedule_hash,
)
from smolvla_grounding.benchmark_records import BenchmarkJournal
from smolvla_grounding.protocol import atomic_json, file_sha256, run_lock


def main():
    config = json.loads((ROOT / "configs/paired_training.json").read_text(encoding="utf-8"))
    training = config["training"]
    if training["arms"] != ["paired_bc", "paired_flow_consistency"] or training["updates"] != 1000:
        raise RuntimeError("Frozen paired arm list or update budget changed")
    if training["lambda_cons"] != 1.0 or training["checkpoint_every"] != 500 or training["batch_size"] != 1:
        raise RuntimeError("Frozen paired training parameters changed")

    ready = json.loads((ROOT / training["data_ready"]).read_text(encoding="utf-8"))
    train_ids = set(ready["selection"]["split"]["train"])
    episodes = [row for row in ready["episodes"] if row["episode_index"] in train_ids]
    schedule = build_paired_schedule(episodes, 1000, training["seed"], training["language_seed"],
                                     tuple(training["paraphrase_variants"]))
    counts = Counter(row["paraphrase_variant"] for row in schedule)
    if counts != {1: 250, 2: 250, 3: 250, 4: 250}:
        raise RuntimeError(f"Unbalanced frozen paraphrase schedule: {counts}")

    directory = ROOT / "outputs/adaptation/runs/task0_paired_flow_training"
    experiment = {
        "updates": 1000,
        "seed": training["seed"],
        "paired_config_sha256": file_sha256(ROOT / "configs/paired_training.json"),
        "schedule_hash": schedule_hash(schedule),
        "data_ready_sha256": file_sha256(ROOT / training["data_ready"]),
    }
    with run_lock(directory):
        for name, value in (("experiment.json", experiment), ("schedule.json", schedule)):
            path = directory / name
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != value:
                    raise RuntimeError(f"Existing run provenance differs: {path}")
            else:
                atomic_json(path, value)
        for arm in training["arms"]:
            print(f"Starting {arm}: 1000 updates, checkpoint every 500", flush=True)
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/train_paired_arm.py")],
                           cwd=ROOT, env=dict(os.environ, SMOLVLA_ADAPTATION_DIR=str(directory),
                                              SMOLVLA_ADAPTATION_ARM=arm), check=True)
            arm_dir = directory / arm
            identity = json.loads((arm_dir / "identity.json").read_text(encoding="utf-8"))
            record = latest_checkpoint(BenchmarkJournal(arm_dir, identity), 1000, 500)
            if record is None or record["step"] != 1000:
                raise RuntimeError(f"{arm} did not commit its final checkpoint")
            print(f"Verified committed final checkpoint for {arm}", flush=True)

        if json.loads((directory / "paired_bc/initial_parameters.json").read_text(encoding="utf-8")) != json.loads(
                (directory / "paired_flow_consistency/initial_parameters.json").read_text(encoding="utf-8")):
            raise RuntimeError("Student initial parameters differ between arms")
        logs = []
        for arm in training["arms"]:
            arm_dir = directory / arm
            identity = json.loads((arm_dir / "identity.json").read_text(encoding="utf-8"))
            record = latest_checkpoint(BenchmarkJournal(arm_dir, identity), 1000, 500)
            logs.append(read_jsonl(arm_dir / record["training_log"]))
        audit = compare_paired_training_logs(*logs, schedule, training["seed"])
        audit["schedule_hash"] = schedule_hash(schedule)
        audit["initial_parameters_identical"] = True
        source = ROOT / config["source_checkpoint"]["path"]
        audit["teacher_source_checkpoint_byte_identical"] = all(
            file_sha256(source / name) == expected
            for name, expected in config["source_checkpoint"]["model_hashes"].items()
        )
        if not audit["teacher_source_checkpoint_byte_identical"]:
            raise RuntimeError("Frozen teacher/source checkpoint bytes changed")
        atomic_json(directory / "paired_control_audit.json", audit)
        print("PASS: 1000 paired controls match step by step; source checkpoint unchanged", flush=True)


if __name__ == "__main__":
    main()