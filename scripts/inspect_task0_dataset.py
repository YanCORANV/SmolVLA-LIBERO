"""Verify the official task-0 demonstration metadata without loading the policy."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.protocol import atomic_json, run_lock
from smolvla_grounding.training_data import fetch_metadata, inspect_metadata


def main():
    config = json.loads((ROOT / "configs/task0_dataset.json").read_text())
    directory = ROOT / "outputs/dataset_metadata" / config["revision"]
    with run_lock(directory):
        print("[1/2] Verify pinned official metadata", flush=True)
        fetch_metadata(config, directory)
        print("[2/2] Match the official instruction and enumerate its demonstration episodes", flush=True)
        report = inspect_metadata(directory, config)
        catalog = [json.loads(s) for s in (ROOT / "configs/instructions/task0.jsonl").read_text().splitlines() if s]
        if catalog[0]["text"] != report["original_instruction"] or catalog[0]["condition"] != "original":
            raise RuntimeError("Evaluation original differs from the dataset instruction")
        atomic_json(directory / "audit.json", report)
        print(f"PASS: exact A match; {report['selected_episode_count']} demonstrations, {report['selected_frame_count']} frames.", flush=True)
        print(f"Required shards: {report['required_download_bytes'] / 1e9:.2f} GB. Demonstrations not downloaded or trained.", flush=True)
        print("Saved " + (directory / "audit.json").relative_to(ROOT).as_posix(), flush=True)


if __name__ == "__main__":
    main()
