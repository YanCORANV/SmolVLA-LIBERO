"""Interactively split and fetch task-0 data with verified per-shard progress."""

import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.protocol import atomic_json, file_sha256, fingerprint, run_lock
from smolvla_grounding.training_data import (
    fetch_metadata, inspect_metadata, split_episodes, verify_file,
    validate_saved_selection, write_prepared_metadata,
)


def main():
    config = json.loads((ROOT / "configs/task0_dataset.json").read_text())
    directory = ROOT / config["output_dir"]
    dataset = ROOT / config["local_root"]
    started = time.perf_counter()
    with run_lock(directory):
        print("[1/3] Verify official metadata and freeze the demonstration split", flush=True)
        source_metadata = ROOT / "outputs/dataset_metadata" / config["revision"]
        fetch_metadata(config, source_metadata)
        audit = inspect_metadata(source_metadata, config)
        sources = ["scripts/prepare_task0_data.py", "src/smolvla_grounding/training_data.py",
                   "src/smolvla_grounding/demonstration_validation.py"]
        identity = {"config": fingerprint(config), "sources": {p: file_sha256(ROOT / p) for p in sources}}
        selection_path = directory / "selection.json"
        ids = [r["episode_index"] for r in audit["episodes"]]
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            expected = validate_saved_selection(selection, config, ids)
            print(f"Saved split: {len(expected['train'])} train / {len(expected['held_out'])} held-out demonstrations.", flush=True)
            if selection["identity"] != identity:
                print("Preparation code/file locations were corrected. Keep the exact split and revalidate all shard evidence.", flush=True)
            if input("Continue and verify/reuse completed downloads? [y/N]: ").strip().lower() != "y":
                return
            if selection["identity"] != identity:
                history = directory / "selection_history" / (fingerprint(selection) + ".json")
                atomic_json(history, selection)
                selection = dict(selection, identity=identity)
                atomic_json(selection_path, selection)
        else:
            while True:
                try:
                    count = int(input(f"Training demonstration count [{config['default_train_episodes']} of {len(ids)}]: ").strip()
                                or str(config["default_train_episodes"]))
                    split = split_episodes(ids, count, config["split_seed"])
                    break
                except ValueError as exc:
                    print(str(exc), flush=True)
            selection = {"identity": identity, "repo_id": config["repo_id"], "revision": config["revision"], "split": split}
            atomic_json(selection_path, selection)
        for record in config["metadata"]:
            target = dataset / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_metadata / record["path"], target)
        atomic_json(directory / "metadata_audit.json", audit)
        print(f"[2/3] Download and validate {len(config['data_files'])} shared shards ({audit['required_download_bytes'] / 1e9:.2f} GB)", flush=True)
        print("Shared shards also contain other tasks; only the 45 exact-matched episodes are selected. No GPU/model import.", flush=True)
        from huggingface_hub import hf_hub_download
        from smolvla_grounding.demonstration_validation import validate_shard

        completed = []
        for index, record in enumerate(config["data_files"], 1):
            print(f"  [{index}/{len(config['data_files'])}] {record['path']} ({record['bytes'] / 1e6:.1f} MB)", flush=True)
            path = dataset / record["path"]
            if not verify_file(path, record):
                if path.exists():
                    raise RuntimeError("Existing shard failed checksum; preserved for inspection: " + record["path"])
                hf_hub_download(repo_id=config["repo_id"], repo_type="dataset", revision=config["revision"],
                                filename=record["path"], local_dir=dataset)
                if not verify_file(path, record):
                    raise RuntimeError("Downloaded shard failed SHA-256 verification")
            marker = directory / "shards" / (path.stem + ".json")
            metadata = [r for r in audit["episodes"] if record["path"] ==
                        f"data/chunk-{r['data/chunk_index']:03d}/file-{r['data/file_index']:03d}.parquet"]
            cache_identity = fingerprint({"identity": identity, "file": record, "episodes": metadata})
            cached = json.loads(marker.read_text()) if marker.exists() else None
            if cached and cached.get("identity") == cache_identity and cached.get("report_hash") == fingerprint(cached["episodes"]):
                result = cached["episodes"]
                print("    SHA-256 verified; reusing completed frame checks", flush=True)
            else:
                result = validate_shard(path, metadata, config["dataset_task_index"], audit["dataset_fps"])
                atomic_json(marker, {"identity": cache_identity, "episodes": result, "report_hash": fingerprint(result)})
                print(f"    Verified {len(result)} episodes, numeric arrays and sampled images", flush=True)
            completed.extend(result)
            print(f"    Progress: {len(completed)}/{len(ids)} demonstrations; elapsed {time.perf_counter() - started:.0f} s", flush=True)
        print("[3/3] Verify complete selection and save the data readiness report", flush=True)
        if sorted(r["episode_index"] for r in completed) != sorted(ids):
            raise RuntimeError("Incomplete or duplicated prepared demonstrations")
        if sum(r["frames"] for r in completed) != config["expected_task_frames"]:
            raise RuntimeError("Wrong total frame count")
        repair = write_prepared_metadata(dataset, config, audit)
        report = {"passed": True, "selection": selection, "dataset_root": config["local_root"],
                  "prepared_metadata": repair,
                  "episodes": completed, "elapsed_seconds": time.perf_counter() - started,
                  "acceptance": "File hashes, episode/frame/task IDs, finite numeric data, timestamps and sampled image decoding only",
                  "training_status": "Not run; official loader, temporal/action conventions and optimizer memory still require acceptance",
                  "held_out_status": "Held out from our future tuning only; not a claim about upstream checkpoint exposure"}
        atomic_json(directory / "ready.json", report)
        print(f"PASS: {len(ids)} demonstrations prepared; {len(selection['split']['train'])} train / {len(selection['split']['held_out'])} held out.", flush=True)
        print("Saved " + (directory / "ready.json").relative_to(ROOT).as_posix(), flush=True)
        print("Next: official dataloader + real optimizer-step memory measurement. No fine-tuning was performed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun and choose continue to reuse verified shards and Hub partial downloads.", flush=True)
        sys.exit(130)
