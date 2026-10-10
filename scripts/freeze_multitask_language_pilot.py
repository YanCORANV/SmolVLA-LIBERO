"""Create the immutable pilot manifest from the completed screen's pinned assets."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.multitask_pilot_protocol import MATRIX, MANIFEST, TASKS, STATES, PROMPTS
from smolvla_grounding.multitask_protocol import _inspect_task_asset, prompt_sha256
from smolvla_grounding.protocol import atomic_json, file_sha256

SCREEN = ROOT / "configs/multitask_evaluation/canonical_screen_manifest.json"
SOURCES = (
    "src/smolvla_grounding/multitask_protocol.py",
    "src/smolvla_grounding/multitask_evaluator.py",
    "src/smolvla_grounding/multitask_pilot_protocol.py",
    "src/smolvla_grounding/multitask_pilot_evaluator.py",
    "scripts/run_multitask_language_pilot.py",
    "scripts/freeze_multitask_language_pilot.py",
)


def freeze():
    matrix = json.loads((ROOT / MATRIX).read_text(encoding="utf-8"))
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if matrix["role"] != "multitask_language_robustness" or len(matrix["tasks"]) != 5:
        raise RuntimeError("Not the approved five-task pilot")
    manifest = {
        "role": "multitask_language_robustness",
        "matrix_sha256": file_sha256(ROOT / MATRIX),
        "source_sha256": {rel: file_sha256(ROOT / rel) for rel in SOURCES},
        "models": screen["models"], "backbone": screen["backbone"],
        "assets": {}, "catalog_sha256": {}, "prompt_sha256": {},
    }
    for task in matrix["tasks"]:
        key = (task["suite"], task["task_id"])
        if key not in TASKS or task["states"] != STATES:
            raise RuntimeError(f"Unapproved task or state set: {key}")
        asset = _inspect_task_asset(*key)
        frozen = screen["assets"][f"{key[0]}:{key[1]}"]
        if asset["name"] != task["task_name"] or asset["language"] != task["canonical"]:
            raise RuntimeError(f"Official task identity changed: {key}")
        for kind in ("bddl", "init"):
            if str(asset[kind]) != frozen[f"{kind}_path"] or file_sha256(asset[kind]) != frozen[f"{kind}_sha256"]:
                raise RuntimeError(f"Official task asset changed: {key}/{kind}")
        if asset["state_count"] != frozen["state_count"]:
            raise RuntimeError(f"Init-state count changed: {key}")
        manifest["assets"][f"{key[0]}:{key[1]}"] = frozen
        catalog = ROOT / task["prompt_catalog"]
        rows = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines() if line.strip()]
        if [row["id"] for row in rows] != PROMPTS or rows[0]["text"] != task["canonical"]:
            raise RuntimeError(f"Wrong prompt catalog: {key}")
        manifest["catalog_sha256"][task["prompt_catalog"]] = file_sha256(catalog)
        for row in rows:
            manifest["prompt_sha256"][f"{key[0]}:{key[1]}:{row['id']}"] = prompt_sha256(row["text"])
    path = ROOT / MANIFEST
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != manifest:
            raise RuntimeError("Existing pilot manifest differs; do not overwrite")
        print("Pilot manifest unchanged", flush=True)
    else:
        atomic_json(path, manifest)
        print(f"Pilot manifest frozen: {path}", flush=True)


if __name__ == "__main__":
    freeze()
