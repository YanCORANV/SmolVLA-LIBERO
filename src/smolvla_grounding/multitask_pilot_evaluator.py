"""Resume-safe runner for the fixed 75-rollout language sensitivity pilot."""
from __future__ import annotations

import json
from pathlib import Path

from .multitask_evaluator import _execute_episode
from .multitask_pilot_protocol import MATRIX, MANIFEST, pilot_preflight
from .multitask_protocol import _valid_completion, new_attempt, trial_path
from .protocol import atomic_json, file_sha256, run_lock


def _paired_reset_audit(plan, task, state):
    output = plan["output_dir"]
    paths = []
    for prompt in ("original", "paraphrase_01", "paraphrase_02"):
        row = ("untuned_fp32", task["suite"], task["task_id"], state, prompt)
        trial = trial_path(output, row, plan["matrix"]["policy"]["policy_seed"])
        complete = _valid_completion(trial, row, task["horizon"], plan["identity"])
        if complete is None:
            raise RuntimeError(f"Paired cell incomplete: {trial}")
        paths.append(complete)
    initial = [json.loads((path / "initial_hashes.json").read_text(encoding="utf-8")) for path in paths]
    if initial[0] != initial[1] or initial[0] != initial[2]:
        raise RuntimeError(f"Paired initial states differ: {task['suite']}/{task['task_id']} state={state}")
    rng, noise = [], []
    for path in paths:
        first = json.loads((path / "inference.jsonl").open(encoding="utf-8").readline())
        if first["step"] != 1:
            raise RuntimeError(f"Missing Step-1 inference record: {path}")
        rng.append(first["rng_before"])
        noise.append(first["noise"])
    if rng[0] != rng[1] or rng[0] != rng[2]:
        raise RuntimeError(f"Paired Step-1 RNG differs: {task['suite']}/{task['task_id']} state={state}")
    if noise[0] != noise[1] or noise[0] != noise[2]:
        raise RuntimeError(f"Paired Step-1 flow noise differs: {task['suite']}/{task['task_id']} state={state}")


def run_pilot(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    print("[Pilot preflight] validating all 75 cells and frozen assets", flush=True)
    plan = pilot_preflight(root)
    print("[Pilot preflight] passed; no environment reset yet", flush=True)
    output = plan["output_dir"]
    with run_lock(output):
        metadata = output / "experiment_meta.json"
        if not metadata.exists():
            atomic_json(metadata, {
                "identity": plan["identity"], "matrix": plan["matrix"],
                "manifest_sha256": file_sha256(root / MANIFEST),
                "matrix_sha256": file_sha256(root / MATRIX),
            })
        tasks = {(t["suite"], t["task_id"]): t for t in plan["matrix"]["tasks"]}
        for ordinal, row in enumerate(plan["rows"], 1):
            task = tasks[row[1:3]]
            trial = trial_path(output, row, plan["matrix"]["policy"]["policy_seed"])
            if _valid_completion(trial, row, task["horizon"], plan["identity"]):
                print(f"[{ordinal:02d}/75] cached {row[1]}/{row[2]} state={row[3]} prompt={row[4]}", flush=True)
            else:
                attempt = new_attempt(trial)
                print(f"[{ordinal:02d}/75] running {row[1]}/{row[2]} state={row[3]} prompt={row[4]} {attempt.name}", flush=True)
                _execute_episode(root, plan, row, attempt)
                if _valid_completion(trial, row, task["horizon"], plan["identity"]) != attempt:
                    raise RuntimeError(f"Rollout did not commit: {attempt}")
                print(f"[{ordinal:02d}/75] completed {row[1]}/{row[2]} state={row[3]} prompt={row[4]}", flush=True)
            if row[4] == "paraphrase_02":
                _paired_reset_audit(plan, task, row[3])
                print(f"[{ordinal:02d}/75] paired reset, Step-1 RNG and flow noise verified", flush=True)
        print("[Pilot] 75/75 committed and paired initial conditions verified", flush=True)
