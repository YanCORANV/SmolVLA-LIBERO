"""Create the canonical screen's immutable input registry before rollout 1.

This is an authoring utility. It never creates an environment or policy.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import try_to_load_from_cache

from smolvla_grounding.multitask_protocol import _inspect_task_asset, prompt_sha256
from smolvla_grounding.protocol import atomic_json, file_sha256

MATRIX = ROOT / "configs/multitask_evaluation/canonical_screen_matrix.json"
MANIFEST = ROOT / "configs/multitask_evaluation/canonical_screen_manifest.json"
IDENTITY = ROOT / "outputs/language_pilot/identity.json"


def build_manifest():
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    baseline = json.loads(IDENTITY.read_text(encoding="utf-8"))
    if matrix["role"] != "canonical_competence_screen":
        raise RuntimeError("Wrong matrix role")
    if matrix["eligibility"] != {"minimum_successes": 3, "trials_per_task": 5}:
        raise RuntimeError("Eligibility gate changed")
    if matrix["expected_rollouts"] != 30:
        raise RuntimeError("Screen matrix count changed")
    backbone = baseline["backbone"]
    cached = try_to_load_from_cache(backbone["repo_id"], "config.json",
                                    revision=backbone["revision"])
    if not isinstance(cached, str):
        raise RuntimeError("Pinned backbone snapshot missing")
    manifest = {
        "role": "canonical_competence_screen",
        "matrix_sha256": file_sha256(MATRIX),
        "source_sha256": {
            rel: file_sha256(ROOT / rel)
            for rel in (
                "src/smolvla_grounding/multitask_protocol.py",
                "src/smolvla_grounding/multitask_evaluator.py",
                "scripts/run_multitask_canonical_screen.py",
            )
        },
        "models": {
            "untuned_fp32": {
                "kind": "official_smolvla",
                "path": baseline["checkpoint"]["local_dir"],
                "revision": baseline["checkpoint"]["revision"],
                "artifact_sha256": baseline["policy_hashes"],
            },
        },
        "backbone": {
            "repo_id": backbone["repo_id"], "revision": backbone["revision"],
            "artifact_sha256": baseline["backbone_hashes"],
        },
        "assets": {},
        "catalog_sha256": {},
        "prompt_sha256": {},
    }
    for task in matrix["tasks"]:
        suite, task_id = task["suite"], task["task_id"]
        asset = _inspect_task_asset(suite, task_id)
        if asset["name"] != task["task_name"] or asset["language"] != task["canonical"]:
            raise RuntimeError(f"Task identity changed: {suite}/{task_id}")
        if asset["state_count"] != 50:
            raise RuntimeError(f"Expected 50 active states: {suite}/{task_id}")
        manifest["assets"][f"{suite}:{task_id}"] = {
            "bddl_path": str(asset["bddl"]), "bddl_sha256": file_sha256(asset["bddl"]),
            "init_path": str(asset["init"]), "init_sha256": file_sha256(asset["init"]),
            "state_count": asset["state_count"],
        }
        catalog = ROOT / task["prompt_catalog"]
        rows = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if len(rows) != 1 or rows[0]["id"] != "original" or rows[0]["text"] != task["canonical"]:
            raise RuntimeError(f"Wrong canonical catalog: {suite}/{task_id}")
        manifest["catalog_sha256"][task["prompt_catalog"]] = file_sha256(catalog)
        manifest["prompt_sha256"][f"{suite}:{task_id}:original"] = prompt_sha256(rows[0]["text"])
    if MANIFEST.exists():
        previous = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError("Existing screen manifest differs; preserve it and investigate")
        print("Screen manifest already matches; no file changed", flush=True)
    else:
        atomic_json(MANIFEST, manifest)
        print(f"Screen manifest created: {MANIFEST}", flush=True)


if __name__ == "__main__":
    build_manifest()
