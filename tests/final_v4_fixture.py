"""Temporary candidate freeze for tests; never writes production frozen artifacts."""

from copy import deepcopy
import json
from pathlib import Path

from smolvla_grounding.protocol import file_sha256


ASSET_REL = (".venv/lib/python3.12/site-packages/libero/libero/init_files/"
             "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
             "and_place_it_on_the_plate.pruned_init")


def candidate_v4_freeze(root, manifest):
    root = Path(root)
    frozen = deepcopy(json.loads(
        (root / "configs/final_confirmatory/freeze_final_protocol_v3.json").read_text(encoding="utf-8")
    ))
    frozen["version"] = 4
    frozen["historical_v3_freeze_path"] = "configs/final_confirmatory/freeze_final_protocol_v3.json"
    frozen["historical_v3_freeze_sha256"] = file_sha256(root / frozen["historical_v3_freeze_path"])
    frozen["active_initial_state_source"] = {
        "path": str((root / ASSET_REL).resolve()),
        "sha256": file_sha256(root / ASSET_REL),
        "shape": [50, 92],
    }
    frozen["final_protocol_artifacts"]["matrix_sha256"] = manifest["final_matrix_sha256"]
    frozen["final_protocol_artifacts"]["manifest_sha256"] = file_sha256(
        root / "configs/final_confirmatory/frozen_model_manifest_v4.json"
    )
    frozen["final_protocol_artifacts"]["file_sha256"].update({
        rel: file_sha256(root / rel) for rel in (
            "configs/final_confirmatory/final_matrix_v4.json",
            "configs/final_confirmatory/frozen_model_manifest_v4.json",
        )
    })
    for rel in frozen["evaluation_runtime_provenance"]["file_sha256"]:
        frozen["evaluation_runtime_provenance"]["file_sha256"][rel] = file_sha256(root / rel)
    return frozen
