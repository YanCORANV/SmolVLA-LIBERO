"""Import hash-verified reference episodes without executing an unchanged policy."""

import json
from pathlib import Path
import shutil

from .protocol import file_sha256


def import_completed(source, destination, key):
    result = source.completed(key)
    if result is None:
        raise RuntimeError("Reference record is incomplete: " + key)
    existing = destination.completed(key)
    if existing is not None:
        if existing != result:
            raise RuntimeError("Imported reference result changed: " + key)
        return existing
    marker = json.loads((source.directory / key / "complete.json").read_text())
    artifacts = []
    for name, expected in marker["artifacts"].items():
        origin, target = source.directory / name, destination.directory / name
        for base, path in ((source.directory, origin), (destination.directory, target)):
            if not path.resolve().is_relative_to(base.resolve()):
                raise ValueError("Artifact path escapes its experiment directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or file_sha256(target) != expected:
            temporary = target.with_name(target.name + ".importing")
            shutil.copyfile(origin, temporary)
            if file_sha256(temporary) != expected:
                raise RuntimeError("Reference artifact changed during import: " + name)
            temporary.replace(target)
        artifacts.append(target)
    destination.finish(key, result, artifacts)
    return result


def require_same_initial_parameters(root, reference_experiment, training):
    previous = reference_experiment["provenance"]["training"]["models"]["untuned_fp32"]["parameters"]
    current = training["models"]["untuned_fp32"]["parameters"]
    if json.loads((Path(root) / previous).read_text()) != json.loads((Path(root) / current).read_text()):
        raise RuntimeError("The new run's starting parameters/precision differ from the reference baseline")
