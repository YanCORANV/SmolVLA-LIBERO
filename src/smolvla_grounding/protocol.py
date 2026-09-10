"""Dependency-light protocol validation and durable per-condition checkpoints."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import socket


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_protocol(root):
    root = Path(root)
    config = json.loads((root / "configs/language_pilot.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (root / config["instruction_catalog"])
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_catalog(rows, config["goals"])
    if config["init_state_index"] < 0 or config["environment_seed"] < 0:
        raise ValueError("Initial-state index and seed must be nonnegative")
    return config, rows


def validate_catalog(rows, goals):
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or not all(re.fullmatch(r"[a-z][a-z0-9_]*", x) for x in ids):
        raise ValueError("Instruction IDs must be unique safe directory names")
    originals = [r for r in rows if r["condition"] == "original"]
    if len(originals) != 1 or rows[0] != originals[0]:
        raise ValueError("Exactly one original instruction must come first")
    original = originals[0]
    if original["goal_id"] != "original_bowl":
        raise ValueError("Original goal must match the task-0 BDDL goal")
    texts = set()
    for row in rows:
        kind, text = row["condition"], row["text"]
        if kind not in {"original", "original_repeat", "paraphrase", "semantic_change"}:
            raise ValueError(f"Unknown condition: {kind}")
        if not text or text != text.strip() or "\n" in text:
            raise ValueError("Instructions must be nonempty single lines without outer whitespace")
        if row["goal_id"] not in goals:
            raise ValueError("Unknown goal")
        if kind in {"original_repeat", "paraphrase"} and row["goal_id"] != original["goal_id"]:
            raise ValueError("Repeat/paraphrase must preserve the original goal")
        if kind == "semantic_change" and row["goal_id"] == original["goal_id"]:
            raise ValueError("Changed instruction must change the goal")
        if kind == "original_repeat":
            if text != original["text"]:
                raise ValueError("Original-repeat text must be identical")
        elif text in texts:
            raise ValueError("Duplicate instruction outside the repeat control")
        texts.add(text)


@contextmanager
def run_lock(directory):
    """Reject concurrent writers; ask before removing a dead local process's lock."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "run.lock"
    if lock.exists():
        owner = json.loads(lock.read_text(encoding="utf-8"))
        if owner["host"] != socket.gethostname():
            raise RuntimeError("Lock belongs to another host; inspect its process before recovery")
        try:
            os.kill(owner["pid"], 0)
        except ProcessLookupError:
            answer = input("The previous process ended but left a lock. Recover this run? [y/N]: ")
            if answer.strip().lower() != "y":
                raise RuntimeError("Existing run preserved")
            lock.unlink()
        else:
            raise RuntimeError(f"Another process may be using this output: PID {owner['pid']}")
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, stream)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


class Checkpoints:
    """Reuse only verified complete records with matching protocol provenance.

    Caller holds run_lock. Each incomplete attempt is retained for diagnosis.
    """

    def __init__(self, directory, identity):
        self.directory = Path(directory)
        identity_path = self.directory / "identity.json"
        self.fingerprint = fingerprint(identity)
        if identity_path.exists():
            previous = json.loads(identity_path.read_text(encoding="utf-8"))
            if previous != identity:
                raise RuntimeError("Protocol/provenance changed. Preserve this run and select a new "
                                   "pairing_output directory in configs/language_pilot.json.")
            answer = input("Existing pairing checks found. Continue and verify completed checks? [y/N]: ")
            if answer.strip().lower() != "y":
                raise RuntimeError("Existing run preserved")
        else:
            atomic_json(identity_path, identity)

    def completed(self, instruction_id):
        path = self.directory / instruction_id / "complete.json"
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["fingerprint"] != self.fingerprint:
            raise RuntimeError("Completed condition has different provenance")
        for name, expected in record["artifacts"].items():
            artifact = self.directory / name
            if not artifact.is_file() or file_sha256(artifact) != expected:
                raise RuntimeError(f"Completed artifact missing or modified: {name}")
        return record["result"]

    def new_attempt(self, instruction_id):
        number = 1
        while True:
            path = self.directory / instruction_id / f"attempt_{number:03d}"
            try:
                path.mkdir(parents=True, exist_ok=False)
                return path
            except FileExistsError:
                number += 1

    def finish(self, instruction_id, result, artifacts):
        hashes = {Path(p).relative_to(self.directory).as_posix(): file_sha256(p) for p in artifacts}
        atomic_json(self.directory / instruction_id / "complete.json", {
            "fingerprint": self.fingerprint, "result": result, "artifacts": hashes,
        })
