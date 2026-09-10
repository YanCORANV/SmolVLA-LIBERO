"""Language isolation, paired sample schedules and checkpoint recovery contracts."""

import hashlib
import json
from pathlib import Path
import random
import re


def normalized_words(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def validate_training_catalog(rows, development, original):
    if len(rows) < 2 or rows[0]["text"] != original or rows[0]["role"] != "original":
        raise ValueError("First training instruction must be the official original")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training instruction ID")
    texts = [normalized_words(r["text"]) for r in rows]
    if len(set(texts)) != len(texts):
        raise ValueError("Duplicate training wording after punctuation/case normalization")
    forbidden = {normalized_words(r["text"]) for r in development if r["condition"] not in ("original", "original_repeat")}
    for index, row in enumerate(rows):
        if row["goal_id"] != "original_bowl" or (index and row["role"] != "paraphrase"):
            raise ValueError("Only meaning-preserving training labels are allowed")
        if not row["text"].strip() or "\n" in row["text"] or (index and texts[index] in forbidden):
            raise ValueError("Training instruction overlaps development or is invalid")


def build_schedule(episodes, updates, seed, language_seed, variant_count):
    if updates < 2 or variant_count < 2:
        raise ValueError("Use at least two updates and two language choices")
    # Separate local RNGs: language selection never changes frame sampling or policy RNG.
    frame_rng, language_rng = random.Random(seed), random.Random(language_seed)
    positions = [(r["episode_index"], frame) for r in sorted(episodes, key=lambda x: x["episode_index"])
                 for frame in range(r["frames"])]
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Training frames must be nonempty and uniquely indexed")
    schedule, block = [], []
    for step in range(1, updates + 1):
        if not block:
            block = list(range(variant_count))
            language_rng.shuffle(block)
        ep, frame = positions[frame_rng.randrange(len(positions))]
        schedule.append({"step": step, "episode_index": ep, "frame_index": frame, "augmented_variant": block.pop()})
    return schedule


def checkpoint_steps(updates, interval):
    if updates < 2 or interval < 1:
        raise ValueError("Invalid update/checkpoint interval")
    return sorted(set(range(interval, updates + 1, interval)) | {updates})


def latest_checkpoint(journal, updates, interval):
    # Search known save boundaries only. An incomplete later attempt cannot displace a completed save.
    for step in reversed(checkpoint_steps(updates, interval)):
        record = journal.completed(f"step_{step:06d}")
        if record is not None:
            if record["step"] != step:
                raise ValueError("Checkpoint marker step mismatch")
            return record
    return None


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def schedule_hash(schedule):
    return hashlib.sha256(json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compare_training_logs(original, augmented, schedule):
    if len(original) != len(schedule) or len(augmented) != len(schedule):
        raise ValueError("Training histories do not cover the fixed budget")
    noise_differences = []
    for left, right, planned in zip(original, augmented, schedule):
        for row in (left, right):
            if any(row[k] != planned[k] for k in ("step", "episode_index", "frame_index")):
                raise ValueError("Training histories differ from the paired frame schedule")
        if left["variant_index"] != 0 or right["variant_index"] != planned["augmented_variant"]:
            raise ValueError("Training language assignment differs from the registered comparison")
        if left["data_hash"] != right["data_hash"] or left["lr_used"] != right["lr_used"]:
            raise ValueError("Training arms received different frame tensors or learning rates")
        if left["noise_hash"] != right["noise_hash"] or left["rng_before"] != right["rng_before"]:
            noise_differences.append(planned["step"])
    return {"same_frame_tensors": True, "same_learning_rates": True, "same_update_budget": True,
            "different_noise_or_rng_steps": noise_differences,
            "interpretation": "Training controls only; language robustness requires simulator evaluation"}
