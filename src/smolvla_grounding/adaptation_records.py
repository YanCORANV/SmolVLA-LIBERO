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


def build_anchored_schedule(episodes, updates, seed, language_seed, variant_count=5):
    """Balanced block sampling with P(original=0) = 0.50, P(paraphrases=1..4) = 0.125 each."""
    if updates < 2 or variant_count != 5:
        raise ValueError("Anchored schedule requires at least two updates and exactly 5 variants")
    # Separate local RNGs: language selection never changes frame sampling or policy RNG.
    frame_rng, language_rng = random.Random(seed), random.Random(language_seed)
    positions = [(r["episode_index"], frame) for r in sorted(episodes, key=lambda x: x["episode_index"])
                 for frame in range(r["frames"])]
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Training frames must be nonempty and uniquely indexed")
    schedule, block = [], []
    for step in range(1, updates + 1):
        if not block:
            block = [0, 0, 0, 0, 1, 2, 3, 4]
            language_rng.shuffle(block)
        ep, frame = positions[frame_rng.randrange(len(positions))]
        schedule.append({"step": step, "episode_index": ep, "frame_index": frame, "anchored_variant": block.pop()})
    return schedule


def build_paired_schedule(episodes, updates, seed, language_seed, paraphrase_variants=(1, 2, 3, 4)):
    """Balanced paired schedule: exactly equal sampling across paraphrase variants (250 each over 1000 steps)."""
    if updates < 2:
        raise ValueError("Updates must be at least 2")
    if updates % len(paraphrase_variants) != 0:
        raise ValueError(f"Updates {updates} must be divisible by variant count {len(paraphrase_variants)}")
    frame_rng = random.Random(seed)
    language_rng = random.Random(language_seed)
    positions = [(r["episode_index"], frame) for r in sorted(episodes, key=lambda x: x["episode_index"])
                 for frame in range(r["frames"])]
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Training frames must be nonempty and uniquely indexed")
    schedule = []
    block = []
    for step in range(1, updates + 1):
        if not block:
            block = list(paraphrase_variants)
            language_rng.shuffle(block)
        ep, frame = positions[frame_rng.randrange(len(positions))]
        schedule.append({
            "step": step,
            "episode_index": ep,
            "frame_index": frame,
            "canonical_variant": 0,
            "paraphrase_variant": block.pop()
        })
    return schedule


def paired_flow_seed(training_seed, step):
    """Isolated flow RNG identity independent of arm and global RNG consumption."""
    if step < 1:
        raise ValueError("Paired update steps start at 1")
    payload = f"paired-flow-v1:{training_seed}:{step}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def compare_paired_training_logs(bc_rows, consistency_rows, schedule, training_seed):
    """Require every registered training control to match at every update."""
    if len(bc_rows) != len(schedule) or len(consistency_rows) != len(schedule):
        raise ValueError("Paired training histories do not cover the full schedule")
    keys = ("step", "episode_index", "frame_index", "paraphrase_variant",
            "data_hash", "action_hash", "flow_seed", "noise_hash", "time_hash", "lr_used")
    for planned, bc, consistency in zip(schedule, bc_rows, consistency_rows):
        for row in (bc, consistency):
            for key in ("step", "episode_index", "frame_index", "paraphrase_variant"):
                if row[key] != planned[key]:
                    raise ValueError(f"Schedule mismatch at step {planned['step']}: {key}")
            if row["flow_seed"] != paired_flow_seed(training_seed, planned["step"]):
                raise ValueError(f"Flow seed mismatch at step {planned['step']}")
        for key in keys:
            if bc[key] != consistency[key]:
                raise ValueError(f"Paired arms differ at step {planned['step']}: {key}")
    return {"updates_compared": len(schedule), "matched_fields": list(keys),
            "all_updates_identical": True}

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


def compare_anchored_training_logs(original, anchored, schedule):
    if len(original) != len(schedule) or len(anchored) != len(schedule):
        raise ValueError("Training histories do not cover the fixed budget")
    noise_differences = []
    for left, right, planned in zip(original, anchored, schedule):
        for row in (left, right):
            if any(row[k] != planned[k] for k in ("step", "episode_index", "frame_index")):
                raise ValueError("Training histories differ from the paired frame schedule")
        if left["variant_index"] != 0 or right["variant_index"] != planned["anchored_variant"]:
            raise ValueError("Training language assignment differs from the registered anchored comparison")
        if left["data_hash"] != right["data_hash"] or left["lr_used"] != right["lr_used"]:
            raise ValueError("Training arms received different frame tensors or learning rates")
        if left["noise_hash"] != right["noise_hash"] or left["rng_before"] != right["rng_before"]:
            noise_differences.append(planned["step"])
    return {"same_frame_tensors": True, "same_learning_rates": True, "same_update_budget": True,
            "different_noise_or_rng_steps": noise_differences,
            "interpretation": "Training controls only; language robustness requires simulator evaluation"}
