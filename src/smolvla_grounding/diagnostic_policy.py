"""Observation-only tracing around official SmolVLA action selection and noise sampling."""

from contextlib import contextmanager
import json

import torch
from torch import nn

from lerobot.utils.random_utils import get_rng_state
from .diagnostic_records import encode_record, pack_rng, pack_tensors
from .libero_pairing import array_hashes


@contextmanager
def capture_noise(policy):
    model = policy.model
    if "sample_noise" in model.__dict__:
        raise RuntimeError("Unexpected existing noise override")
    original = model.sample_noise
    values = []

    def record(*args, **kwargs):
        noise = original(*args, **kwargs)
        values.append(noise.detach().clone())
        return noise

    model.sample_noise = record
    try:
        yield values
    finally:
        delattr(model, "sample_noise")


def require_empty_queue(policy):
    if set(policy._queues) != {"action"} or len(policy._queues["action"]):
        raise RuntimeError("Expected the inspected action-only empty queue with n_action_steps=1")


class TracedPolicy(nn.Module):
    def __init__(self, policy, env, archive, trace_stream):
        super().__init__()
        self.policy = policy
        self.env = env
        self.archive = archive
        self.trace_stream = trace_stream
        self.records = []

    def reset(self):
        self.policy.reset()
        self.records.clear()

    def select_action(self, batch):
        require_empty_queue(self.policy)
        step = len(self.records) + 1
        metadata, arrays = {}, {}
        tensors = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        metadata["batch_other"] = {k: v for k, v in batch.items() if k not in tensors}
        hashes = {"step": step, "raw_pixels": array_hashes(self.env.latest_pixels),
                  "input": pack_tensors(tensors, "input", metadata, arrays)}
        hashes["rng_before"] = pack_rng(get_rng_state(), "rng_before", metadata, arrays)
        with capture_noise(self.policy) as noise:
            action = self.policy.select_action(batch)
        if len(noise) != 1 or not torch.isfinite(action).all():
            raise RuntimeError("Expected one sampled noise tensor and finite normalized action")
        require_empty_queue(self.policy)
        hashes["rng_after"] = pack_rng(get_rng_state(), "rng_after", metadata, arrays)
        hashes["noise"] = pack_tensors({"noise": noise[0]}, "noise", metadata, arrays)
        hashes["normalized_action"] = pack_tensors({"action": action}, "output", metadata, arrays)
        metadata["trace"] = hashes
        self.archive.writestr(f"step_{step:04d}.npz", encode_record(metadata, arrays))
        self.archive.fp.flush()
        self.trace_stream.write(json.dumps(hashes, allow_nan=False) + "\n")
        self.trace_stream.flush()
        self.records.append(hashes)
        return action
