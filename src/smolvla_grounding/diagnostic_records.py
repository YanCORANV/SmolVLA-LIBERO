"""Lossless compressed tensor/RNG records and conservative divergence summaries."""

import hashlib
import io
import json

import numpy as np


def encode_record(metadata, arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata, allow_nan=False)), **arrays)
    return stream.getvalue()


def decode_record(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        arrays = {k: saved[k].copy() for k in saved.files if k != "metadata"}
    return metadata, arrays


def pack_tensors(tensors, prefix, metadata, arrays):
    """Store raw tensor bytes so BF16 and integer/mask inputs retain exact values."""
    import torch
    descriptions, hashes = {}, {}
    for name, tensor in sorted(tensors.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected tensor: {name}")
        key = prefix + "/" + name
        description = {"dtype": str(tensor.dtype).removeprefix("torch."),
                       "shape": list(tensor.shape), "device": str(tensor.device)}
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().copy()
        arrays[key] = raw
        descriptions[name] = description
        header = json.dumps(description, sort_keys=True).encode()
        hashes[name] = hashlib.sha256(header + b"\0" + raw.tobytes()).hexdigest()
    metadata[prefix] = descriptions
    return hashes


def unpack_tensors(prefix, metadata, arrays):
    import torch
    return {name: torch.from_numpy(arrays[prefix + "/" + name].copy())
            .view(getattr(torch, spec["dtype"])).reshape(spec["shape"]).to(spec["device"])
            for name, spec in metadata[prefix].items()}


def pack_rng(state, prefix, metadata, arrays):
    """Serialize the exact values returned by upstream get_rng_state, including Gaussian caches."""
    keys = {k: v for k, v in state.items() if k.startswith("torch")}
    hashes = pack_tensors(keys, prefix, metadata, arrays)
    numpy_state = state["numpy_random_state"]
    arrays[prefix + "/numpy_keys"] = numpy_state[1].copy()
    other = {"python": state["random_state"], "numpy": [numpy_state[0], numpy_state[2],
                                                        numpy_state[3], numpy_state[4]]}
    metadata[prefix + "_other"] = other
    hashes["python_numpy"] = hashlib.sha256(json.dumps(other).encode() + numpy_state[1].tobytes()).hexdigest()
    return hashes


def unpack_rng(prefix, metadata, arrays):
    state = unpack_tensors(prefix, metadata, arrays)
    other = metadata[prefix + "_other"]
    py = other["python"]
    state["random_state"] = (py[0], tuple(py[1]), py[2])
    ns = other["numpy"]
    state["numpy_random_state"] = (ns[0], arrays[prefix + "/numpy_keys"].copy(), ns[1], ns[2], ns[3])
    return state


def compare_traces(first, second):
    if len(first) != len(second) or not first:
        raise ValueError("Expected two nonempty equal-length traces")
    fields = ("raw_pixels", "input", "rng_before", "noise", "normalized_action", "rng_after")
    for index, (a, b) in enumerate(zip(first, second, strict=True), 1):
        if a["step"] != index or b["step"] != index:
            raise ValueError("Trace steps must be contiguous and one-based")
    earliest = {field: next((a["step"] for a, b in zip(first, second, strict=True)
                             if a[field] != b[field]), None) for field in fields}
    step = earliest["normalized_action"]
    evidence = None
    if step is not None:
        a, b = first[step - 1], second[step - 1]
        evidence = {field + "_equal": a[field] == b[field] for field in fields}
    return {"first_different_step": earliest, "evidence_at_first_action_difference": evidence,
            "interpretation": "Hash comparisons localize observed differences; replay evidence is needed before assigning a cause."}
