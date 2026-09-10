"""Official architecture/loader with FP32 storage for trainable parameters."""

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class FP32ExpertSmolVLAPolicy(SmolVLAPolicy):
    """Cast before upstream weight loading so saved FP32 updates survive reload.

    No forward, loss, noise sampler or optimizer implementation is replaced.
    Keep this construction policy for every reload of this training checkpoint.
    """

    def __init__(self, config, **kwargs):
        if not config.train_expert_only or not config.freeze_vision_encoder:
            raise ValueError("This adapter is for frozen-VLM expert tuning only")
        super().__init__(config, **kwargs)
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(dtype=torch.float32)


def parameter_evidence(policy):
    import hashlib
    from collections import Counter

    hashes, counts = {}, Counter()
    for name, parameter in policy.named_parameters():
        # Transfer one parameter at a time, rather than cloning the model in VRAM.
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        hashes[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "dtype": str(parameter.dtype),
                        "shape": list(parameter.shape), "trainable": parameter.requires_grad}
        counts[("trainable" if parameter.requires_grad else "frozen") + ":" + str(parameter.dtype)] += parameter.numel()
    return hashes, dict(counts)
