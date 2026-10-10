"""Project-local velocity forward helper and paired loss formulations for SmolVLA.

Reproduces the official SmolVLA flow-matching forward computation to expose the
predicted velocity field v_t, without modifying upstream packages.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_STATE,
    ACTION,
    make_att_2d_masks,
)
from smolvla_grounding.training_policy import FP32ExpertSmolVLAPolicy


def sample_paired_flow_inputs(model, action_shape, device, training_seed, step):
    """Use the official samplers in a per-update RNG stream isolated from both arms."""
    from .adaptation_records import paired_flow_seed

    seed = paired_flow_seed(training_seed, step)
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        noise = model.sample_noise(action_shape, device)
        time = model.sample_time(action_shape[0], device)
    return noise, time, seed

def forward_velocity(
    policy: SmolVLAPolicy,
    batch: Dict[str, Tensor],
    noise: Optional[Tensor] = None,
    time: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the predicted velocity field v_t and target velocity u_t for SmolVLA.

    Args:
        policy: SmolVLAPolicy instance.
        batch: Training batch dict containing images, state, tokens, actions.
        noise: Optional flow-matching noise tensor of shape (B, chunk_size, max_action_dim).
        time: Optional flow-matching time tensor of shape (B,).

    Returns:
        v_t: Predicted velocity field of shape (B, chunk_size, action_dim).
        u_t: Target velocity field (noise - actions) of shape (B, chunk_size, action_dim).
        noise: The noise tensor used.
        time: The time tensor used.
    """
    if policy.config.adapt_to_pi_aloha:
        batch[OBS_STATE] = policy._pi_aloha_decode_state(batch[OBS_STATE])
        batch[ACTION] = policy._pi_aloha_encode_actions_inv(batch[ACTION])

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
    actions = policy.prepare_action(batch)

    if noise is None:
        noise = policy.model.sample_noise(actions.shape, actions.device)
    if time is None:
        time = policy.model.sample_time(actions.shape[0], actions.device)

    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1.0 - time_expanded) * actions
    u_t = noise - actions

    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks = policy.model.embed_suffix(x_t, time)

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_out), _ = policy.model.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
    )
    suffix_out = suffix_out[:, -policy.config.chunk_size :]
    # Upcast attention output to float32 as in official openpi implementation
    suffix_out = suffix_out.to(dtype=torch.float32)
    v_t = policy.model.action_out_proj(suffix_out)

    # Slice to active action dimension (e.g. 7 for LIBERO)
    original_action_dim = policy.config.action_feature.shape[0]
    v_t = v_t[:, :, :original_action_dim]
    u_t = u_t[:, :, :original_action_dim]

    actions_is_pad = batch.get("action_is_pad")
    if actions_is_pad is not None:
        in_episode_bound = (~actions_is_pad).unsqueeze(-1)
        v_t = v_t * in_episode_bound
        u_t = u_t * in_episode_bound

    return v_t, u_t, noise, time


def compute_paired_bc_loss(
    v_orig: Tensor,
    v_para: Tensor,
    u_t: Tensor,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute compute-matched paired BC flow-matching loss.

    L_paired_bc = 0.5 * (MSE(v_orig, u_t) + MSE(v_para, u_t))
    """
    l_fm_orig = F.mse_loss(v_orig, u_t)
    l_fm_para = F.mse_loss(v_para, u_t)
    loss = 0.5 * (l_fm_orig + l_fm_para)
    metrics = {
        "loss": float(loss.detach().item()),
        "l_fm_orig": float(l_fm_orig.detach().item()),
        "l_fm_para": float(l_fm_para.detach().item()),
    }
    return loss, metrics


def compute_paired_flow_consistency_loss(
    v_orig: Tensor,
    v_para: Tensor,
    v_teacher: Tensor,
    u_t: Tensor,
    lambda_cons: float = 1.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute paired flow consistency loss with frozen canonical teacher.

    L_FM = 0.5 * (MSE(v_orig, u_t) + MSE(v_para, u_t))
    L_cons = 0.5 * (MSE(v_orig, stop_gradient(v_teacher)) + MSE(v_para, stop_gradient(v_teacher)))
    L_total = L_FM + lambda_cons * L_cons
    """
    v_teacher_sg = v_teacher.detach()

    l_fm_orig = F.mse_loss(v_orig, u_t)
    l_fm_para = F.mse_loss(v_para, u_t)
    l_fm = 0.5 * (l_fm_orig + l_fm_para)

    l_cons_orig = F.mse_loss(v_orig, v_teacher_sg)
    l_cons_para = F.mse_loss(v_para, v_teacher_sg)
    l_cons = 0.5 * (l_cons_orig + l_cons_para)

    loss = l_fm + lambda_cons * l_cons
    metrics = {
        "loss": float(loss.detach().item()),
        "l_fm": float(l_fm.detach().item()),
        "l_fm_orig": float(l_fm_orig.detach().item()),
        "l_fm_para": float(l_fm_para.detach().item()),
        "l_cons": float(l_cons.detach().item()),
        "l_cons_orig": float(l_cons_orig.detach().item()),
        "l_cons_para": float(l_cons_para.detach().item()),
        "lambda_cons": float(lambda_cons),
    }
    return loss, metrics


def create_frozen_teacher(policy_or_path, config=None, device="cuda") -> SmolVLAPolicy:
    """Create a strictly frozen teacher policy in eval mode."""
    if isinstance(policy_or_path, SmolVLAPolicy):
        teacher = policy_or_path
    else:
        teacher = FP32ExpertSmolVLAPolicy.from_pretrained(str(policy_or_path), config=config, strict=True)
    teacher = teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher
