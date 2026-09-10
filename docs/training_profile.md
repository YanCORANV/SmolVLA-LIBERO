# Real optimizer-step acceptance on the 8 GB GPU

Status: real GPU update and fresh-process reload passed. The guarded update took
3.69 seconds with 3.08 GiB peak PyTorch reserved memory. All reloaded parameter bytes
matched and optimizer step 1 restored. Data preparation passed for 45 demonstrations
with the original 36/9 split. No full fine-tuning experiment has run. Continue with
the [paired fine-tuning comparison](paired_finetuning.md).

Run in WSL after `outputs/task0_data/ready.json` passes:

```bash
# Run from the repository root.
bash scripts/profile_training.sh
```

This performs exactly one real batch-1 optimizer update, then exits that process
before loading its checkpoint in a fresh process. It is an acceptance measurement,
not a success-rate experiment or a final training budget. Each invocation creates a
new timestamped directory under `outputs/training_profile`; interrupted profiling
attempts remain there and rerunning restarts the single update. It does not resume
a partial optimizer step. Long training will use upstream optimizer/RNG checkpoints
with interactive resume choices after this acceptance passes.

## What it verifies

1. Prepared shard hashes, corrected episode metadata, dataset revision and exact
   training/held-out IDs match the accepted data selection.
2. Official `LeRobotDataset` selects only the training episodes. First and last
   frame action chunks match source actions and the upstream episode-padding mask.
   Dataset timestamps remain at the declared 10 FPS. No action rescaling or change
   to the separate 20 Hz simulator evaluation is introduced.
3. Official checkpoint processors supply normalization and tokenization; actions
   round-trip through checkpoint normalization. Official CHW image values are kept
   unchanged and two preview PNGs are saved for inspection.
4. Official `lerobot_train.update_policy`, the policy optimizer preset and
   Accelerate perform loss/backward/AdamW. A pre-step hook rejects nonfinite gradients.
   Every frozen parameter must retain its byte hash and at least one trainable
   parameter must change. Loss and gradient norm must be finite.
5. Official policy/processor and optimizer/RNG save APIs write a checkpoint. A fresh
   process reloads it with strict keys, checks every parameter value/dtype/freeze
   flag, restores optimizer step 1, and predicts one finite action on the recorded
   real observation. This is not a simulator rollout or held-out evaluation.

## Memory and numerical settings

`configs/training_profile.json` fixes one process, batch 1, frozen VLM, action expert
and projection updates, BF16 autocast, and FP32 trainable parameter/AdamW state storage.
Frozen weights retain their original storage precision. No LoRA, gradient accumulation,
compilation, gradient checkpointing or image-resolution changes are introduced.

`FP32ExpertSmolVLAPolicy` only changes trainable parameter storage during construction,
before the upstream loader copies weights. It inherits the entire official forward,
loss and action sampler. This ordering matters: loading updated FP32 expert weights
into an ordinary BF16-initialized expert can round away updates. Therefore both
tuning arms and subsequent evaluation must use the documented loading adapter.
The frozen baseline's numerical configuration must be controlled explicitly in
future comparisons; profiling is not itself a language-augmentation result.

Autocast is explicitly supplied to Accelerate for this profile. The checkpoint's
evaluation `use_amp` setting remains unchanged. The profile uses the configured
learning rate without a scheduler for this one update; the training schedule is
chosen later. Do not infer training VRAM from the earlier 1.24 GiB inference peak.

Per-step timing includes finite-gradient checks and first-use CUDA overhead. It is
not a steady-state throughput estimate. No total training-time promise is made from
one update. Saved artifacts include step/reload reports, parameter hashes, real
sample arrays and previews, policy/processors, optimizer/RNG state, console logs,
sampled device memory and per-process peak RSS. Large artifacts remain Git-ignored.

The wrapper enables Hub/Datasets offline access to consume the prepared local files;
it does not install packages or silently fetch a replacement dataset/model. Cold
imports and first Arrow cache construction on `/mnt/d` can still take minutes.
