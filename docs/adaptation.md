# Controlled paraphrase adaptation: next implementation stage

The immediate question is: can meaning-preserving language augmentation improve
success under alternative phrasings while retaining official-instruction performance?
Establishing overfitting or a low-level cause for every failed rollout is not a
prerequisite. Meaning-changing instructions and their positive-goal checks are
deferred from this first adaptation comparison.

## Evidence already available

The multi-state run completed 18 episodes in 3 matched groups. Original and repeat
each achieved 1/3 successes; each paraphrase achieved 0/3. Repeated-original ever-success
disagreement was zero. These are fixed-280-step, one-task, one-policy-seed observations,
not a standard whole-LIBERO score. Execution took 4262.87 seconds, including 419.43
seconds through policy preparation. Source: `outputs/state_benchmark/runs/task0_states`.

## Official data audit: passed

Source: [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero),
revision `86958911c0f959db2bbbdb107eb3e17c5f9c798e`.
The [task table](https://huggingface.co/datasets/HuggingFaceVLA/libero/blob/86958911c0f959db2bbbdb107eb3e17c5f9c798e/meta/tasks.parquet)
contains an exact match to our original instruction:

> pick up the black bowl between the plate and the ramekin and place it on the plate

This is dataset task index **34**, corresponding by exact text to simulator
LIBERO-Spatial task **0**. Never equate the two numeric indices. Metadata identifies
45 demonstrations and 4487 frames, spread across 36 corrected shared Parquet shards
totaling 3,483,030,726 bytes. Images are embedded in Parquet. Metadata SHA-256 values and Hub
LFS shard hashes are pinned in `configs/task0_dataset.json`.

The first data download exposed stale upstream file pointers. The previous 12-file,
1.06 GB selection was incorrect: file 055 has the expected Hub hash but contains
episodes 137–139, not the requested 1272, 1273, 1275 and 1282. Frame validation
correctly rejected this mismatch; its original error message did not explain it.

The repair uses the episode-location metadata from official
[lerobot/libero at a1aaacb](https://huggingface.co/datasets/lerobot/libero/blob/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4/meta/episodes/chunk-000/file-000.parquet).
All 45 episode IDs, lengths and global frame bounds match the original metadata.
Only file locations are taken from this reference: downloads remain PNG-in-Parquet
from the original `HuggingFaceVLA/libero` revision. A bounded HTTP range read of
corrected file 309 returned episode 1272 with all 84 expected numeric frames,
task index 34 and global indices 221834–221917; numeric checks passed.
The other corrected shards must still pass full download-time validation.
See [repair evidence](dataset_mapping_repair.json).

This proves the source of demonstrations for our future tuning, not the complete
training history of the pretrained checkpoint. Dataset metadata declares 10 FPS,
whereas our fixed evaluation uses 20 Hz control. Loader temporal offsets must use
the declared dataset FPS; action semantics and conversion provenance need inspection
before training. Do not silently halve actions, alter evaluation control frequency,
or call the difference an established cause of failure.

Reproduce the small metadata audit without Torch/model imports:

```bash
# Run from the repository root.
source .venv/bin/activate
python scripts/inspect_task0_dataset.py
```

## Download and split: completed after the file-location repair

The user-run preparation completed all 45 demonstrations with 36 training and 9
held-out episodes, in 1449.49 seconds. `outputs/task0_data/ready.json` records all
4487 checked frames, image sample checks and prepared metadata provenance.
The [real optimizer-step profile](training_profile.md) and both 200-update training
arms have since passed. [Post-training simulation](adaptation_evaluation.md) is next.

```bash
# Run from the repository root.
source .venv/bin/activate
python scripts/prepare_task0_data.py
```

The program asks for the number of training demonstrations (default 36 of 45).
The remaining 9 are held out from our tuning. Split membership is reproducible
using SHA-256 ranking with seed 20260909 and is saved before downloading. Shared
shards may contain other tasks; those rows are excluded from the selected episode
IDs. Upstream metadata bytes are preserved in `_original_metadata`. After all
selected frame checks pass, a local metadata copy changes only the selected
episodes' data-file pointers so the official loader can find their actual data.
The prepared metadata hash and exact corrected IDs are recorded in `ready.json`.
No image, action, state, instruction or episode ID is rewritten.

Downloads use official `hf_hub_download` with a fixed revision. Each shard must pass
its pinned byte count and SHA-256, all selected numeric/frame/task/timestamp checks,
and first/middle/last image decoding for both cameras before it is marked complete.
Validation reads 64 rows per Arrow batch to bound host memory. Rerun the same command
and answer `y` to continue; completed shards are rehashed, completed validation
reports are reused, and Hub partial downloads can resume. Output is in
`outputs/task0_data`, data in the ignored revision-specific `data/` directory.

Existing 36/9 split selections from the failed first run are accepted only after
their original dataset/config identity and exact split are verified. On interactive
continuation, the old selection identity is archived and shard checks are rebuilt
under the repaired code/config hashes. The unrelated downloaded file 055 is retained
but excluded from the selected file list. Thirty-seven CPU regression checks pass,
including split-preserving recovery and a metadata repair that preserves other columns.

The resulting `ready.json` proves data preparation, not successful use of the
official dataloader or a training step. No training dependencies are installed
automatically, and no model/GPU computation runs in these preparation scripts.

## Next gates after data preparation

1. Load an actual selected demonstration through installed official LeRobot APIs.
   Verify image orientation, state/action preprocessing, action chunk padding and
   temporal interpretation. Reuse the pretrained normalization for both tuning arms.
2. Run a real loss/backward/optimizer/checkpoint-reload sequence with batch size 1,
   one process and frozen VLM/action-expert tuning. Measure VRAM and time before
   selecting the training budget. Mixed precision must be validated against the
   actual trainer; gradient accumulation is not yet implemented or assumed.
3. Train original-language and augmented-language models from the same checkpoint,
   on identical demonstration splits, equal update budgets and matched sampling.
   Only change task text for augmentation; never attach a changed goal to an
   unchanged demonstration. Keep new training paraphrases separate from evaluation.
4. Compare pretrained, original-language-tuned and augmented-language-tuned models
   with the same evaluation cases and checkpoint-compatible processors. Start with
   an interactively chosen small case set, then expand after pipeline acceptance.
   Current paraphrases and states have already informed development: they can be
   development outcomes, but must not be advertised as an untouched final test set.
   New final-test phrasings are reserved before model selection.

Demonstration holdout does not establish unseen simulator states or absence from
upstream pretraining. All such claims require separate evidence. Positive gains are
an experimental outcome, not a release acceptance requirement.

## Contribution boundaries

Upstream supplies the policy, demonstrations, simulation, processing, training
machinery and Hub downloads. This project supplies exact instruction matching,
fixed episode splits, language augmentation, paired evaluations, experiment
provenance and controlled before/after comparisons. Both training arms completed;
robustness gains remain unmeasured and must not be described as established.
