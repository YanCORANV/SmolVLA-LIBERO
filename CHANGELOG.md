# Changelog

## 2026-10-10 — audited five-task development pilot

- Preserved all 75 already completed untuned SmolVLA rollouts; no new evaluation or training was run for this audit.
- Published 75 per-rollout outcomes with goal_ever and goal_final, 50 matched outcome/action/EEF comparisons, an integrity audit, and four reproducible figures.
- Added 12 deterministic video-backed comparison cases across the five tasks (10 failure pairs, two successful contrasts) with source-video and case-sheet SHA256 values.
- Added a model-free public-data verifier, bounded tests for missing records and EEF values, and separate reproduction instructions. The five-task pilot remains distinct from the Task-0 paired-training final result.
## 2026-10-09 — completed language-robustness study and public documentation

- First confirmatory, anchored and paired development, action/trajectory diagnosis, and amended final v4 evaluation are complete.
- Final v4: 153/153 Task-0 rollouts on States 33–49. `paired_bc` improved observed canonical success but did not improve unseen-paraphrase success; the development result did not generalize.
- Added a current README, experiment/method/results documents, curated result copies and reproducible figures. Historical protocol records and failed partial attempts remain preserved.


## v0.1.0 — prepared for initial publication

- Official SmolVLA/LIBERO evaluation running on an RTX 4070 Laptop GPU with 8 GB VRAM.
- Controlled task-0 instruction variants, matched initial-state checks, repeat
  diagnostics and CSV/JSON episode records.
- Validated selection of 45 official demonstrations, with 36 training and 9 held out.
- Matched original-language and language-augmented fine-tuning at two learning rates
  and two training budgets, including fresh-process checkpoint reload checks.
- Three development comparisons, curated rollout keyframes and failure reviews.
- Setup, experiment, contribution and publication documentation, with an MIT license
  for project-specific code.

Scope: one task, three development states. The latest augmented model achieves
paraphrase C in 2/3 cases and B in 0/3. This is an early research prototype, not a
full LIBERO benchmark or a demonstrated general robustness improvement.

The GitHub release/tag is created separately after the initial repository upload.
