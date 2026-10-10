# Changelog

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
