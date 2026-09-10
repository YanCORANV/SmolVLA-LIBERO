# Changelog

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
