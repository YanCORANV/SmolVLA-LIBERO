# Contributing

Start with the [project overview](README.md), [reproduction guide](docs/reproduction.md)
and [current results](results/README.md). Small, focused changes are welcome.

## Local checks

The existing record-validation and protocol tests run without a GPU or model downloads:

```bash
python -m unittest discover -s tests -v
```

Use the project environment, or a separate Python 3.12 environment with NumPy,
Pillow and PyArrow installed. Two process-lifecycle tests require POSIX and are
skipped on Windows. These tests do not validate CUDA, rendering or model performance.
GPU integration is checked separately using the staged setup and rollout commands.

## Experiment changes

- Keep source, configuration, data revisions and seeds with each experiment.
- Existing runs hash their inputs and source files. Do not overwrite completed
  results or relax provenance checks to force a resume after changing the protocol.
- Give new experiments separate output directories. Use the existing interactive
  prompts for run budgets and resumption.
- Report failed episodes, repeat disagreements and reused baselines as well as gains.
- Preserve the distinction between development results and untouched evaluation.

Commit source, configuration, documentation and small curated results. Keep credentials,
environments, datasets, weights and raw run artifacts out of Git. See `.gitignore`.
For a pull request, describe the behavioral change, the evidence/checks performed,
and any effect on comparability with previous runs. Contributions to project-specific
code use this repository's MIT license; upstream material retains its own terms.
