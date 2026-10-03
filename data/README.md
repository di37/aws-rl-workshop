# data/

The two prompt files that define the study. Each row is one customer's opening message, in a single `prompt` column. There are no answers or labels: the support simulator in `agent/environment.py` scores every move.

| File | Rows | SHA-256 (start) | Used for |
|---|---|---|---|
| `training_prompts.csv` | 64 unique tickets | `1c61077a792df2c7…` | Training: 10 steps of 32 tickets, so every ticket comes up 5 times, 4 tries each |
| `evaluation_prompts.csv` | 32 unique held-out tickets | `213538946018bdc3…` | The baseline, the base-vs-fine-tuned comparison (2 tries per ticket) and live inference (the first 6) |

- **No overlap.** No held-out ticket appears in training. Every ticket has the same underlying fault and fix; only the customer's wording changes.
- **On S3.** `scripts/04_upload_datasets.py` uploads both files as Parquet, and the jobs read those copies. `artifacts/dataset_provenance.json` records that they match these files prompt by prompt.
- **Why 64 training tickets?** Training needs more prompts than its batch size of 32. The original 32 are kept in `artifacts/training_prompts.32.csv`.
- **Fixed on purpose.** The full SHA-256 values are in the reproducibility sheet. Changing a file changes its fingerprint, and the invariants then flag the run as a different protocol.
