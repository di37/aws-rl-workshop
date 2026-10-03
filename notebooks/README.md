# notebooks/

What you show an audience. Open from the project root with `python -m jupyter lab notebooks/`; the notebook finds the project files itself.

| Notebook | Use it to |
|---|---|
| `sagemaker_mtrl_real_demo.executed.ipynb` | **Present.** The replay of the run of record with every output, sections 1–14 in the order of the scripts. |
| `sagemaker_mtrl_real_demo.ipynb` | **Run.** The same notebook without outputs. |
| `sagemaker_mtrl_real_demo.training-run.ipynb` | **Read only.** The log of the live training run, kept as recorded; its code predates the current folder layout. |

The logs of the live inference run and of the first endpoint attempt are in `artifacts/`.

## Running it

- **Replay by default.** The notebook re-attaches to the recorded jobs and submits nothing. It needs AWS credentials for the account the jobs ran in, or your own account after `scripts/01`–`04`.
- **Billable steps are opt-in.** Each one runs only when its environment variable holds its exact confirmation phrase: `MTRL_TRAIN_CONFIRMATION`, `MTRL_COMPARISON_CONFIRMATION`, `MTRL_DEPLOY_CONFIRMATION` or `MTRL_BEDROCK_IMPORT_CONFIRMATION`. Live inference runs only with `MTRL_LIVE_INFERENCE=1`.
- **Keep the presentation copy.** Execute into a new file rather than over the `executed` copy.

## Maintaining it

The notebook is generated from code: `python scripts/dev/build_notebook.py` rebuilds it, and `--check` confirms that the committed notebook matches the builder. After executing a notebook, remove sign-in links and check it before sharing:

```bash
python scripts/dev/scrub_notebook.py notebooks/sagemaker_mtrl_real_demo.executed.ipynb
python scripts/dev/check_notebook.py notebooks/sagemaker_mtrl_real_demo.executed.ipynb
```
