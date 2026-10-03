# tests/

Unit tests for the `aws/` library, in folders that mirror it: `setup/`, `rl/`, `deploy/`, `costs/`, `records/` and `reporting/`, plus `test_workflow.py` and `test_guards.py` for the facade and the confirmation check.

```bash
python -m pytest                       # 302 tests: tests/ (254) and agent/ (48)
python -m pytest tests/rl              # one package
ruff check aws agent scripts tests
```

Run them from the project root; `pyproject.toml` sets the paths. They need no AWS access or credentials: `conftest.py` sets a default region so the SageMaker SDK imports, and wherever code would call AWS, the tests use fakes.

**What they cover**

- **Billable steps refuse to run** without their exact phrase, and a recorded job is never submitted twice.
- **The budget guard** projects spend correctly and blocks anything over the $25 cap.
- **Evidence and notebooks** are redacted and scrubbed of signed links.
- **Report tables, figures, the reproducibility record and sheet** build correctly from evidence.
- **The invariants** pass on good evidence and fail on broken evidence.

The agent's own tests live next to it in `agent/`.
