"""Library for the SageMaker multi-turn RL workshop; start at ``workflow.py``.

- ``config.py``: settings, project paths, and provisioned resource state
- ``guards.py``: exact confirmation phrases for billable steps
- ``workflow.py``: one facade that wires a session, the evidence store, and every stage
- ``setup/``: account setup and teardown
- ``rl/``: multi-turn RL training and evaluation on SageMaker
- ``deploy/``: deployment (endpoint or Bedrock import) and live inference
- ``costs/``: budget guard, spend ledger, and cost snapshot
- ``records/``: evidence files and provenance captured from AWS
- ``reporting/``: report tables and figures, reproducibility record, and invariants
"""
