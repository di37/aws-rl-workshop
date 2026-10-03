"""Test setup: a default AWS region so the SageMaker SDK can be imported.

The SDK creates a client when it is imported, which fails on a machine without
an AWS region configured. Tests make no AWS calls. Import paths (the project
root and ``agent/``) come from ``pyproject.toml``.
"""

import os

os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
