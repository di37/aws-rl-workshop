"""Report the SageMaker quotas the demo needs; with --submit, request missing ones.

Free. Quotas: one MTRL fine-tuning job, one MTRL evaluation job, and one
ml.g6e.12xlarge endpoint instance (deployment option 1 only). Approval can
take hours, so run this early.
"""

# region Imports & setup
from __future__ import annotations

import _bootstrap as bootstrap

from aws.setup.request_quotas import MtrlQuotaManager

# endregion


# region Entry point
def main() -> None:
    """Prints quota values, requesting a value of one where the quota is zero."""
    args = bootstrap.parse_args(__doc__.splitlines()[0], extra=lambda parser: parser.add_argument(
        "--submit", action="store_true", help="request every quota that is still zero"))
    bootstrap.banner("Quotas (02)", "MTRL training, MTRL evaluation, endpoint instance")
    manager = MtrlQuotaManager(bootstrap.CONFIG)
    report = manager.ensure_requested() if args.submit else manager.current()
    bootstrap.show([{"quota": name, **item} for name, item in report.items()])


if __name__ == "__main__":
    bootstrap.run(main)
# endregion
