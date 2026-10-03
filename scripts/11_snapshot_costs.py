"""Freeze what the study cost into artifacts/cost_accounting.json (read-only).

Reads the token counts AWS billed for every recorded job (BillableTokenUsage),
the official rates from the AWS Price List API, and the recorded amounts
(endpoint uptime, cross-region copy, imported-model use), then saves each
priced line with its inputs so it can be re-checked by hand. Steps 12-14 use
this snapshot offline. AWS Cost Explorer remains the final bill.
"""

# region Imports & setup
from __future__ import annotations

from datetime import datetime, timezone

import _bootstrap as bootstrap

from aws.costs.cost_guard import MtrlCostGuard
from aws.costs.cost_snapshot import RECORD, build_cost_snapshot
from aws.reporting.report_tables import cost_table

# endregion


# region Entry point
def main() -> None:
    """Prices every recorded stage and saves the snapshot."""
    bootstrap.parse_args(__doc__.splitlines()[0])
    bootstrap.banner("Cost snapshot (11)", "BillableTokenUsage x AWS Price List, plus recorded amounts")
    workflow = bootstrap.load_workflow()
    guard = MtrlCostGuard.from_price_list(bootstrap.CONFIG)
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    snapshot = build_cost_snapshot(workflow.ledger().spent(), guard, captured_at)
    workflow.store.save(RECORD, snapshot)
    bootstrap.show(cost_table(snapshot))
    print(f"\nBilled tokens ${snapshot['billed_tokens_usd']:.2f} + computed or estimated "
          f"${snapshot['computed_or_estimated_usd']:.2f} = ${snapshot['total_usd']:.2f}; with the allowance for "
          f"unmeasured items ${snapshot['total_with_allowance_usd']:.2f} (hard cap ${snapshot['budget_cap_usd']}).")
    print("Not measured automatically: the cross-region weight copy, imported-model minutes, Bedrock "
          "storage, AgentCore, CodeBuild, S3, and logs. Add amounts from AWS Cost Explorer to "
          'artifacts/extra_costs.json as {"items": [{"label": ..., "usd": ...}]}, then rerun this step.')


if __name__ == "__main__":
    bootstrap.run(main)
# endregion
