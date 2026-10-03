"""Freezes what the study cost into an offline evidence file.

Billed token counts (``BillableTokenUsage``) and the official rates come from
AWS. The snapshot keeps both next to each priced line, so every amount can be
re-checked by hand, and the report scripts run without AWS access.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from aws.costs.cost_guard import MtrlCostGuard

RECORD = "cost_accounting.json"
SOURCE_TOKENS = "billed tokens (BillableTokenUsage) x AWS Price List rate"
SOURCE_RECORDED = "computed or estimated; see the label (extra_costs.json, endpoint.json)"
SOURCE_ALLOWANCE = "allowance for unmeasured items"


def build_cost_snapshot(
    records: Iterable[Mapping[str, Any]], guard: MtrlCostGuard, captured_at: str
) -> dict[str, Any]:
    """Prices every recorded billable stage and adds the unmeasured allowance.

    Args:
        records: Spend records from :meth:`SpendLedger.spent`.
        guard: Cost guard holding the official rates.
        captured_at: ISO timestamp of when usage and rates were read.

    Returns:
        Rates, one item per stage (with its token usage when priced from
        tokens), and totals that keep billed-token amounts apart from computed
        or estimated ones and from the allowance.
    """
    items = [_priced_item(record, guard) for record in records]
    tokens = round(sum(item["usd"] for item in items if item["source"] == SOURCE_TOKENS), 4)
    other = round(sum(item["usd"] for item in items if item["source"] != SOURCE_TOKENS), 4)
    allowance = float(guard.CONTINGENCY_USD)
    items.append({"item": guard.CONTINGENCY_LABEL, "usd": allowance, "source": SOURCE_ALLOWANCE})
    return {
        "captured_at": captured_at,
        "rates_usd_per_million_tokens": {name: str(rate) for name, rate in sorted(guard.rates.items())},
        "hosting_usd_per_hour": str(guard.hosting_hourly_usd),
        "items": items,
        "billed_tokens_usd": round(tokens, 2),
        "computed_or_estimated_usd": round(other, 2),
        "total_usd": round(tokens + other, 2),
        "total_with_allowance_usd": round(tokens + other + allowance, 2),
        "budget_cap_usd": guard.config.budget_limit_usd,
    }


def _priced_item(record: Mapping[str, Any], guard: MtrlCostGuard) -> dict[str, Any]:
    """Prices one spend record, keeping its inputs.

    Args:
        record: A ``usd`` amount, or a ``component`` with billed ``usage``.
        guard: Cost guard holding the official rates.

    Returns:
        Item label, USD amount, source, and token usage when priced from tokens.
    """
    if "usd" in record:
        return {"item": record["label"], "usd": round(float(record["usd"]), 4), "source": SOURCE_RECORDED}
    usage = {field: int(count) for field, count in (record.get("usage") or {}).items()}
    usd = guard.billed_cost(record["component"], usage)
    return {
        "item": record["label"],
        "component": record["component"],
        "usage": usage,
        "usd": round(float(usd), 4),
        "source": SOURCE_TOKENS,
    }
