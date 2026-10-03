"""Hard-capped budget guard built from measured usage and official AWS prices.

Projections use token counts AWS actually billed for this account's evaluation
rollouts (``BillableTokenUsage``), scaled to the planned rollouts with a safety
margin, plus the endpoint's maximum lifetime at the official hourly rate.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any

import boto3

from aws.config import MODEL_ID, DemoConfig

MILLION = Decimal(1_000_000)
_TOKEN_FIELDS = {
    "prefill": "PrefillTokenCount",
    "sample": "SampleTokenCount",
    "train": "TrainTokenCount",
}


class MtrlCostGuard:
    """Projects total demo spend and refuses plans above the hard cap."""

    TOKEN_SAFETY = Decimal("1.5")
    ENDPOINT_TAIL_MINUTES = 15
    """Allowance for watchdog poll lag and the time deletion itself takes."""
    CONTINGENCY_USD = Decimal("2")
    CONTINGENCY_LABEL = "Contingency (AgentCore, CodeBuild, S3, logs)"

    def __init__(
        self,
        config: DemoConfig,
        rates: Mapping[str, Decimal],
        hosting_hourly_usd: Decimal,
    ) -> None:
        """Initializes the guard with official prices.

        Args:
            config: Shared settings, including the cap and planned run sizes.
            rates: USD per million tokens, keyed like ``training_prefill``.
            hosting_hourly_usd: On-demand hourly price of the endpoint instance.
        """
        self.config = config
        self.rates = dict(rates)
        self.hosting_hourly_usd = hosting_hourly_usd

    @classmethod
    def from_price_list(cls, config: DemoConfig) -> MtrlCostGuard:
        """Builds a guard from the AWS Price List API.

        Args:
            config: Shared settings.

        Returns:
            Guard using current official MTRL and hosting prices.
        """
        pricing = boto3.client("pricing", region_name="us-east-1")
        reader = _PriceListReader(pricing, config.region)
        rates = {
            "training_prefill": reader.token_rate("Finetuning", "Prefill"),
            "training_sample": reader.token_rate("Finetuning", "Sample"),
            "training_train": reader.token_rate("Finetuning", "Train"),
            "evaluation_prefill": reader.token_rate("Evaluation", "Prefill"),
            "evaluation_sample": reader.token_rate("Evaluation", "Sample"),
        }
        hourly = reader.hosting_rate(config.endpoint_instance_type)
        return cls(config, rates, hourly)

    def billed_cost(self, component: str, usage: Mapping[str, int] | None) -> Decimal:
        """Prices token counts that AWS reported as billable.

        Args:
            component: ``Finetuning`` or ``Evaluation``.
            usage: ``BillableTokenUsage`` mapping from a job.

        Returns:
            Cost in USD.
        """
        prefix = "training" if component == "Finetuning" else "evaluation"
        total = Decimal(0)
        for kind, field in _TOKEN_FIELDS.items():
            rate = self.rates.get(f"{prefix}_{kind}")
            if rate is not None:
                total += Decimal((usage or {}).get(field, 0)) / MILLION * rate
        return total

    def planned_training_cost(self, measured: Mapping[str, int]) -> Decimal:
        """Projects training cost from measured per-rollout evaluation usage.

        Assumes the costlier reading of ``global_batch_size`` (prompts per step),
        and bounds trained tokens by each rollout's prefill plus sample tokens.

        Args:
            measured: Billed ``prefill`` and ``sample`` tokens over ``rollouts``.

        Returns:
            Projected cost in USD, including the safety margin.

        Raises:
            ValueError: If the measurement covers no rollouts.
        """
        prefill, sample = self._per_rollout(measured)
        rollouts = Decimal(
            self.config.training_steps
            * self.config.training_batch_size
            * self.config.training_group_size
        )
        per_rollout = (
            prefill * self.rates["training_prefill"]
            + sample * self.rates["training_sample"]
            + (prefill + sample) * self.rates["training_train"]
        )
        return rollouts * self.TOKEN_SAFETY * per_rollout / MILLION

    def planned_comparison_cost(self, measured: Mapping[str, int]) -> Decimal:
        """Projects the base-plus-fine-tuned evaluation pipeline cost.

        Args:
            measured: Billed ``prefill`` and ``sample`` tokens over ``rollouts``.

        Returns:
            Projected cost in USD, including the safety margin.
        """
        prefill, sample = self._per_rollout(measured)
        rollouts = Decimal(2 * Decimal(measured["rollouts"]))
        per_rollout = (
            prefill * self.rates["evaluation_prefill"]
            + sample * self.rates["evaluation_sample"]
        )
        return rollouts * self.TOKEN_SAFETY * per_rollout / MILLION

    def endpoint_cost(self, minutes: float) -> Decimal:
        """Prices endpoint uptime at the on-demand hourly rate.

        Args:
            minutes: Endpoint lifetime in minutes.

        Returns:
            Cost in USD.
        """
        return self.hosting_hourly_usd * Decimal(str(minutes)) / 60

    def evaluate(
        self,
        measured: Mapping[str, int],
        spent: Iterable[Mapping[str, Any]],
        remaining: Mapping[str, bool],
    ) -> dict[str, Any]:
        """Builds the budget report and enforces the hard cap.

        Args:
            measured: Billed usage of a reference evaluation and its rollout count.
            spent: Records with ``label``, ``component``, and billed ``usage``,
                or a precomputed ``usd`` amount.
            remaining: Which billable stages still need to run
                (``training``, ``comparison``, ``endpoint``).

        Returns:
            Report with one line per cost item and the projected total.

        Raises:
            PermissionError: If the projected total exceeds the cap.
        """
        lines = [
            {"item": record["label"], "usd": Decimal(str(record["usd"]))}
            if "usd" in record
            else {
                "item": record["label"],
                "usd": self.billed_cost(record["component"], record["usage"]),
            }
            for record in spent
        ]
        if remaining.get("training"):
            lines.append({"item": "Training (planned)", "usd": self.planned_training_cost(measured)})
        if remaining.get("comparison"):
            lines.append(
                {"item": "Comparison evaluation (planned)",
                 "usd": self.planned_comparison_cost(measured)}
            )
        if remaining.get("endpoint"):
            lines.append(
                {"item": "Endpoint (planned)",
                 "usd": self.endpoint_cost(
                     self.config.endpoint_max_minutes + self.ENDPOINT_TAIL_MINUTES
                 )}
            )
        lines.append({"item": self.CONTINGENCY_LABEL, "usd": self.CONTINGENCY_USD})
        total = sum((line["usd"] for line in lines), Decimal(0))
        cap = self.config.budget_limit_usd
        report = {
            "lines": [{"item": line["item"], "usd": round(float(line["usd"]), 4)} for line in lines],
            "projected_total_usd": round(float(total), 2),
            "budget_cap_usd": cap,
            "within_budget": total <= Decimal(cap),
        }
        if not report["within_budget"]:
            raise PermissionError(
                f"Projected spend ${total:.2f} exceeds the hard ${cap} cap."
            )
        return report

    @staticmethod
    def _per_rollout(measured: Mapping[str, int]) -> tuple[Decimal, Decimal]:
        """Returns measured prefill and sample tokens per rollout.

        Args:
            measured: Billed ``prefill`` and ``sample`` tokens over ``rollouts``.

        Returns:
            Prefill and sample tokens per rollout.

        Raises:
            ValueError: If the measurement covers no rollouts.
        """
        rollouts = Decimal(measured.get("rollouts", 0))
        if rollouts <= 0:
            raise ValueError("Measured usage must cover at least one rollout.")
        return Decimal(measured["prefill"]) / rollouts, Decimal(measured["sample"]) / rollouts


class _PriceListReader:
    """Reads single official prices from the AWS Price List API."""

    def __init__(self, pricing: Any, region: str) -> None:
        """Initializes the reader.

        Args:
            pricing: Boto3 Price List client (``us-east-1``).
            region: Region whose prices are read.
        """
        self.pricing = pricing
        self.region = region

    def token_rate(self, component: str, token_type: str) -> Decimal:
        """Returns USD per million tokens for one MTRL dimension.

        Args:
            component: ``Finetuning`` or ``Evaluation``.
            token_type: ``Prefill``, ``Sample``, or ``Train``.

        Returns:
            USD per million tokens.
        """
        return self._single_price(
            {"modelname": MODEL_ID, "tokentype": token_type,
             "component": f"MultiTurnRFT:{component}"},
            lambda attributes: True,
        )

    def hosting_rate(self, instance_type: str) -> Decimal:
        """Returns the on-demand real-time hosting price per hour.

        Args:
            instance_type: SageMaker instance type, such as ``ml.g6e.12xlarge``.

        Returns:
            USD per instance hour.
        """
        return self._single_price(
            {"instanceName": instance_type},
            lambda attributes: attributes.get("usagetype", "").endswith(
                f"-Host:{instance_type}"
            ),
        )

    def _single_price(self, fields: Mapping[str, str], accept: Any) -> Decimal:
        """Finds exactly one matching on-demand price.

        Args:
            fields: Price List attribute filters besides the region.
            accept: Predicate over product attributes for extra filtering.

        Returns:
            The unique price.

        Raises:
            RuntimeError: If zero or several distinct prices match.
        """
        filters = [{"Type": "TERM_MATCH", "Field": "regionCode", "Value": self.region}]
        filters += [{"Type": "TERM_MATCH", "Field": k, "Value": v} for k, v in fields.items()]
        response = self.pricing.get_products(
            ServiceCode="AmazonSageMaker", Filters=filters, MaxResults=100
        )
        prices: set[Decimal] = set()
        for raw in response["PriceList"]:
            product = json.loads(raw)
            if not accept(product["product"]["attributes"]):
                continue
            for offer in product["terms"].get("OnDemand", {}).values():
                for dimension in offer["priceDimensions"].values():
                    prices.add(Decimal(dimension["pricePerUnit"]["USD"]))
        if len(prices) != 1:
            raise RuntimeError(f"Expected one price for {dict(fields)}; found {prices}.")
        return prices.pop()
