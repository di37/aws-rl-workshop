"""Request the minimum SageMaker MTRL concurrency quotas."""

from __future__ import annotations

from typing import Any

import boto3
from botocore.exceptions import ClientError

from aws.config import DemoConfig


class MtrlQuotaManager:
    """Ensures the account can run one MTRL job of each type and one endpoint instance."""

    QUOTAS = {
        "fine_tuning": "L-A9CEE0B1",
        "evaluation": "L-2633C043",
    }

    def __init__(self, config: DemoConfig) -> None:
        """Initializes the Service Quotas client.

        Args:
            config: Shared region configuration.
        """
        self.client = boto3.client("service-quotas", region_name=config.region)
        self.quotas = {**self.QUOTAS, "endpoint": config.endpoint_quota_code}

    def current(self) -> dict[str, Any]:
        """Reports the applied value of every required quota, requesting nothing.

        Returns:
            Quota code, name, and current value per quota.
        """
        report: dict[str, Any] = {}
        for name, quota_code in self.quotas.items():
            quota = self.client.get_service_quota(
                ServiceCode="sagemaker",
                QuotaCode=quota_code,
            )["Quota"]
            report[name] = {
                "quota_code": quota_code,
                "quota_name": quota["QuotaName"],
                "current_value": quota["Value"],
            }
        return report

    def ensure_requested(self) -> dict[str, Any]:
        """Requests a value of one for each zero-valued quota.

        Returns:
            Current quota values and request statuses.
        """
        return {
            name: {**item, "request_status": "NOT_NEEDED"}
            if item["current_value"] >= 1
            else {**item, **self._request(item["quota_code"])}
            for name, item in self.current().items()
        }

    def _request(self, quota_code: str) -> dict[str, Any]:
        """Creates or reuses a quota-increase request.

        Args:
            quota_code: SageMaker Service Quotas identifier.

        Returns:
            Request identifier and status.
        """
        try:
            request = self.client.request_service_quota_increase(
                ServiceCode="sagemaker",
                QuotaCode=quota_code,
                DesiredValue=1,
            )["RequestedQuota"]
        except ClientError as error:
            if error.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                raise
            request = self._find_existing_request(quota_code)
        return {
            "request_id": request["Id"],
            "request_status": request["Status"],
            "desired_value": request["DesiredValue"],
        }

    def _find_existing_request(self, quota_code: str) -> dict[str, Any]:
        """Finds the newest existing request for a quota.

        Args:
            quota_code: SageMaker Service Quotas identifier.

        Returns:
            Existing quota request.

        Raises:
            RuntimeError: If AWS reports a duplicate but returns no request.
        """
        requests = self.client.list_requested_service_quota_change_history_by_quota(
            ServiceCode="sagemaker",
            QuotaCode=quota_code,
        ).get("RequestedQuotas", [])
        if not requests:
            raise RuntimeError(f"No existing request was found for quota {quota_code}.")
        return max(requests, key=lambda item: item["Created"])
