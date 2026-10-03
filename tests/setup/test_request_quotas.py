"""Unit tests for the quota requests the demo needs."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from aws.config import DemoConfig
from aws.setup.request_quotas import MtrlQuotaManager


class QuotaTests(unittest.TestCase):
    """Verifies training, evaluation, and endpoint quotas are all covered."""

    def test_endpoint_quota_is_requested_with_the_job_quotas(self) -> None:
        client = MagicMock()
        client.get_service_quota.return_value = {"Quota": {"QuotaName": "q", "Value": 1.0}}
        with patch("aws.setup.request_quotas.boto3.client", return_value=client):
            report = MtrlQuotaManager(DemoConfig()).ensure_requested()

        self.assertEqual(set(report), {"fine_tuning", "evaluation", "endpoint"})
        self.assertEqual(report["endpoint"]["quota_code"], DemoConfig().endpoint_quota_code)
        self.assertTrue(all(item["request_status"] == "NOT_NEEDED" for item in report.values()))

    def test_current_reports_values_without_requesting(self) -> None:
        client = MagicMock()
        client.get_service_quota.return_value = {"Quota": {"QuotaName": "q", "Value": 0.0}}
        with patch("aws.setup.request_quotas.boto3.client", return_value=client):
            report = MtrlQuotaManager(DemoConfig()).current()

        self.assertEqual(len(report), 3)
        self.assertEqual(report["endpoint"]["current_value"], 0.0)
        client.request_service_quota_increase.assert_not_called()


if __name__ == "__main__":
    unittest.main()
