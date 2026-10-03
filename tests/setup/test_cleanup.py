"""Unit tests for tearing down the Bedrock import resources."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from aws.config import DemoConfig
from aws.setup.cleanup import InfrastructureCleaner

BUCKET = "sagemaker-mtrl-demo-123456789012-us-east-1"
TAGS = [{"Key": "Project", "Value": "sagemaker-mtrl-demo"}]


class BedrockCleanupTests(unittest.TestCase):
    """Verifies scope, ownership checks, tag checks, and error collection."""

    def setUp(self) -> None:
        self.cleaner = InfrastructureCleaner.__new__(InfrastructureCleaner)
        self.cleaner.config = DemoConfig()
        self.cleaner.state = SimpleNamespace(account_id="123456789012")
        self.cleaner.bedrock_east = MagicMock()
        self.cleaner.bedrock_east.list_model_import_jobs.return_value = {"modelImportJobSummaries": []}
        self.cleaner.bedrock_east.list_imported_models.return_value = {"modelSummaries": [
            {"modelName": "mtrl-support-gpt-oss-20b-ft"}, {"modelName": "mtrl-support-gpt-oss-20b-ft-v2"}]}
        self.cleaner.s3_east = MagicMock()
        self.cleaner.s3_east.get_paginator.return_value.paginate.return_value = [
            {"Versions": [{"Key": "imported-models/x/config.json", "VersionId": "null"}]}
        ]
        self.cleaner.s3_east.delete_objects.return_value = {}
        self.cleaner.iam = MagicMock()
        self.cleaner.iam.list_role_tags.return_value = {"Tags": TAGS}
        self.cleaner._delete_role = MagicMock()

    def test_demo_models_bucket_and_tagged_role_are_deleted(self) -> None:
        errors = self.cleaner._delete_bedrock_import()

        self.assertEqual(errors, [])
        deleted = [c.kwargs["modelIdentifier"] for c in self.cleaner.bedrock_east.delete_imported_model.call_args_list]
        self.assertEqual(deleted, ["mtrl-support-gpt-oss-20b-ft", "mtrl-support-gpt-oss-20b-ft-v2"])
        objects_call = self.cleaner.s3_east.delete_objects.call_args.kwargs
        self.assertEqual((objects_call["Bucket"], objects_call["ExpectedBucketOwner"]), (BUCKET, "123456789012"))
        self.cleaner.s3_east.delete_bucket.assert_called_once_with(
            Bucket=BUCKET, ExpectedBucketOwner="123456789012"
        )
        self.cleaner._delete_role.assert_called_once_with("BedrockMTRLModelImportRole")

    def test_untagged_role_is_kept(self) -> None:
        self.cleaner.iam.list_role_tags.return_value = {"Tags": []}

        self.cleaner._delete_bedrock_import()

        self.cleaner._delete_role.assert_not_called()

    def test_running_import_job_blocks_deleting_its_files(self) -> None:
        self.cleaner.bedrock_east.list_model_import_jobs.return_value = {"modelImportJobSummaries": [
            {"jobName": "job-1", "importedModelName": "mtrl-support-gpt-oss-20b-ft"}]}

        errors = self.cleaner._delete_bedrock_import()

        self.assertTrue(any("still running" in e for e in errors))
        self.cleaner.s3_east.delete_bucket.assert_not_called()
        self.cleaner.bedrock_east.delete_imported_model.assert_not_called()

    def test_errors_are_collected_and_later_steps_still_run(self) -> None:
        self.cleaner.bedrock_east.delete_imported_model.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "DeleteImportedModel")

        errors = self.cleaner._delete_bedrock_import()

        self.assertEqual(len(errors), 1)
        self.cleaner.s3_east.delete_bucket.assert_called_once()
        self.cleaner._delete_role.assert_called_once()


if __name__ == "__main__":
    unittest.main()
