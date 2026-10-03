"""Unit tests for importing the fine-tuned model into Amazon Bedrock."""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import MagicMock

from aws.deploy.bedrock_import import (
    GENERATION_EOS_TOKEN_IDS,
    BedrockInvokeClient,
    chat_with_imported_model,
    files_to_copy,
    fixed_generation_config,
    fixed_tokenizer_config,
    import_role_policy,
    import_role_trust,
    merged_weights_uri,
)


class ConfigFixTests(unittest.TestCase):
    """Verifies the two documented GPT-OSS import fixes, without mutation."""

    def test_chat_template_is_embedded_in_tokenizer_config(self) -> None:
        original = {"tokenizer_class": "PreTrainedTokenizerFast"}

        fixed = fixed_tokenizer_config(original, "{{ messages }}")

        self.assertEqual(fixed["chat_template"], "{{ messages }}")
        self.assertNotIn("chat_template", original)

    def test_generation_config_gets_all_three_end_tokens(self) -> None:
        original = {"eos_token_id": [200002, 199999], "pad_token_id": 199999}

        fixed = fixed_generation_config(original)

        self.assertEqual(fixed["eos_token_id"], [200002, 199999, 200012])
        self.assertEqual(GENERATION_EOS_TOKEN_IDS, [200002, 199999, 200012])
        self.assertEqual(original["eos_token_id"], [200002, 199999])
        self.assertEqual(fixed["pad_token_id"], 199999)


class FileSelectionTests(unittest.TestCase):
    """Verifies only Hugging Face model files are imported."""

    def test_service_metadata_and_empty_scripts_are_skipped(self) -> None:
        names = ["config.json", "model-00000-of-00002.safetensors", "inference.py",
                 "__model_info__.json", "__script_info__.json", "version", "tokenizer.json"]

        self.assertEqual(
            files_to_copy(names),
            ["config.json", "model-00000-of-00002.safetensors", "tokenizer.json"],
        )


class RoleDocumentTests(unittest.TestCase):
    """Verifies the import role is limited to Bedrock jobs and one bucket."""

    def test_trust_policy_is_scoped_to_this_account_import_jobs(self) -> None:
        statement = import_role_trust("123456789012", "us-east-1")["Statement"][0]

        self.assertEqual(statement["Principal"], {"Service": "bedrock.amazonaws.com"})
        self.assertEqual(statement["Condition"]["StringEquals"]["aws:SourceAccount"], "123456789012")
        self.assertEqual(
            statement["Condition"]["ArnLike"]["aws:SourceArn"],
            "arn:aws:bedrock:us-east-1:123456789012:model-import-job/*",
        )

    def test_permissions_only_read_the_model_bucket(self) -> None:
        statement = import_role_policy("model-bucket")["Statement"][0]

        self.assertEqual(sorted(statement["Action"]), ["s3:GetObject", "s3:ListBucket"])
        self.assertEqual(
            statement["Resource"], ["arn:aws:s3:::model-bucket", "arn:aws:s3:::model-bucket/*"]
        )


class InvokeClientTests(unittest.TestCase):
    """Verifies Strands' OpenAI-format request reaches Bedrock InvokeModel."""

    def test_invoke_endpoint_forwards_body_and_returns_it(self) -> None:
        runtime = MagicMock()
        runtime.invoke_model.return_value = {"body": io.BytesIO(b'{"choices": []}')}
        client = BedrockInvokeClient(runtime, "arn:model", exceptions="errors")
        body = json.dumps({"messages": [], "stream": False})

        response = client.invoke_endpoint(EndpointName="unused", Body=body,
                                          ContentType="application/json", Accept="application/json")

        runtime.invoke_model.assert_called_once_with(
            modelId="arn:model", body=body, contentType="application/json", accept="application/json"
        )
        self.assertEqual(json.loads(response["Body"].read()), {"choices": []})
        self.assertEqual(client.exceptions, "errors")

    def test_streaming_is_refused(self) -> None:
        client = BedrockInvokeClient(MagicMock(), "arn:model", exceptions=None)

        with self.assertRaises(NotImplementedError):
            client.invoke_endpoint_with_response_stream(Body="{}")



class ChatTests(unittest.TestCase):
    """Verifies the guide-style plain chat call to an imported model."""

    def test_chat_sends_openai_messages_and_returns_reply_text(self) -> None:
        runtime = MagicMock()
        runtime.invoke_model.return_value = {"body": io.BytesIO(json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "Check for an outage."}}]}
        ).encode())}

        reply = chat_with_imported_model(runtime, "arn:model", "My internet is down.", 128)

        sent = json.loads(runtime.invoke_model.call_args.kwargs["body"])
        self.assertEqual(sent["messages"], [{"role": "user", "content": "My internet is down."}])
        self.assertEqual(sent["max_tokens"], 128)
        self.assertEqual(reply, "Check for an outage.")



class MergedWeightsUriTests(unittest.TestCase):
    """Verifies the merged checkpoint folder is found from the model package."""

    def test_uri_points_at_hf_merged_folder(self) -> None:
        sagemaker = MagicMock()
        sagemaker.describe_model_package.return_value = {"InferenceSpecification": {"Containers": [
            {"ModelDataSource": {"S3DataSource": {"S3Uri": "s3://b/out/model/"}}}]}}

        self.assertEqual(
            merged_weights_uri(sagemaker, "arn:package"), "s3://b/out/model/checkpoints/hf_merged/"
        )


if __name__ == "__main__":
    unittest.main()


class ImporterRecordTests(unittest.TestCase):
    """Verifies per-package folders, gates, and safe reuse of import records."""

    PACKAGE = "arn:aws:sagemaker:us-west-2:123456789012:model-package/demo-group/3"

    def setUp(self) -> None:
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from aws.config import DemoConfig
        from aws.deploy.bedrock_import import BedrockImporter
        from aws.records.evidence import EvidenceStore

        self.directory = tempfile.TemporaryDirectory()
        self.store = EvidenceStore(Path(self.directory.name))
        self.gate = MagicMock()
        self.importer = BedrockImporter.__new__(BedrockImporter)
        self.importer.config = DemoConfig()
        self.importer.state = SimpleNamespace(account_id="123456789012", region="us-west-2")
        self.importer.store = self.store
        self.importer.budget_gate = self.gate
        self.importer.bucket = "b"
        self.importer._bedrock = MagicMock()
        self.importer._bedrock.create_model_import_job.return_value = {"jobArn": "arn:job"}

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_destination_folder_is_unique_per_model_package(self) -> None:
        from aws.deploy.bedrock_import import destination_prefix

        self.assertEqual(destination_prefix(self.PACKAGE), "imported-models/demo-group-v3/")

    def test_import_bucket_name_is_shared(self) -> None:
        from aws.config import DemoConfig
        from aws.deploy.bedrock_import import import_bucket_name

        self.assertEqual(import_bucket_name(DemoConfig(), "123456789012"),
                         "sagemaker-mtrl-demo-123456789012-us-east-1")

    def test_start_requires_phrase_and_budget_gate(self) -> None:
        from aws.deploy.bedrock_import import BedrockImporter

        with self.assertRaises(PermissionError):
            self.importer.start("go", "arn:role", self.PACKAGE)
        self.importer.start(BedrockImporter.CONFIRMATION, "arn:role", self.PACKAGE)

        self.gate.assert_called_once()
        record = self.store.load(BedrockImporter.RECORD)
        self.assertEqual(record["model_package_arn"], self.PACKAGE)
        self.assertTrue(record["source_uri"].endswith("imported-models/demo-group-v3/"))

    def test_completed_record_for_same_package_is_reused(self) -> None:
        from aws.deploy.bedrock_import import BedrockImporter

        self.store.save(BedrockImporter.RECORD, {"model_package_arn": self.PACKAGE, "status": "Completed"})

        record = self.importer.start(BedrockImporter.CONFIRMATION, "arn:role", self.PACKAGE)

        self.assertEqual(record["status"], "Completed")
        self.importer._bedrock.create_model_import_job.assert_not_called()

    def test_failed_or_other_package_record_is_refused(self) -> None:
        from aws.deploy.bedrock_import import BedrockImporter

        for record in ({"model_package_arn": self.PACKAGE, "status": "Failed"},
                       {"model_package_arn": "arn:other", "status": "Completed"}):
            self.store.save(BedrockImporter.RECORD, record)
            with self.assertRaisesRegex(RuntimeError, "move artifacts/bedrock_import.json aside"):
                self.importer.start(BedrockImporter.CONFIRMATION, "arn:role", self.PACKAGE)


class RoleTaggingTests(unittest.TestCase):
    """Verifies the import role is tagged and its trust policy refreshed."""

    def test_existing_role_gets_current_trust_and_tags(self) -> None:
        from types import SimpleNamespace

        from botocore.exceptions import ClientError

        from aws.config import DemoConfig
        from aws.deploy.bedrock_import import BedrockImporter

        importer = BedrockImporter.__new__(BedrockImporter)
        importer.config = DemoConfig()
        importer.state = SimpleNamespace(account_id="123456789012")
        importer.bucket = "b"
        importer._iam = MagicMock()
        importer._iam.create_role.side_effect = ClientError(
            {"Error": {"Code": "EntityAlreadyExists", "Message": "exists"}}, "CreateRole")
        importer._iam.get_role.return_value = {"Role": {"Arn": "arn:role"}}

        self.assertEqual(importer.ensure_role(), "arn:role")
        importer._iam.update_assume_role_policy.assert_called_once()
        tags = importer._iam.tag_role.call_args.kwargs["Tags"]
        self.assertIn({"Key": "Project", "Value": "sagemaker-mtrl-demo"}, tags)


class AdapterContractTests(unittest.TestCase):
    """Runs the adapter through Strands' real SageMakerAIModel.stream."""

    def test_tool_call_from_bedrock_reaches_strands_as_tool_use(self) -> None:
        import asyncio

        import boto3
        from strands.models.sagemaker import SageMakerAIModel

        model = SageMakerAIModel(
            endpoint_config={"endpoint_name": "unused", "region_name": "us-east-1"},
            payload_config={"max_tokens": 64, "stream": False},
            boto_session=boto3.Session(region_name="us-east-1"),
        )
        runtime = MagicMock()
        runtime.invoke_model.return_value = {"body": io.BytesIO(json.dumps({"choices": [{
            "message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "take_support_action", "arguments": "{\"action\": \"check_outage\"}"}}]},
            "finish_reason": "tool_calls"}]}).encode())}
        model.client = BedrockInvokeClient(runtime, "arn:model", model.client.exceptions)
        spec = {"name": "take_support_action", "description": "Act.",
                "inputSchema": {"json": {"type": "object", "properties": {"action": {"type": "string"}}}}}

        async def collect() -> list:
            return [event async for event in model.stream(
                [{"role": "user", "content": [{"text": "help"}]}], tool_specs=[spec])]

        events = asyncio.run(collect())

        self.assertIn("take_support_action", json.dumps(events, default=str))
        self.assertIn("toolUse", json.dumps(events, default=str))
        self.assertEqual(runtime.invoke_model.call_args.kwargs["modelId"], "arn:model")
