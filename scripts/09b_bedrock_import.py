"""Deployment option 2: import the fine-tuned model into Amazon Bedrock (needs --confirm).

The official guide's second option, Bedrock Custom Model Import: serverless
(no GPU instance and no endpoint quota), billed per active model minute and
scaled to zero when idle, plus monthly storage. GPT-OSS imports run only in
us-east-1, so the merged fine-tuned weights are copied there first. Two files
are fixed in that copy, as the Bedrock documentation requires for GPT-OSS: the
chat template in tokenizer_config.json and all three end-of-sequence IDs in
generation_config.json. The training output is not changed. A recorded import
of the same model package is reused.
"""

# region Imports & setup
from __future__ import annotations

import _bootstrap as bootstrap

from aws.deploy.bedrock_import import BedrockImporter, merged_weights_uri

TIMEOUT_SECONDS = 3 * 3600
TERMINAL = ("Completed", "Failed")
FIELDS = ("job_name", "status", "source_uri", "imported_model_arn", "started_at", "ended_at", "failure_message")
# endregion


# region Entry point
def main() -> None:
    """Imports (when confirmed) or shows the recorded import, waiting while it runs."""
    args = bootstrap.parse_args(__doc__.splitlines()[0], BedrockImporter.CONFIRMATION)
    bootstrap.banner("Bedrock import (09b)", "merged fine-tuned weights -> Bedrock Custom Model Import")
    workflow = bootstrap.load_workflow()
    importer = BedrockImporter(bootstrap.CONFIG, workflow.state, workflow.store, workflow.session,
                               budget_gate=workflow.budget_gate)
    record = workflow.store.load(BedrockImporter.RECORD)
    if record is None:
        if not args.confirm:
            print(f"No import recorded. Pass --confirm {BedrockImporter.CONFIRMATION} to import.")
            return
        job = workflow.training_stage().run_or_attach("")
        if job is None or job.job_status != "Completed":
            raise RuntimeError("No completed training job: run scripts/06_train.py first.")
        package = job.output_model_package_arn
        record = importer.run(args.confirm, package, merged_weights_uri(workflow.session.client("sagemaker"), package))
    if record.get("status") not in TERMINAL:
        record = importer.wait(timeout_seconds=TIMEOUT_SECONDS)
    bootstrap.show_fields(record, FIELDS)
    if record.get("status") == "Completed":
        details = importer.describe()
        print("Imported model:", details["modelArn"], "| architecture:", details["modelArchitecture"])


if __name__ == "__main__":
    bootstrap.run(main)
# endregion
