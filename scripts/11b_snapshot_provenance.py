"""Record what actually ran, from AWS, for offline audit (read-only).

Saves three evidence records in artifacts/:
- agent_provenance.json (+ agent_deployed_source.zip): the AgentCore runtime
  version that served the rollouts, its image and base-image digests, every
  package in the image (from the CodeBuild log), the exact source bundle, and
  when each recorded job started;
- dataset_provenance.json: the S3 datasets the jobs read, compared prompt by
  prompt with data/*.csv;
- bedrock_import_provenance.json: the files the Bedrock import read, compared
  file by file with the trained model package's merged weights.
"""

# region Imports & setup
from __future__ import annotations

import _bootstrap as bootstrap

from aws.records.provenance import capture_agent, capture_datasets, capture_import

# endregion


# region Entry point
def main() -> None:
    """Captures and saves the three provenance records."""
    bootstrap.parse_args(__doc__.splitlines()[0])
    bootstrap.banner("Provenance (11b)", "agent image and source, datasets, imported weights")
    workflow = bootstrap.load_workflow()
    agent = capture_agent(workflow.session, bootstrap.CONFIG, workflow.state, workflow.store)
    print(f"Runtime {agent['runtime']['arn']} version {agent['runtime']['version']}, "
          f"last updated {agent['runtime']['last_updated']}")
    print(f"Image {agent['image']['digest']} on {agent['image']['base_image']}; "
          f"{len(agent['packages'])} packages; source bundle sha256 {agent['source_bundle']['sha256'][:16]}...")
    bootstrap.show(agent["jobs"])
    datasets = capture_datasets(workflow.session, workflow.state, bootstrap.DATA_DIR, workflow.store)
    bootstrap.show(datasets["splits"])
    imported = capture_import(workflow.session, workflow.store)
    if imported is None:
        print("No Bedrock import recorded.")
        return
    bootstrap.show(imported["files"])


if __name__ == "__main__":
    bootstrap.run(main)
# endregion
