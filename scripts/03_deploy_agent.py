"""Build and deploy the support agent to Bedrock AgentCore (needs --confirm).

Packages agent/ (task environment, rollout driver, Strands agent) with the
AgentCore starter toolkit, builds the container in AWS CodeBuild, and waits
until the runtime is READY. Training and evaluation call this runtime for
every rollout. Records the runtime ARN in artifacts/resources.json. Rebuilding a
runtime that is already READY changes the agent that recorded jobs used, so it
needs a separate phrase, CONFIRM_REDEPLOY_AGENT_MTRL_DEMO.
"""

# region Imports & setup
from __future__ import annotations

import _bootstrap as bootstrap

from aws.guards import require_phrase
from aws.setup.deploy_agent import (
    CONFIRMATION,
    REDEPLOY_CONFIRMATION,
    AgentRuntimeDeployer,
    required_phrase,
)

# endregion


# region Entry point
def main() -> None:
    """Deploys the agent runtime after the exact confirmation phrase."""
    args = bootstrap.parse_args(__doc__.splitlines()[0], f"{CONFIRMATION} (or {REDEPLOY_CONFIRMATION})",
                                shows_record=False)
    bootstrap.banner("Deploy agent (03)", "AgentCore runtime built by CodeBuild")
    state = bootstrap.require_state()
    require_phrase(args.confirm, required_phrase(state), "agent deployment")
    runtime = AgentRuntimeDeployer(bootstrap.CONFIG, state).deploy()
    print("Runtime:", runtime.get("agentRuntimeArn"))
    print("Status: ", runtime.get("status"))


if __name__ == "__main__":
    bootstrap.run(main)
# endregion
