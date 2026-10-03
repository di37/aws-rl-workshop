# agent/

The support agent that Amazon Bedrock AgentCore runs. During training and evaluation, SageMaker calls it once per rollout: it opens a ticket, asks the policy model (GPT-OSS-20B) for one move at a time, applies each move to a small support simulator, and returns the score.

| File | What it is |
|---|---|
| `app.py` | The AgentCore entrypoint, `handle_rollout`. A `smoke_test` payload gets a health check; anything else is one rollout, run by a Strands agent with a single tool, `take_support_action(action)`, and returned as reward plus diagnostics for SageMaker and MLflow. |
| `environment.py` | The support simulator: deterministic states, moves and rewards (below). |
| `rollout_driver.py` | Keeps a rollout going when the model names its next move in plain text instead of a tool call. Without it every baseline rollout ended after two turns. |
| `Dockerfile` | `python:3.12-slim`, serves `python -m app` on port 8080. |
| `requirements.txt`, `requirements.lock.txt` | The five direct dependencies, and the exact 139 packages of the image that served every rollout. |
| `test_*.py` | 48 unit tests for the three modules. |

## The environment

A ticket has four steps, and only one move per step makes progress:

| State | Moves offered | The move that helps → next state |
|---|---|---|
| `new_ticket` | check_outage, restart_router, close_ticket | `check_outage` → `no_outage` |
| `no_outage` | inspect_router_lights, restart_router, close_ticket | `inspect_router_lights` → `red_wan_light` |
| `red_wan_light` | check_account_config, replace_router, close_ticket | `check_account_config` → `config_mismatch` |
| `config_mismatch` | apply_config_fix, escalate, close_ticket | `apply_config_fix` → `resolved` |

- **Reward:** progress ÷ 4, so 0.25 per correct step and 1.0 when the connection is restored.
- **Wrong moves** use up one of the 4 turns and change nothing, so a wrong first move caps the score at 0.75.
- **Move order** is shuffled for each state from a seed the rollout records, so the model can't learn positions.

## Deploying and testing

- **Deploy:** `python scripts/03_deploy_agent.py --confirm CONFIRM_DEPLOY_AGENT_MTRL_DEMO` builds the image with CodeBuild and deploys it to AgentCore.
- **Test:** `python -m pytest agent` from the project root, with no AWS access needed.
- **Provenance:** `environment.py`, `rollout_driver.py` and `Dockerfile` are byte-identical to the bundle that ran, saved as `artifacts/agent_deployed_source.zip`. `app.py` keeps the same system prompt, tool and opening message; it was later refactored so live inference can reuse the episode code. An invariant checks this.
