"""HTTP client for the Red Team Arena environment."""

from __future__ import annotations

from typing import Any, Dict, List

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import (
    RedTeamAction,
    RedTeamObservation,
    RedTeamState,
    ToolCall,
    ToolDef,
    ToolResult,
)


class RedTeamArenaEnv(EnvClient[RedTeamAction, RedTeamObservation, RedTeamState]):
    """HTTP client for connecting to a Red Team Arena server.

    Example:
        >>> client = RedTeamArenaEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.content)
        >>>
        >>> result = client.step(RedTeamAction(
        ...     tool_calls=[ToolCall(tool="read_file", arguments={"path": "report.pdf"})],
        ...     reasoning="Reading the requested file."
        ... ))
        >>> print(result.reward)

    Example with Docker:
        >>> client = RedTeamArenaEnv.from_docker_image("red-team-arena:latest")
        >>> result = client.reset()

    Example with HuggingFace Space:
        >>> client = RedTeamArenaEnv.from_hub("your-username/red-team-arena")
        >>> result = client.reset()
    """

    def _step_payload(self, action: RedTeamAction) -> Dict:
        return {
            "tool_calls": [
                {"tool": tc.tool, "arguments": tc.arguments}
                for tc in action.tool_calls
            ],
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[RedTeamObservation]:
        obs_data = payload.get("observation", {})

        available_tools = [
            ToolDef(
                name=t.get("name", ""),
                description=t.get("description", ""),
                parameters=t.get("parameters", {}),
            )
            for t in obs_data.get("available_tools", [])
            if isinstance(t, dict)
        ]

        tool_results = [
            ToolResult(
                tool=tr.get("tool", ""),
                success=tr.get("success", True),
                result=tr.get("result", ""),
                blocked_by_policy=tr.get("blocked_by_policy", False),
            )
            for tr in obs_data.get("tool_results", [])
            if isinstance(tr, dict)
        ]

        observation = RedTeamObservation(
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
            channel=obs_data.get("channel", ""),
            sender=obs_data.get("sender", ""),
            content=obs_data.get("content", ""),
            available_tools=available_tools,
            active_policies=obs_data.get("active_policies", {}),
            conversation_history=obs_data.get("conversation_history", []),
            tool_results=tool_results,
            expert_feedback=obs_data.get("expert_feedback", ""),
            policy_drift_notice=obs_data.get("policy_drift_notice", ""),
            step_number=obs_data.get("step_number", 0),
            total_steps=obs_data.get("total_steps", 0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> RedTeamState:
        return RedTeamState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            episode_reward=payload.get("episode_reward", 0.0),
            tier_catch_rates=payload.get("tier_catch_rates", {}),
            active_tiers=payload.get("active_tiers", []),
            max_tier_unlocked=payload.get("max_tier_unlocked", 2),
            episodes_completed=payload.get("episodes_completed", 0),
            total_injections_seen=payload.get("total_injections_seen", 0),
            total_injections_caught=payload.get("total_injections_caught", 0),
            drift_active=payload.get("drift_active", False),
            drift_count=payload.get("drift_count", 0),
        )
