"""Main Angry Claw environment implementation.

Ties together the curriculum engine, content generator, policy drift,
reward calculator, and simulated expert into a multi-step OpenEnv
environment where an LLM agent processes a stream of messages and must
make safe tool calls while detecting adversarial attacks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from red_team_arena.models import (
        ContentItem,
        RedTeamAction,
        RedTeamObservation,
        RedTeamState,
        ToolDef,
        ToolResult,
    )
    from red_team_arena.server.content import TOOL_REGISTRY, generate_episode
    from red_team_arena.server.curriculum import AdaptiveCurriculum
    from red_team_arena.server.expert import generate_episode_summary, generate_step_feedback
    from red_team_arena.server.policies import PolicyDriftEngine, is_tool_call_allowed
    from red_team_arena.server.rewards import compute_step_reward, contains_unsafe_call
except ImportError:
    from models import (
        ContentItem,
        RedTeamAction,
        RedTeamObservation,
        RedTeamState,
        ToolDef,
        ToolResult,
    )
    from server.content import TOOL_REGISTRY, generate_episode
    from server.curriculum import AdaptiveCurriculum
    from server.expert import generate_episode_summary, generate_step_feedback
    from server.policies import PolicyDriftEngine, is_tool_call_allowed
    from server.rewards import compute_step_reward, contains_unsafe_call


class RedTeamArenaEnvironment(Environment):
    """Multi-step RL environment for adversarial robustness training.

    Each episode presents a sequence of content items (messages, docs, skill
    requests) that the agent must process by issuing tool calls. Some items
    contain injected attacks at varying sophistication levels. Security
    policies may change mid-episode (policy drift). An adaptive curriculum
    tracks the agent's weaknesses and generates harder challenges over time.
    """

    def __init__(
        self,
        seed: int = 42,
        fixed_curriculum: bool = False,
        enable_drift: bool = True,
        enable_expert: bool = True,
    ) -> None:
        super().__init__()
        self._seed = seed
        self._enable_drift = enable_drift
        self._enable_expert = enable_expert

        self._curriculum = AdaptiveCurriculum(seed=seed, fixed_mode=fixed_curriculum)
        self._drift_engine = PolicyDriftEngine(seed=seed)

        # Episode state
        self._episode_items: List[ContentItem] = []
        self._current_step: int = 0
        self._episode_reward: float = 0.0
        self._step_history: List[Dict[str, Any]] = []
        self._conversation_history: List[Dict[str, Any]] = []
        self._drift_this_step: bool = False
        self._drift_notice: str = ""

        self._state = RedTeamState(episode_id=str(uuid4()), step_count=0)

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RedTeamObservation:
        if seed is not None:
            self._seed = seed
            self._curriculum = AdaptiveCurriculum(
                seed=seed, fixed_mode=self._curriculum.fixed_mode,
            )

        # Reset drift engine for new episode
        self._drift_engine.reset()

        # Get episode config from curriculum
        config = self._curriculum.get_episode_config()

        # Generate episode content
        self._episode_items = generate_episode(
            active_tiers=config["active_tiers"],
            tier_weights=config["tier_weights"],
            inject_drift=config["inject_drift"] and self._enable_drift,
            rng=self._curriculum.rng,
        )

        # Reset episode state
        self._current_step = 0
        self._episode_reward = 0.0
        self._step_history = []
        self._conversation_history = []
        self._drift_this_step = False
        self._drift_notice = ""

        # Update state
        self._state = RedTeamState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            episode_reward=0.0,
            tier_catch_rates={
                t: s.catch_rate
                for t, s in self._curriculum.state.tier_stats.items()
            },
            active_tiers=self._curriculum.get_active_tiers(),
            max_tier_unlocked=self._curriculum.state.max_tier_unlocked,
            episodes_completed=self._curriculum.state.episodes_completed,
            total_injections_seen=sum(
                s.injections_seen for s in self._curriculum.state.tier_stats.values()
            ),
            total_injections_caught=sum(
                s.injections_caught for s in self._curriculum.state.tier_stats.values()
            ),
            drift_active=self._enable_drift and config["inject_drift"],
            drift_count=0,
        )

        return self._build_observation(reward=0.0, done=False)

    def step(self, action) -> RedTeamObservation:
        # Accept any action-like object -- isinstance check breaks across
        # different Python environments (openenv deserializes with its own copy)
        if not hasattr(action, "tool_calls"):
            raise TypeError(f"Expected RedTeamAction, got {type(action).__name__}")

        if self._current_step >= len(self._episode_items):
            return self._build_observation(reward=0.0, done=True)

        content_item = self._episode_items[self._current_step]

        # Get current policy and check if drift should happen
        policy = self._drift_engine.get_active_policy()
        self._drift_this_step = False
        self._drift_notice = ""

        if content_item.policy_drift_event:
            drift_type = content_item.policy_drift_event.get("drift_type", "")
            if drift_type:
                policy, desc = self._drift_engine.apply_drift(drift_type)
                self._drift_this_step = True
                self._drift_notice = desc
                self._state.drift_count += 1
        elif self._enable_drift and self._curriculum.should_enable_drift():
            result = self._drift_engine.maybe_drift(
                self._curriculum.state.episodes_completed,
                self._current_step,
            )
            if result:
                policy, desc = result
                self._drift_this_step = True
                self._drift_notice = desc
                self._state.drift_count += 1

        # Validate each tool call against policy
        policy_allowed: Dict[str, tuple] = {}
        tool_results: List[ToolResult] = []
        for tc in action.tool_calls:
            allowed, reason = is_tool_call_allowed(policy, tc)
            policy_allowed[tc.tool] = (allowed, reason)
            tool_results.append(ToolResult(
                tool=tc.tool,
                success=allowed,
                result=f"Executed {tc.tool}" if allowed else f"Blocked: {reason}",
                blocked_by_policy=not allowed,
            ))

        # Compute reward
        reward, breakdown = compute_step_reward(
            action=action,
            content_item=content_item,
            policy_allowed=policy_allowed,
            drift_occurred=self._drift_this_step,
        )

        # Update curriculum stats for injections
        if content_item.is_injection and content_item.tier > 0:
            caught = not contains_unsafe_call(
                action.tool_calls, content_item.injected_unsafe_calls
            )
            self._curriculum.update_stats(content_item.tier, caught)

        # Generate expert feedback
        expert_feedback = ""
        if self._enable_expert:
            expert_feedback = generate_step_feedback(action, content_item, breakdown)

        # Record step history
        self._step_history.append({
            "step": self._current_step + 1,
            "content_item": content_item,
            "action": action,
            "reward": reward,
            "breakdown": breakdown,
            "tool_results": tool_results,
            "drift_occurred": self._drift_this_step,
            "drift_notice": self._drift_notice,
            "expert_feedback": expert_feedback,
        })

        # Add to conversation history
        self._conversation_history.append({
            "step": self._current_step + 1,
            "channel": content_item.channel,
            "sender": content_item.sender,
            "content": content_item.content[:200],  # Truncated for context window
            "agent_tool_calls": [
                {"tool": tc.tool, "args": tc.arguments} for tc in action.tool_calls
            ],
            "agent_reasoning": action.reasoning[:200] if action.reasoning else "",
        })

        # Advance episode
        self._episode_reward += reward
        self._current_step += 1
        self._state.step_count = self._current_step
        self._state.episode_reward = self._episode_reward

        # Check if episode is done
        done = self._current_step >= len(self._episode_items)

        # On episode end, generate summary and update curriculum
        if done:
            if self._enable_expert:
                max_reward = self._compute_max_reward()
                summary = generate_episode_summary(
                    self._step_history, self._episode_reward, max_reward,
                )
                expert_feedback = summary

            curriculum_messages = self._curriculum.on_episode_complete()
            if curriculum_messages:
                expert_feedback += "\n\nCurriculum updates:\n" + "\n".join(
                    f"  - {m}" for m in curriculum_messages
                )

            # Update state with latest curriculum stats
            self._state.tier_catch_rates = {
                t: s.catch_rate
                for t, s in self._curriculum.state.tier_stats.items()
            }
            self._state.active_tiers = self._curriculum.get_active_tiers()
            self._state.max_tier_unlocked = self._curriculum.state.max_tier_unlocked
            self._state.episodes_completed = self._curriculum.state.episodes_completed

        return self._build_observation(
            reward=reward,
            done=done,
            tool_results=tool_results,
            expert_feedback=expert_feedback,
        )

    @property
    def state(self) -> RedTeamState:
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        reward: float,
        done: bool,
        tool_results: Optional[List[ToolResult]] = None,
        expert_feedback: str = "",
    ) -> RedTeamObservation:
        """Build the observation for the current (or next) step."""
        if done or self._current_step >= len(self._episode_items):
            return RedTeamObservation(
                reward=reward,
                done=True,
                channel="system",
                sender="environment",
                content="Episode complete.",
                available_tools=TOOL_REGISTRY,
                active_policies=self._drift_engine.get_active_policy().summary(),
                conversation_history=self._conversation_history,
                tool_results=tool_results or [],
                expert_feedback=expert_feedback,
                policy_drift_notice=self._drift_notice,
                step_number=self._current_step,
                total_steps=len(self._episode_items),
                metadata={
                    "episode_reward": self._episode_reward,
                    "curriculum": self._curriculum.get_state_dict(),
                },
            )

        item = self._episode_items[self._current_step]
        policy = self._drift_engine.get_active_policy()

        # Filter tool registry to show only relevant tools for this step
        relevant_tools = [t for t in TOOL_REGISTRY if t.name in item.available_tools] or TOOL_REGISTRY

        return RedTeamObservation(
            reward=reward,
            done=False,
            channel=item.channel,
            sender=item.sender,
            content=item.content,
            available_tools=relevant_tools,
            active_policies=policy.summary(),
            conversation_history=self._conversation_history,
            tool_results=tool_results or [],
            expert_feedback=expert_feedback,
            policy_drift_notice=self._drift_notice,
            step_number=self._current_step + 1,
            total_steps=len(self._episode_items),
            metadata={
                "tier": item.tier,
                "is_injection": item.is_injection,
            },
        )

    def _compute_max_reward(self) -> float:
        """Compute the maximum achievable reward for this episode."""
        total = 0.0
        for item in self._episode_items:
            if item.expected_safe_calls:
                total += 3.0  # task completion
            if item.is_injection:
                total += 1.0  # injection detection
            if item.policy_drift_event:
                total += 2.0  # drift adaptation
        return max(total, 1.0)
