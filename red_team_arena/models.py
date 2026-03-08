"""Data models for the Angry Claw environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# --- Tool Definitions ---

class ToolDef(BaseModel):
    """Definition of an available tool."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A single tool call issued by the agent."""
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Simulated result of a tool call."""
    tool: str
    success: bool = True
    result: str = ""
    blocked_by_policy: bool = False


# --- Content Items ---

class ContentItem(BaseModel):
    """A single content item in an episode -- message, document, skill request, etc."""
    channel: str  # telegram, slack, email, skill_store, system
    sender: str
    content: str
    available_tools: List[str] = Field(default_factory=list)
    expected_safe_calls: List[ToolCall] = Field(default_factory=list)
    injected_unsafe_calls: List[ToolCall] = Field(default_factory=list)
    is_injection: bool = False
    tier: int = 0  # 0 = legitimate, 1-6 = attack tier
    attack_type: str = ""
    policy_drift_event: Optional[Dict[str, str]] = None  # If set, policy changes at this step


# --- OpenEnv Types ---

class RedTeamAction(Action):
    """Agent's response: one or more tool calls."""
    tool_calls: List[ToolCall] = Field(default_factory=list)
    reasoning: str = ""


class RedTeamObservation(Observation):
    """What the agent sees at each step."""
    channel: str = ""
    sender: str = ""
    content: str = ""
    available_tools: List[ToolDef] = Field(default_factory=list)
    active_policies: Dict[str, str] = Field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)
    expert_feedback: str = ""
    policy_drift_notice: str = ""
    step_number: int = 0
    total_steps: int = 0


class RedTeamState(State):
    """Full environment state for checkpointing and logging."""
    episode_id: Optional[str] = None
    step_count: int = 0
    episode_reward: float = 0.0
    # Curriculum stats
    tier_catch_rates: Dict[int, float] = Field(default_factory=dict)
    active_tiers: List[int] = Field(default_factory=list)
    max_tier_unlocked: int = 2
    episodes_completed: int = 0
    total_injections_seen: int = 0
    total_injections_caught: int = 0
    # Policy drift
    drift_active: bool = False
    drift_count: int = 0
