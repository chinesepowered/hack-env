"""Gradio web interface for the Red Team Arena environment.

Provides an interactive UI on Hugging Face Spaces (or locally) that lets users
step through episodes, issue tool calls, observe reward breakdowns, and monitor
the adaptive curriculum.  Designed for the ``RedTeamArenaEnvironment`` from
``red_team_arena.server.environment``.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from red_team_arena.models import RedTeamAction, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_policies(policies: Dict[str, str]) -> str:
    """Format the active policies dict into a readable multiline string."""
    if not policies:
        return "No active policies."
    lines: List[str] = []
    tool_lines: List[str] = []
    for key, value in sorted(policies.items()):
        if key.startswith("tool:"):
            tool_lines.append(f"  {key.removeprefix('tool:'):20s} {value}")
        else:
            lines.append(f"  {key:26s} {value}")
    result = "Global Policies:\n" + "\n".join(lines)
    if tool_lines:
        result += "\n\nTool Permissions:\n" + "\n".join(tool_lines)
    return result


def _fmt_tool_results(tool_results: list) -> str:
    """Format tool results into a readable block."""
    if not tool_results:
        return "No tool results yet."
    lines: List[str] = []
    for tr in tool_results:
        status = "BLOCKED" if tr.blocked_by_policy else ("OK" if tr.success else "FAIL")
        icon = "[x]" if tr.blocked_by_policy else ("[+]" if tr.success else "[-]")
        lines.append(f"{icon} {tr.tool:20s}  {status}  — {tr.result}")
    return "\n".join(lines)


def _fmt_reward_breakdown(breakdown: Dict[str, float]) -> str:
    """Format a reward breakdown dict into a readable table."""
    if not breakdown:
        return "No reward data."
    lines: List[str] = []
    total = 0.0
    for component, value in breakdown.items():
        sign = "+" if value > 0 else (" " if value == 0 else "")
        lines.append(f"  {component:24s} {sign}{value:.2f}")
        total += value
    lines.append(f"  {'':24s} ------")
    sign = "+" if total > 0 else (" " if total == 0 else "")
    lines.append(f"  {'TOTAL':24s} {sign}{total:.2f}")
    return "\n".join(lines)


def _fmt_conversation_history(history: List[Dict[str, Any]]) -> str:
    """Format conversation history into a compact readable log."""
    if not history:
        return "No history yet."
    lines: List[str] = []
    for entry in history:
        step = entry.get("step", "?")
        channel = entry.get("channel", "?")
        sender = entry.get("sender", "?")
        content_preview = entry.get("content", "")[:120]
        calls = entry.get("agent_tool_calls", [])
        reasoning = entry.get("agent_reasoning", "")

        lines.append(f"--- Step {step} [{channel}] from {sender} ---")
        lines.append(f"  Content: {content_preview}...")
        if calls:
            call_strs = [f"{c['tool']}({json.dumps(c.get('args', {}), default=str)[:80]})" for c in calls]
            lines.append(f"  Agent calls: {', '.join(call_strs)}")
        else:
            lines.append("  Agent calls: (none)")
        if reasoning:
            lines.append(f"  Reasoning: {reasoning[:120]}...")
        lines.append("")
    return "\n".join(lines)


def _fmt_curriculum_dashboard(state: Any) -> str:
    """Format curriculum state into a dashboard string."""
    lines: List[str] = [
        "=" * 50,
        "  CURRICULUM DASHBOARD",
        "=" * 50,
        "",
        f"  Episodes completed:    {state.episodes_completed}",
        f"  Max tier unlocked:     {state.max_tier_unlocked}",
        f"  Active tiers:          {', '.join(str(t) for t in state.active_tiers) if state.active_tiers else 'N/A'}",
        f"  Drift active:          {'Yes' if state.drift_active else 'No'}",
        f"  Drift count:           {state.drift_count}",
        f"  Total injections seen: {state.total_injections_seen}",
        f"  Total caught:          {state.total_injections_caught}",
        "",
        "  Catch Rates per Tier:",
    ]
    if state.tier_catch_rates:
        for tier in sorted(state.tier_catch_rates.keys()):
            rate = state.tier_catch_rates[tier]
            bar_len = int(rate * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            lines.append(f"    Tier {tier}: [{bar}] {rate:.0%}")
    else:
        lines.append("    No data yet.")
    lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines)


def _available_tool_names(obs: Any) -> List[str]:
    """Extract available tool names from the observation."""
    if obs and hasattr(obs, "available_tools") and obs.available_tools:
        return [t.name for t in obs.available_tools]
    return []


# ---------------------------------------------------------------------------
# State container  (held per-session via gr.State)
# ---------------------------------------------------------------------------

def _make_session_state() -> Dict[str, Any]:
    return {
        "obs": None,
        "last_breakdown": {},
        "episode_active": False,
        "step_reward": 0.0,
        "cumulative_reward": 0.0,
    }


# ---------------------------------------------------------------------------
# Core callbacks
# ---------------------------------------------------------------------------

def _reset_episode(
    env: Any,
    seed_text: str,
    fixed_curriculum: bool,
    enable_drift: bool,
    enable_expert: bool,
    session: Dict[str, Any],
) -> Tuple:
    """Reset the environment and return updated UI components."""
    # Apply toggles that require re-init
    seed = int(seed_text) if seed_text.strip().isdigit() else 42
    env._enable_drift = enable_drift
    env._enable_expert = enable_expert
    if hasattr(env, "_curriculum"):
        env._curriculum.fixed_mode = fixed_curriculum

    obs = env.reset(seed=seed)
    session["obs"] = obs
    session["last_breakdown"] = {}
    session["episode_active"] = not obs.done
    session["step_reward"] = 0.0
    session["cumulative_reward"] = 0.0

    tool_names = _available_tool_names(obs)
    tool_dropdown_choices = tool_names if tool_names else ["(no tools)"]

    return (
        # Episode panel
        _build_episode_display(obs),
        # Policies
        _fmt_policies(obs.active_policies),
        # Conversation history
        _fmt_conversation_history(obs.conversation_history),
        # Results: reward breakdown
        "Episode reset. Take your first action.",
        # Results: tool results
        _fmt_tool_results(obs.tool_results),
        # Results: expert feedback
        obs.expert_feedback or "No feedback yet.",
        # Results: drift notice
        obs.policy_drift_notice or "No drift.",
        # Curriculum dashboard
        _fmt_curriculum_dashboard(env.state),
        # Tool dropdown update
        gr.update(choices=tool_dropdown_choices, value=tool_dropdown_choices[0] if tool_dropdown_choices else None),
        # Status bar
        _build_status_bar(obs, session),
        # Session state
        session,
    )


def _submit_action(
    env: Any,
    tool_name: str,
    tool_args_json: str,
    reasoning_text: str,
    session: Dict[str, Any],
) -> Tuple:
    """Parse user input, create a RedTeamAction, step the env, update UI."""
    if not session.get("episode_active", False):
        no_obs = session.get("obs")
        return (
            _build_episode_display(no_obs) if no_obs else "No active episode. Click Reset to start.",
            _fmt_policies(no_obs.active_policies) if no_obs else "",
            _fmt_conversation_history(no_obs.conversation_history) if no_obs else "",
            "Episode is over. Click Reset to start a new episode.",
            "",
            "",
            "",
            _fmt_curriculum_dashboard(env.state),
            _build_status_bar(no_obs, session) if no_obs else "No episode.",
            session,
        )

    # Build tool calls list
    tool_calls: List[ToolCall] = []
    if tool_name and tool_name != "(no tools)":
        try:
            args = json.loads(tool_args_json) if tool_args_json.strip() else {}
        except json.JSONDecodeError as exc:
            return (
                _build_episode_display(session["obs"]),
                _fmt_policies(session["obs"].active_policies),
                _fmt_conversation_history(session["obs"].conversation_history),
                f"JSON parse error in tool arguments: {exc}",
                "",
                "",
                "",
                _fmt_curriculum_dashboard(env.state),
                _build_status_bar(session["obs"], session),
                session,
            )
        tool_calls.append(ToolCall(tool=tool_name, arguments=args))

    action = RedTeamAction(tool_calls=tool_calls, reasoning=reasoning_text)

    # Step the environment
    obs = env.step(action)
    session["obs"] = obs

    # Extract reward breakdown from env internals
    breakdown: Dict[str, float] = {}
    if env._step_history:
        last_step = env._step_history[-1]
        breakdown = last_step.get("breakdown", {})
    session["last_breakdown"] = breakdown
    session["step_reward"] = obs.reward
    session["cumulative_reward"] = env._episode_reward
    session["episode_active"] = not obs.done

    tool_names = _available_tool_names(obs)

    return (
        # Episode panel
        _build_episode_display(obs),
        # Policies
        _fmt_policies(obs.active_policies),
        # Conversation history
        _fmt_conversation_history(obs.conversation_history),
        # Results: reward breakdown
        _fmt_reward_breakdown(breakdown),
        # Results: tool results
        _fmt_tool_results(obs.tool_results),
        # Results: expert feedback
        obs.expert_feedback or "No feedback.",
        # Results: drift notice
        obs.policy_drift_notice or "No drift this step.",
        # Curriculum dashboard
        _fmt_curriculum_dashboard(env.state),
        # Status bar
        _build_status_bar(obs, session),
        # Session state
        session,
    )


def _build_episode_display(obs: Any) -> str:
    """Build the main episode info display."""
    if obs is None:
        return "No episode loaded. Click Reset to start."

    lines: List[str] = []

    if obs.done:
        lines.append("*** EPISODE COMPLETE ***")
        lines.append("")

    lines.append(f"Step:    {obs.step_number} / {obs.total_steps}")
    lines.append(f"Channel: {obs.channel}")
    lines.append(f"Sender:  {obs.sender}")
    lines.append("")
    lines.append("--- Message Content ---")
    lines.append(obs.content)

    if obs.available_tools:
        lines.append("")
        lines.append("--- Available Tools ---")
        for tool in obs.available_tools:
            params = ", ".join(tool.parameters.keys()) if tool.parameters else "(none)"
            lines.append(f"  {tool.name}({params})")
            lines.append(f"    {tool.description}")

    return "\n".join(lines)


def _build_status_bar(obs: Any, session: Dict[str, Any]) -> str:
    """Build a compact status line."""
    if obs is None:
        return "No episode."
    status = "DONE" if obs.done else "ACTIVE"
    step_r = session.get("step_reward", 0.0)
    cum_r = session.get("cumulative_reward", 0.0)
    return (
        f"Status: {status}  |  "
        f"Step: {obs.step_number}/{obs.total_steps}  |  "
        f"Step Reward: {step_r:+.2f}  |  "
        f"Cumulative: {cum_r:+.2f}"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_red_team_gradio_app(env) -> gr.Blocks:
    """Build Gradio Blocks app for the Red Team Arena environment.

    Parameters
    ----------
    env : RedTeamArenaEnvironment
        A fully constructed environment instance.  The UI will call
        ``env.reset()`` and ``env.step(action)`` directly.

    Returns
    -------
    gr.Blocks
        The Gradio application, ready to ``.launch()``.
    """

    # -- Theme & CSS ----------------------------------------------------------
    custom_css = textwrap.dedent("""\
        .status-bar {
            font-family: monospace;
            font-size: 14px;
            padding: 8px 12px;
            border-radius: 6px;
            background: #1a1a2e;
            color: #e0e0e0;
        }
        .panel-header {
            font-weight: 700;
            font-size: 15px;
            margin-bottom: 4px;
        }
        .mono-box textarea {
            font-family: 'Fira Code', 'Cascadia Code', 'Consolas', monospace !important;
            font-size: 13px !important;
            line-height: 1.45 !important;
        }
    """)

    with gr.Blocks(
        title="Red Team Arena",
        theme=gr.themes.Soft(
            primary_hue="red",
            secondary_hue="slate",
        ),
        css=custom_css,
    ) as demo:

        # Shared state -------------------------------------------------------
        session_state = gr.State(_make_session_state)

        # Title ---------------------------------------------------------------
        gr.Markdown(
            "# Red Team Arena\n"
            "Interactive environment for adversarial robustness training. "
            "Process incoming messages, make tool calls, and try to detect "
            "injected attacks while respecting security policies."
        )

        # Status bar ----------------------------------------------------------
        status_bar = gr.Textbox(
            label="Status",
            value="No episode. Click Reset to start.",
            interactive=False,
            elem_classes=["status-bar"],
        )

        # =====================================================================
        # CONTROLS
        # =====================================================================
        with gr.Accordion("Controls", open=True):
            with gr.Row():
                seed_input = gr.Textbox(
                    label="Seed",
                    value="42",
                    max_lines=1,
                    scale=1,
                )
                fixed_curriculum_toggle = gr.Checkbox(
                    label="Fixed Curriculum",
                    value=False,
                    scale=1,
                )
                drift_toggle = gr.Checkbox(
                    label="Enable Policy Drift",
                    value=True,
                    scale=1,
                )
                expert_toggle = gr.Checkbox(
                    label="Enable Expert Feedback",
                    value=True,
                    scale=1,
                )
                reset_btn = gr.Button(
                    "Reset Episode",
                    variant="primary",
                    scale=1,
                )

        # =====================================================================
        # MAIN LAYOUT — three-column
        # =====================================================================
        with gr.Row(equal_height=False):

            # -----------------------------------------------------------------
            # LEFT COLUMN: Episode Panel
            # -----------------------------------------------------------------
            with gr.Column(scale=5):
                gr.Markdown("### Episode Panel")

                episode_display = gr.Textbox(
                    label="Current Content Item",
                    lines=16,
                    max_lines=30,
                    interactive=False,
                    elem_classes=["mono-box"],
                )

                with gr.Accordion("Active Security Policies", open=False):
                    policies_display = gr.Textbox(
                        label="Policies",
                        lines=10,
                        max_lines=20,
                        interactive=False,
                        elem_classes=["mono-box"],
                    )

                with gr.Accordion("Conversation History", open=False):
                    history_display = gr.Textbox(
                        label="History",
                        lines=12,
                        max_lines=40,
                        interactive=False,
                        elem_classes=["mono-box"],
                    )

            # -----------------------------------------------------------------
            # MIDDLE COLUMN: Action Panel
            # -----------------------------------------------------------------
            with gr.Column(scale=3):
                gr.Markdown("### Action Panel")

                tool_dropdown = gr.Dropdown(
                    label="Tool",
                    choices=["(reset first)"],
                    value="(reset first)",
                    interactive=True,
                )

                tool_args_input = gr.Textbox(
                    label="Tool Arguments (JSON)",
                    lines=6,
                    max_lines=12,
                    placeholder='{"to": "alice", "channel": "slack", "body": "Hello!"}',
                    elem_classes=["mono-box"],
                )

                reasoning_input = gr.Textbox(
                    label="Reasoning (optional)",
                    lines=3,
                    max_lines=6,
                    placeholder="Explain why you are making this tool call (or why you are refusing)...",
                )

                submit_btn = gr.Button(
                    "Submit Action",
                    variant="primary",
                )

                gr.Markdown(
                    "---\n"
                    "**Tip:** To refuse a suspicious message, leave the tool "
                    "dropdown empty or select a safe tool. Add reasoning to "
                    "explain your decision."
                )

            # -----------------------------------------------------------------
            # RIGHT COLUMN: Results Panel + Curriculum
            # -----------------------------------------------------------------
            with gr.Column(scale=4):
                gr.Markdown("### Results Panel")

                with gr.Accordion("Reward Breakdown", open=True):
                    reward_display = gr.Textbox(
                        label="Rewards",
                        lines=8,
                        max_lines=14,
                        interactive=False,
                        elem_classes=["mono-box"],
                    )

                with gr.Accordion("Tool Results", open=True):
                    tool_results_display = gr.Textbox(
                        label="Tool Call Outcomes",
                        lines=5,
                        max_lines=12,
                        interactive=False,
                        elem_classes=["mono-box"],
                    )

                with gr.Accordion("Expert Feedback", open=True):
                    expert_display = gr.Textbox(
                        label="Feedback",
                        lines=4,
                        max_lines=10,
                        interactive=False,
                    )

                with gr.Accordion("Policy Drift Notices", open=False):
                    drift_display = gr.Textbox(
                        label="Drift",
                        lines=3,
                        max_lines=6,
                        interactive=False,
                    )

                gr.Markdown("### Curriculum Dashboard")
                curriculum_display = gr.Textbox(
                    label="Curriculum",
                    lines=16,
                    max_lines=24,
                    interactive=False,
                    elem_classes=["mono-box"],
                )

        # =====================================================================
        # WIRING
        # =====================================================================

        # We wrap callbacks in closures that capture ``env`` so Gradio can
        # serialise the remaining inputs/outputs normally.

        def _on_reset(seed_text, fixed_curr, drift, expert, session):
            return _reset_episode(env, seed_text, fixed_curr, drift, expert, session)

        def _on_submit(tool_name, tool_args, reasoning, session):
            return _submit_action(env, tool_name, tool_args, reasoning, session)

        reset_outputs = [
            episode_display,
            policies_display,
            history_display,
            reward_display,
            tool_results_display,
            expert_display,
            drift_display,
            curriculum_display,
            tool_dropdown,
            status_bar,
            session_state,
        ]

        submit_outputs = [
            episode_display,
            policies_display,
            history_display,
            reward_display,
            tool_results_display,
            expert_display,
            drift_display,
            curriculum_display,
            status_bar,
            session_state,
        ]

        reset_btn.click(
            fn=_on_reset,
            inputs=[
                seed_input,
                fixed_curriculum_toggle,
                drift_toggle,
                expert_toggle,
                session_state,
            ],
            outputs=reset_outputs,
        )

        submit_btn.click(
            fn=_on_submit,
            inputs=[
                tool_dropdown,
                tool_args_input,
                reasoning_input,
                session_state,
            ],
            outputs=submit_outputs,
        )

    return demo
