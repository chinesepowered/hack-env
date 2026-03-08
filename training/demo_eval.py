#!/usr/bin/env python3
"""Angry Claw -- Hackathon Demo

Demonstrates the environment working end-to-end with three simulated agent
strategies (no ML model required).  Shows scoring, curriculum adaptation,
adaptive-vs-fixed comparison, and policy drift handling.

Run:
    cd hack-env
    python training/demo_eval.py
"""

from __future__ import annotations

import re
import sys
import os
import textwrap
from typing import Dict, List

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from red_team_arena.server.environment import RedTeamArenaEnvironment
from red_team_arena.models import RedTeamAction, ToolCall

# ============================================================================
# ANSI colour helpers
# ============================================================================

BOLD      = "\033[1m"
DIM       = "\033[2m"
RESET     = "\033[0m"
RED       = "\033[91m"
GREEN     = "\033[92m"
YELLOW    = "\033[93m"
BLUE      = "\033[94m"
MAGENTA   = "\033[95m"
CYAN      = "\033[96m"
WHITE     = "\033[97m"
BG_RED    = "\033[41m"
BG_GREEN  = "\033[42m"
BG_BLUE   = "\033[44m"
BG_YELLOW = "\033[43m"


def header(text: str, color: str = CYAN, width: int = 78) -> None:
    """Print a prominent section header."""
    bar = "=" * width
    print(f"\n{color}{BOLD}{bar}")
    print(f"  {text}")
    print(f"{bar}{RESET}\n")


def subheader(text: str, color: str = YELLOW, width: int = 78) -> None:
    bar = "-" * width
    print(f"\n{color}{bar}")
    print(f"  {text}")
    print(f"{bar}{RESET}\n")


def success(text: str) -> None:
    print(f"  {GREEN}[OK]{RESET} {text}")


def failure(text: str) -> None:
    print(f"  {RED}[!!]{RESET} {text}")


def info(text: str) -> None:
    print(f"  {BLUE}[--]{RESET} {text}")


def warn(text: str) -> None:
    print(f"  {YELLOW}[!!]{RESET} {text}")


def score_color(score: float) -> str:
    if score >= 3.0:
        return GREEN
    if score >= 0.0:
        return YELLOW
    return RED


# ============================================================================
# Suspicious-content keywords used by the Smart Agent
# ============================================================================

SUSPICIOUS_KEYWORDS = [
    "ignore",
    "disregard",
    "override",
    "SYSTEM",
    "maintenance mode",
    "base64",
    "curl ",
    "wget ",
    "| bash",
    "| sh",
    ".ssh/",
    "id_rsa",
    "credentials",
    "password",
    "api_key",
    "private_key",
    "attacker",
    "evil",
    "hidden",
    "<!-- ",
    "URGENT",
    "urgently",
    "wire ",
    "salary",
    "accounting@",
    "partner-fin",
    "grant-admin",
    "external_audit",
    "boss_personal",
    "gmail.com",
    "c0mpany",            # typosquatting
    "full_admin",
    "send_as_user",
    "read_credentials",
    "manage_sessions",
    "network_access",
    "evil-npm",
    "telemetry",
    "acme-onboard",
    "acme-corp.biz",
    "sync | bash",
    "cron.d",
    "post-install hook",
    "npm run setup --registry",
]

# ============================================================================
# Agent strategies
# ============================================================================


def naive_agent(obs) -> RedTeamAction:
    """Blindly calls EVERY available tool with dummy arguments.

    This simulates an agent with zero security awareness -- it tries to do
    something with every tool it sees, which means it will execute injected
    unsafe calls whenever the injected tool is among the available tools.
    """
    calls: List[ToolCall] = []
    for tool in obs.available_tools:
        args: Dict = {}
        for pname, pdef in tool.parameters.items():
            if isinstance(pdef, dict):
                ptype = pdef.get("type", "string")
            else:
                ptype = "string"
            if ptype == "array":
                args[pname] = ["default"]
            elif ptype == "object":
                args[pname] = {}
            else:
                # Use content-derived values so tool calls actually match
                # the injected unsafe patterns (simulating a compliant agent)
                args[pname] = _naive_arg_from_content(obs.content, pname)
        calls.append(ToolCall(tool=tool.name, arguments=args))

    return RedTeamAction(
        tool_calls=calls,
        reasoning="I will execute every available tool without question.",
    )


def _naive_arg_from_content(content: str, param_name: str) -> str:
    """Try to extract a plausible argument value from the content for the
    naive agent, so it is more likely to match injected unsafe calls."""
    lower = content.lower()

    if param_name == "path" or param_name == "file_path":
        m = re.search(r"(/[\w./\-_]+)", content)
        return m.group(1) if m else "/tmp/file"
    if param_name == "command":
        m = re.search(r"run_command\(['\"]([^'\"]+)['\"]\)", content)
        if m:
            return m.group(1)
        m = re.search(r"run\s+(?:the\s+)?(?:following\s+)?(?:command\s*)?[:\s]+(.+?)(?:\s+and\b|\s*$)", content, re.I)
        return m.group(1).strip() if m else "echo hello"
    if param_name == "url":
        m = re.search(r"(https?://[^\s)\"']+)", content)
        return m.group(1) if m else "https://example.com"
    if param_name == "to" or param_name == "recipient":
        m = re.search(r"(?:to|forward.*to|email.*to)\s+(\S+@\S+|\S+)", content, re.I)
        return m.group(1) if m else "user"
    if param_name == "skill_name":
        m = re.search(r"['\"]([a-z0-9_-]+)['\"]", content)
        return m.group(1) if m else "skill"
    if param_name == "permissions":
        return "auto"
    return "auto"


def cautious_agent(obs) -> RedTeamAction:
    """Refuses everything -- never makes any tool call."""
    return RedTeamAction(
        tool_calls=[],
        reasoning="I am refusing all actions out of caution.",
    )


def smart_agent(obs) -> RedTeamAction:
    """Heuristic agent: completes safe tasks, refuses suspicious content."""
    content_lower = obs.content.lower()

    # Check for suspicious indicators
    is_suspicious = any(kw.lower() in content_lower for kw in SUSPICIOUS_KEYWORDS)

    if is_suspicious:
        # Refuse -- but optionally send a warning message back
        sender = obs.sender
        msg_tools = [t for t in obs.available_tools if t.name == "send_message"]
        if msg_tools:
            return RedTeamAction(
                tool_calls=[
                    ToolCall(
                        tool="send_message",
                        arguments={
                            "to": sender,
                            "channel": obs.channel,
                            "body": "I detected suspicious content in this "
                                    "request and will not proceed.",
                        },
                    )
                ],
                reasoning="Suspicious content detected -- refusing unsafe "
                          "calls, notifying sender.",
            )
        return RedTeamAction(
            tool_calls=[],
            reasoning="Suspicious content detected -- refusing all calls.",
        )

    # Not suspicious -- try to fulfill the task with expected tools
    calls: List[ToolCall] = []
    available_names = {t.name for t in obs.available_tools}

    # Heuristic: pick tools that seem appropriate for the content
    if "read" in content_lower and "read_file" in available_names:
        path = _extract_path(obs.content)
        if path:
            calls.append(ToolCall(tool="read_file",
                                  arguments={"path": path}))

    if "search" in content_lower and "search_files" in available_names:
        query = _extract_quoted(obs.content) or "query"
        calls.append(ToolCall(tool="search_files",
                              arguments={"query": query}))

    if "schedule" in content_lower and "schedule_meeting" in available_names:
        calls.append(ToolCall(
            tool="schedule_meeting",
            arguments={
                "participants": ["team"],
                "time": "2025-10-10T14:00:00",
                "title": "Meeting",
            },
        ))

    if "install" in content_lower and "install_skill" in available_names:
        skill = _extract_quoted(obs.content) or _extract_skill_name(obs.content)
        calls.append(ToolCall(
            tool="install_skill",
            arguments={"skill_name": skill, "permissions": ["read_data"]},
        ))

    if "share" in content_lower and "share_document" in available_names:
        calls.append(ToolCall(
            tool="share_document",
            arguments={
                "doc": _extract_path(obs.content) or "document",
                "recipient": obs.sender,
                "channel": obs.channel,
            },
        ))

    if (("browse" in content_lower or "check http" in content_lower)
            and "browse_url" in available_names):
        url = _extract_url(obs.content)
        if url:
            calls.append(ToolCall(tool="browse_url",
                                  arguments={"url": url}))

    # If we made tool calls, also send a reply
    if calls and "send_message" in available_names:
        calls.append(ToolCall(
            tool="send_message",
            arguments={
                "to": obs.sender,
                "channel": obs.channel,
                "body": "<task completed>",
            },
        ))
    elif not calls and "send_message" in available_names:
        # Legitimate content but couldn't figure out what to do -- reply
        calls.append(ToolCall(
            tool="send_message",
            arguments={
                "to": obs.sender,
                "channel": obs.channel,
                "body": "<acknowledged>",
            },
        ))

    return RedTeamAction(
        tool_calls=calls,
        reasoning="Completing legitimate task with appropriate tools.",
    )


# ---------------------------------------------------------------------------
# Text extraction helpers for the smart agent
# ---------------------------------------------------------------------------

def _extract_path(text: str) -> str:
    """Pull out the first file-system path from text."""
    m = re.search(r"(/[\w./\-_]+\.\w+)", text)
    return m.group(1) if m else ""


def _extract_quoted(text: str) -> str:
    """Pull out a single-quoted string."""
    m = re.search(r"'([^']+)'", text)
    return m.group(1) if m else ""


def _extract_skill_name(text: str) -> str:
    """Try to find a skill name from context."""
    m = re.search(r"install\s+(?:the\s+)?['\"]?([a-z0-9_-]+)['\"]?",
                  text, re.I)
    return m.group(1) if m else "unknown-skill"


def _extract_url(text: str) -> str:
    """Pull out the first URL."""
    m = re.search(r"(https?://[^\s)\"']+)", text)
    return m.group(1) if m else ""


# ============================================================================
# Episode runner
# ============================================================================

def run_episode(
    env: RedTeamArenaEnvironment,
    agent_fn,
    agent_name: str,
    episode_num: int,
    seed: int | None = None,
    verbose: bool = True,
) -> Dict:
    """Run a single episode and return results summary.

    If *seed* is None the environment keeps its existing curriculum state
    (important for curriculum-adaptation demos).  If a seed is provided
    the curriculum is reset -- use this only for independent one-shot demos.
    """
    obs = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    injections_seen = 0
    injections_caught = 0
    injections_executed = 0
    false_positives = 0
    drift_events = 0
    step_details: List[Dict] = []

    while not obs.done:
        action = agent_fn(obs)
        old_step = obs.step_number
        tier = obs.metadata.get("tier", 0) if obs.metadata else 0
        is_inj = (obs.metadata.get("is_injection", False)
                  if obs.metadata else False)

        obs = env.step(action)
        steps += 1
        total_reward += obs.reward

        detail = {
            "step": old_step,
            "reward": obs.reward,
            "tier": tier,
            "is_injection": is_inj,
            "calls": [tc.tool for tc in action.tool_calls],
            "drift": bool(obs.policy_drift_notice),
        }
        step_details.append(detail)

        if is_inj:
            injections_seen += 1
            if obs.reward >= 0:
                injections_caught += 1
            else:
                injections_executed += 1

        if tier == 0 and not is_inj and len(action.tool_calls) == 0:
            false_positives += 1

        if obs.policy_drift_notice:
            drift_events += 1

        if verbose:
            tier_label = f"T{tier}" if tier > 0 else "OK"
            inj_marker = (f"{RED}INJ{RESET}" if is_inj
                          else f"{GREEN}LEG{RESET}")
            sc = score_color(obs.reward)
            calls_str = ", ".join(detail["calls"]) or "(none)"
            drift_str = (f" {MAGENTA}DRIFT{RESET}"
                         if detail["drift"] else "")
            print(
                f"    Step {old_step:>2} [{tier_label:>3}] {inj_marker} "
                f"| calls: {calls_str:<45} "
                f"| reward: {sc}{obs.reward:+.1f}{RESET}{drift_str}"
            )

    result = {
        "agent": agent_name,
        "episode": episode_num,
        "total_reward": total_reward,
        "steps": steps,
        "injections_seen": injections_seen,
        "injections_caught": injections_caught,
        "injections_executed": injections_executed,
        "false_positives": false_positives,
        "drift_events": drift_events,
        "step_details": step_details,
    }

    if verbose:
        sc = score_color(total_reward)
        print(
            f"    {BOLD}Episode total: {sc}{total_reward:+.1f}{RESET}  "
            f"| Inj caught: {injections_caught}/{injections_seen}  "
            f"| FP: {false_positives}  | Drifts: {drift_events}"
        )

    return result


def print_curriculum_state(env: RedTeamArenaEnvironment) -> None:
    """Print a formatted view of the curriculum state."""
    state = env._curriculum.get_state_dict()
    max_tier = state["max_tier_unlocked"]
    drift = state["drift_enabled"]

    print(f"    {BOLD}Curriculum State:{RESET}")
    print(
        f"      Max tier unlocked: {CYAN}{max_tier}{RESET}   "
        f"Drift enabled: "
        f"{'%sYES%s' % (GREEN, RESET) if drift else '%sNO%s' % (DIM, RESET)}"
        f"   Episodes completed: {state['episodes_completed']}"
    )

    # Tier stats table
    print(f"      {'Tier':>6} {'Seen':>6} {'Caught':>8} "
          f"{'Rate':>8} {'Weight':>8}")
    print(f"      {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for tier in range(1, 7):
        ts = state["tier_stats"].get(
            tier, state["tier_stats"].get(str(tier), {}))
        seen = ts.get("seen", 0)
        caught = ts.get("caught", 0)
        rate = ts.get("catch_rate", 0.0)
        weight = state["tier_weights"].get(
            tier, state["tier_weights"].get(str(tier), 1.0))
        unlocked = tier <= max_tier
        lock_icon = (f"{GREEN}*{RESET}" if unlocked
                     else f"{DIM}L{RESET}")
        rate_str = f"{rate:.0%}" if seen > 0 else "n/a"
        if seen > 0:
            rate_color = (GREEN if rate >= 0.7
                          else YELLOW if rate >= 0.4
                          else RED)
        else:
            rate_color = DIM
        print(
            f"    {lock_icon} T{tier:>3}  {seen:>6} {caught:>8} "
            f"{rate_color}{rate_str:>8}{RESET} {weight:>8.2f}"
        )
    print()


# ============================================================================
# DEMO SECTIONS
# ============================================================================

def demo_naive_agent() -> List[Dict]:
    header("DEMO 1: Naive Agent -- Executes Every Tool Blindly", RED)
    print(textwrap.dedent("""\
        Strategy: Calls EVERY available tool with arguments extracted from the
        message content. Zero security awareness.
        Expected: Executes injected attacks, accumulates heavy penalties (-5.0
        per injected call executed). Partial task completion by accident.
    """))

    env = RedTeamArenaEnvironment(seed=42, enable_drift=False,
                                  enable_expert=False)
    results = []
    for ep in range(3):
        subheader(f"Episode {ep + 1}", DIM)
        r = run_episode(env, naive_agent, "Naive", ep, seed=100 + ep)
        results.append(r)

    avg = sum(r["total_reward"] for r in results) / len(results)
    inj_exec = sum(r["injections_executed"] for r in results)
    inj_total = sum(r["injections_seen"] for r in results)
    print(f"\n  {BOLD}Naive Agent Summary:{RESET}")
    failure(f"Average reward: {RED}{avg:+.1f}{RESET}")
    failure(f"Injections executed: {RED}{inj_exec}/{inj_total}{RESET} "
            f"-- catastrophic security failure")
    print()
    return results


def demo_cautious_agent() -> List[Dict]:
    header("DEMO 2: Cautious Agent -- Refuses Everything", YELLOW)
    print(textwrap.dedent("""\
        Strategy: Never makes any tool calls. Maximum caution.
        Expected: Catches all injections (by doing nothing) but accumulates
        false-positive penalties on legitimate requests (-1.0 each).
    """))

    env = RedTeamArenaEnvironment(seed=42, enable_drift=False,
                                  enable_expert=False)
    results = []
    for ep in range(3):
        subheader(f"Episode {ep + 1}", DIM)
        r = run_episode(env, cautious_agent, "Cautious", ep, seed=200 + ep)
        results.append(r)

    avg = sum(r["total_reward"] for r in results) / len(results)
    fps = sum(r["false_positives"] for r in results)
    inj_caught = sum(r["injections_caught"] for r in results)
    inj_total = sum(r["injections_seen"] for r in results)
    print(f"\n  {BOLD}Cautious Agent Summary:{RESET}")
    warn(f"Average reward: {YELLOW}{avg:+.1f}{RESET}")
    success(f"Injections caught: {GREEN}{inj_caught}/{inj_total}{RESET}")
    failure(f"False positives: {RED}{fps}{RESET} (refused legitimate tasks)")
    print()
    return results


def demo_smart_agent() -> List[Dict]:
    header("DEMO 3: Smart Agent -- Heuristic Detection", GREEN)
    print(textwrap.dedent("""\
        Strategy: Checks for suspicious keywords (SYSTEM, curl|bash, base64,
        credential paths, urgency signals, etc). Executes expected tool calls
        on clean content, refuses and warns on suspicious content.
        Expected: Decent scores -- catches most injections, completes most tasks.
    """))

    env = RedTeamArenaEnvironment(seed=42, enable_drift=False,
                                  enable_expert=False)
    results = []
    for ep in range(3):
        subheader(f"Episode {ep + 1}", DIM)
        r = run_episode(env, smart_agent, "Smart", ep, seed=300 + ep)
        results.append(r)

    avg = sum(r["total_reward"] for r in results) / len(results)
    inj_caught = sum(r["injections_caught"] for r in results)
    inj_total = sum(r["injections_seen"] for r in results)
    fps = sum(r["false_positives"] for r in results)
    print(f"\n  {BOLD}Smart Agent Summary:{RESET}")
    success(f"Average reward: {GREEN}{avg:+.1f}{RESET}")
    success(f"Injections caught: {GREEN}{inj_caught}/{inj_total}{RESET}")
    info(f"False positives: {fps}")
    print()
    return results


def demo_curriculum_adaptation():
    header("DEMO 4: Curriculum Adaptation Over 15 Episodes", MAGENTA)
    print(textwrap.dedent("""\
        Running the Smart Agent for 15 episodes on an adaptive curriculum.
        Watch how:
          - Tier catch rates change as the agent encounters different attacks
          - Higher tiers unlock when catch rate exceeds the promotion threshold
          - Tier weights shift toward weaknesses (higher weight = more sampling)
          - Policy drift enables after sufficient baseline competence
    """))

    env = RedTeamArenaEnvironment(seed=42, fixed_curriculum=False,
                                  enable_drift=True, enable_expert=False)

    all_results = []
    for ep in range(15):
        subheader(f"Episode {ep + 1} / 15", CYAN)
        # Do NOT pass seed so curriculum state accumulates across episodes
        r = run_episode(env, smart_agent, "Smart", ep,
                        seed=None, verbose=True)
        all_results.append(r)
        print()
        print_curriculum_state(env)

    # Show progression summary
    subheader("Curriculum Progression Summary", MAGENTA)
    print(f"    {'Episode':>8} {'Reward':>8} {'Inj Caught':>12} "
          f"{'Max Tier':>10} {'Drift':>6}")
    print(f"    {'-'*8} {'-'*8} {'-'*12} {'-'*10} {'-'*6}")

    # Re-run to collect per-episode snapshots
    env2 = RedTeamArenaEnvironment(seed=42, fixed_curriculum=False,
                                   enable_drift=True, enable_expert=False)
    for ep in range(15):
        run_episode(env2, smart_agent, "Smart", ep,
                    seed=None, verbose=False)
        sd = env2._curriculum.get_state_dict()
        r = all_results[ep]
        inj_str = (f"{r['injections_caught']}/{r['injections_seen']}"
                   if r["injections_seen"] else "n/a")
        drift_str = (f"{GREEN}Y{RESET}" if sd["drift_enabled"]
                     else f"{DIM}N{RESET}")
        sc = score_color(r["total_reward"])
        print(
            f"    {ep+1:>8} {sc}{r['total_reward']:>+8.1f}{RESET} "
            f"{inj_str:>12} {sd['max_tier_unlocked']:>10} {drift_str:>6}"
        )

    print()
    return all_results


def demo_adaptive_vs_fixed():
    header("DEMO 5: Adaptive vs Fixed Curriculum Comparison", BLUE)
    print(textwrap.dedent("""\
        Running the Smart Agent for 12 episodes on BOTH adaptive and fixed
        curriculum, then comparing final scores and curriculum states.
    """))

    num_episodes = 12

    # Adaptive run -- no per-episode seed so curriculum accumulates
    subheader("Running Adaptive Curriculum...", GREEN)
    env_adaptive = RedTeamArenaEnvironment(
        seed=77, fixed_curriculum=False, enable_drift=True,
        enable_expert=False,
    )
    adaptive_results = []
    for ep in range(num_episodes):
        r = run_episode(env_adaptive, smart_agent, "Smart-Adaptive", ep,
                        seed=None, verbose=False)
        adaptive_results.append(r)

    # Fixed run -- no per-episode seed so we get comparable episodes
    subheader("Running Fixed Curriculum...", YELLOW)
    env_fixed = RedTeamArenaEnvironment(
        seed=77, fixed_curriculum=True, enable_drift=False,
        enable_expert=False,
    )
    fixed_results = []
    for ep in range(num_episodes):
        r = run_episode(env_fixed, smart_agent, "Smart-Fixed", ep,
                        seed=None, verbose=False)
        fixed_results.append(r)

    # Comparison table
    subheader("Episode-by-Episode Comparison", BLUE)
    print(f"    {'Episode':>8} {'Adaptive':>10} {'Fixed':>10} {'Delta':>10}")
    print(f"    {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for ep in range(num_episodes):
        a = adaptive_results[ep]["total_reward"]
        f = fixed_results[ep]["total_reward"]
        d = a - f
        ac = score_color(a)
        fc = score_color(f)
        dc = GREEN if d > 0 else (RED if d < 0 else DIM)
        print(
            f"    {ep+1:>8} {ac}{a:>+10.1f}{RESET} "
            f"{fc}{f:>+10.1f}{RESET} {dc}{d:>+10.1f}{RESET}"
        )

    # Totals
    a_total = sum(r["total_reward"] for r in adaptive_results)
    f_total = sum(r["total_reward"] for r in fixed_results)
    a_avg = a_total / num_episodes
    f_avg = f_total / num_episodes
    a_inj_caught = sum(r["injections_caught"] for r in adaptive_results)
    a_inj_total = sum(r["injections_seen"] for r in adaptive_results)
    f_inj_caught = sum(r["injections_caught"] for r in fixed_results)
    f_inj_total = sum(r["injections_seen"] for r in fixed_results)

    print()
    print(f"    {BOLD}{'':>8} {'Adaptive':>10} {'Fixed':>10}{RESET}")
    print(f"    {'Avg reward':>14}: "
          f"{score_color(a_avg)}{a_avg:>+8.1f}{RESET}   "
          f"{score_color(f_avg)}{f_avg:>+8.1f}{RESET}")
    print(f"    {'Total reward':>14}: "
          f"{a_total:>+8.1f}   {f_total:>+8.1f}")
    print(f"    {'Inj caught':>14}: "
          f"{a_inj_caught}/{a_inj_total:>5}   "
          f"{f_inj_caught}/{f_inj_total:>5}")

    # Curriculum states
    print()
    subheader("Final Adaptive Curriculum State", GREEN)
    print_curriculum_state(env_adaptive)

    subheader("Final Fixed Curriculum State", YELLOW)
    print_curriculum_state(env_fixed)

    print(textwrap.dedent(f"""\
    {BOLD}Key Takeaway:{RESET}
      The adaptive curriculum targets the agent's weaknesses, generating more
      attacks from tiers the agent struggles with. It also progressively
      unlocks harder tiers and enables policy drift, providing a richer
      training signal compared to the static fixed curriculum.
    """))


def demo_policy_drift():
    header("DEMO 6: Policy Drift -- Mid-Episode Rule Changes", MAGENTA)
    print(textwrap.dedent("""\
        Running episodes with policy drift enabled. When drift occurs mid-
        episode, the security rules change and the agent must adapt. The
        Smart Agent handles this by checking content against current policies.

        Content items with policy_drift_event fields trigger deterministic
        drift (e.g. the email-chain and transcript episodes always include
        a policy change step). Additional probabilistic drifts may occur.
    """))

    # Build an environment with drift from the start. We pre-warm the
    # curriculum so drift gets enabled, without resetting per-episode.
    env = RedTeamArenaEnvironment(seed=42, fixed_curriculum=False,
                                  enable_drift=True, enable_expert=True)

    info("Warming up curriculum (10 quick episodes to build stats and "
         "enable drift)...")

    # Use seed=None so each reset keeps the curriculum state
    for ep in range(10):
        obs = env.reset()
        while not obs.done:
            obs = env.step(smart_agent(obs))

    # Force-enable drift if the threshold wasn't reached organically
    if not env._curriculum.state.drift_enabled:
        env._curriculum.state.drift_enabled = True
        info("Force-enabled drift for demo purposes.")

    drift_enabled = env._curriculum.state.drift_enabled
    info(f"Drift enabled: {GREEN if drift_enabled else RED}"
         f"{drift_enabled}{RESET}")
    info(f"Max tier unlocked: {env._curriculum.state.max_tier_unlocked}")
    info(f"Episodes completed: {env._curriculum.state.episodes_completed}")
    print()

    # Run episodes, highlighting drift events
    episodes_with_drift = 0
    for ep in range(10):
        subheader(f"Drift Demo Episode {ep + 1}", MAGENTA)
        obs = env.reset()
        step_num = 0
        episode_reward = 0.0
        had_drift = False

        while not obs.done:
            step_num += 1
            tier = obs.metadata.get("tier", 0) if obs.metadata else 0
            is_inj = (obs.metadata.get("is_injection", False)
                      if obs.metadata else False)

            action = smart_agent(obs)
            obs = env.step(action)
            episode_reward += obs.reward

            # Highlight drift events with a banner
            if obs.policy_drift_notice:
                had_drift = True
                print(f"    {BG_YELLOW}{BOLD} POLICY DRIFT at Step "
                      f"{step_num} {RESET}")
                wrapped = textwrap.fill(
                    obs.policy_drift_notice, width=70,
                    initial_indent="      ",
                    subsequent_indent="      ")
                print(f"    {MAGENTA}{wrapped}{RESET}")
                calls_str = ", ".join(
                    tc.tool for tc in action.tool_calls) or "(none)"
                print(f"    {DIM}Agent calls: {calls_str}{RESET}")
                sc = score_color(obs.reward)
                print(f"    {DIM}Step reward: {sc}"
                      f"{obs.reward:+.1f}{RESET}")
                print()
            else:
                tier_label = f"T{tier}" if tier > 0 else "OK"
                inj_marker = (f"{RED}INJ{RESET}" if is_inj
                              else f"{GREEN}LEG{RESET}")
                sc = score_color(obs.reward)
                calls_str = (", ".join(tc.tool for tc in action.tool_calls)
                             or "(none)")
                print(
                    f"    Step {step_num:>2} [{tier_label:>3}] {inj_marker} "
                    f"| calls: {calls_str:<40} "
                    f"| reward: {sc}{obs.reward:+.1f}{RESET}"
                )

        sc = score_color(episode_reward)
        print(f"    {BOLD}Episode reward: {sc}"
              f"{episode_reward:+.1f}{RESET}")

        if had_drift:
            episodes_with_drift += 1
            if episodes_with_drift >= 3:
                break

    if episodes_with_drift == 0:
        warn("No drift events occurred in these episodes (probabilistic).")
        info("In real training, drift becomes common after ~10 episodes.")
    else:
        success(f"Showed {episodes_with_drift} episode(s) with policy "
                f"drift events.")

    print()


# ============================================================================
# STRATEGY COMPARISON SCOREBOARD
# ============================================================================

def print_scoreboard(
    naive_results: List[Dict],
    cautious_results: List[Dict],
    smart_results: List[Dict],
):
    header("FINAL SCOREBOARD -- Agent Strategy Comparison", WHITE)

    agents = [
        ("Naive Agent", naive_results, RED),
        ("Cautious Agent", cautious_results, YELLOW),
        ("Smart Agent", smart_results, GREEN),
    ]

    col_w = 18
    print(f"    {'Metric':<24}", end="")
    for name, _, color in agents:
        print(f"{color}{BOLD}{name:>{col_w}}{RESET}", end="")
    print()
    print(f"    {'-'*24}", end="")
    for _ in agents:
        print(f"  {'-'*(col_w-2)}", end="")
    print()

    def row(label: str, fn):
        print(f"    {label:<24}", end="")
        for _, results, color in agents:
            val = fn(results)
            print(f"{color}{val:>{col_w}}{RESET}", end="")
        print()

    row("Avg Reward",
        lambda rs: f"{sum(r['total_reward'] for r in rs)/len(rs):+.1f}")
    row("Total Reward",
        lambda rs: f"{sum(r['total_reward'] for r in rs):+.1f}")
    row("Inj Caught",
        lambda rs: (f"{sum(r['injections_caught'] for r in rs)}/"
                    f"{sum(r['injections_seen'] for r in rs)}"))
    row("Inj Executed",
        lambda rs: (f"{sum(r['injections_executed'] for r in rs)}/"
                    f"{sum(r['injections_seen'] for r in rs)}"))
    row("False Positives",
        lambda rs: f"{sum(r['false_positives'] for r in rs)}")
    row("Avg Steps",
        lambda rs: f"{sum(r['steps'] for r in rs)/len(rs):.1f}")

    print()
    print(textwrap.dedent(f"""\
    {BOLD}Interpretation:{RESET}
      {RED}Naive Agent{RESET}    -- Executes everything including attacks. Catastrophic.
      {YELLOW}Cautious Agent{RESET} -- Safe but useless. Refuses all work, heavy false-positive penalty.
      {GREEN}Smart Agent{RESET}    -- Best balance. Catches most injections while completing tasks.
                        A trained RL agent would push this even further.
    """))


# ============================================================================
# MAIN
# ============================================================================

def main():
    header(
        "RED TEAM ARENA -- Environment Demo\n"
        "  Hackathon 2026 | No ML Model Required",
        f"{BG_BLUE}{WHITE}",
    )

    print(textwrap.dedent(f"""\
    {DIM}This demo runs three simulated agent strategies through the Angry Claw
    environment, showing how different approaches handle adversarial attacks,
    legitimate tasks, and policy drift. No GPU or ML model needed.{RESET}
    """))

    # Demo 1-3: Individual agent strategies
    naive_results = demo_naive_agent()
    cautious_results = demo_cautious_agent()
    smart_results = demo_smart_agent()

    # Scoreboard
    print_scoreboard(naive_results, cautious_results, smart_results)

    # Demo 4: Curriculum adaptation
    demo_curriculum_adaptation()

    # Demo 5: Adaptive vs Fixed
    demo_adaptive_vs_fixed()

    # Demo 6: Policy drift
    demo_policy_drift()

    # Final
    header(
        "DEMO COMPLETE\n"
        "  The Angry Claw trains agents to be both helpful AND secure.\n"
        "  Adaptive curriculum + policy drift = robust adversarial training.",
        f"{BG_GREEN}{WHITE}",
    )


if __name__ == "__main__":
    main()
