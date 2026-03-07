"""FastAPI application for the Red Team Arena environment."""

from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

try:
    from red_team_arena.models import RedTeamAction, RedTeamObservation
    from red_team_arena.server.environment import RedTeamArenaEnvironment
except ImportError:
    from models import RedTeamAction, RedTeamObservation
    from server.environment import RedTeamArenaEnvironment


# Configuration via environment variables
SEED = int(os.getenv("RED_TEAM_SEED", "42"))
FIXED_CURRICULUM = os.getenv("RED_TEAM_FIXED_CURRICULUM", "0") in {"1", "true", "True"}
ENABLE_DRIFT = os.getenv("RED_TEAM_ENABLE_DRIFT", "1") in {"1", "true", "True"}
ENABLE_EXPERT = os.getenv("RED_TEAM_ENABLE_EXPERT", "1") in {"1", "true", "True"}


def create_red_team_environment():
    """Factory function for creating environment instances."""
    return RedTeamArenaEnvironment(
        seed=SEED,
        fixed_curriculum=FIXED_CURRICULUM,
        enable_drift=ENABLE_DRIFT,
        enable_expert=ENABLE_EXPERT,
    )


app = create_app(
    create_red_team_environment,
    RedTeamAction,
    RedTeamObservation,
    env_name="red_team_arena",
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
