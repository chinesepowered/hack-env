"""FastAPI application for the Red Team Arena environment."""

from __future__ import annotations

import inspect
import logging
import os

from openenv.core.env_server.http_server import create_app

try:
    from red_team_arena.models import RedTeamAction, RedTeamObservation
    from red_team_arena.server.environment import RedTeamArenaEnvironment
    from red_team_arena.server.gradio_ui import build_red_team_gradio_app
except ImportError:
    from models import RedTeamAction, RedTeamObservation
    from server.environment import RedTeamArenaEnvironment
    try:
        from server.gradio_ui import build_red_team_gradio_app
    except ImportError:
        build_red_team_gradio_app = None


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


# Create the FastAPI app with optional Gradio UI
_logger = logging.getLogger(__name__)
_sig = inspect.signature(create_app)

if build_red_team_gradio_app is not None and "gradio_builder" in _sig.parameters:
    app = create_app(
        create_red_team_environment,
        RedTeamAction,
        RedTeamObservation,
        env_name="red_team_arena",
        gradio_builder=build_red_team_gradio_app,
    )
else:
    if build_red_team_gradio_app is not None and "gradio_builder" not in _sig.parameters:
        _logger.warning(
            "Installed openenv-core does not support gradio_builder; "
            "Gradio UI will not be available."
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
