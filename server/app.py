"""
FastAPI application for the Meta Ad Optimizer Environment.

Exposes the AdOptimizerEnvironment over HTTP and WebSocket
endpoints using the standard OpenEnv ``create_app`` factory.

Usage:
    uvicorn meta_ad_optimizer.server.app:app --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import AdAction, AdObservation, AuctionAction, AuctionObservation
    from .ad_environment import AdOptimizerEnvironment
    from .arena_env import AdMarketArenaEnvironment
except ImportError:
    from models import AdAction, AdObservation, AuctionAction, AuctionObservation
    from server.ad_environment import AdOptimizerEnvironment
    from server.arena_env import AdMarketArenaEnvironment

# Original single-agent optimizer
app = create_app(
    AdOptimizerEnvironment,
    AdAction,
    AdObservation,
    env_name="meta_ad_optimizer",
    max_concurrent_envs=50,
)

# AdMarket Arena — multi-agent long-horizon auction (Plan 2)
_arena_app = create_app(
    AdMarketArenaEnvironment,
    AuctionAction,
    AuctionObservation,
    env_name="meta_ad_optimizer_arena",
    max_concurrent_envs=50,
)
app.mount("/arena", _arena_app)


@app.get("/")
def root():
    return {
        "name": "meta_ad_optimizer",
        "status": "running",
        "endpoints": ["/reset", "/step", "/state", "/health"],
        "tasks": ["creative_matcher", "placement_optimizer", "campaign_optimizer"],
        "arena_endpoints": ["/arena/reset", "/arena/step", "/arena/state", "/arena/health"],
        "arena_tasks": ["arena_easy", "arena_medium", "arena_hard"],
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for ``uv run server`` or ``python -m meta_ad_optimizer.server.app``."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
