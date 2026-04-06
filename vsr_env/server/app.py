"""FastAPI server for VSR-Env.

Exposes the VSREnvironment via HTTP endpoints:
  POST /reset   — Start new episode
  POST /step    — Execute an action
  GET  /state   — Get current state
  GET  /health  — Health check
"""

import logging
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from vsr_env.models import VSRAction, VSRObservation, VSRState
from vsr_env.server.vsr_environment import VSREnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("vsr_env")

# Create app and environment
app = FastAPI(
    title="VSR-Env",
    description="Volatility Surface Reasoning Environment for options portfolio management",
    version="1.0.0",
)

env = VSREnvironment()


# === Request/Response models ===

class ResetRequest(BaseModel):
    task_name: str = "iv_reading"
    seed: int = 42


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


# === Endpoints ===

@app.get("/", response_class=RedirectResponse)
async def root():
    """Redirect root to the Web UI."""
    return RedirectResponse(url="/web")


@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """Serve the Custom Web UI dashboard."""
    ui_path = os.path.join(os.path.dirname(__file__), "index.html")
    try:
        with open(ui_path, "r") as f:
            html = f.read()
        return HTMLResponse(content=html, status_code=200)
    except Exception as e:
        logger.error(f"Failed to load UI: {e}")
        return HTMLResponse(content=f"Error loading UI: {e}", status_code=500)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "vsr_env", "version": "1.0.0"}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    """Reset the environment and start a new episode.

    Args:
        request: Optional reset parameters (task_name, seed)

    Returns:
        Initial observation
    """
    try:
        task_name = request.task_name if request else "iv_reading"
        seed = request.seed if request else 42

        logger.info(f"Resetting environment: task={task_name}, seed={seed}")
        observation = env.reset(task_name=task_name, seed=seed)

        return {"observation": observation.model_dump()}
    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(action: VSRAction):
    """Execute one step in the environment.

    Args:
        action: The agent's action (VSRAction)

    Returns:
        observation, reward, done, info
    """
    try:
        logger.info(
            f"Step: strike={action.selected_strike}, mat={action.selected_maturity}, "
            f"dir={action.direction.value}, qty={action.quantity}"
        )
        result = env.step(action)

        return {
            "observation": result["observation"].model_dump(),
            "reward": result["reward"],
            "done": result["done"],
            "info": result["info"],
        }
    except Exception as e:
        logger.error(f"Step failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Get current environment state (including hidden info)."""
    try:
        return {"state": env.state.model_dump()}
    except Exception as e:
        logger.error(f"State retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))