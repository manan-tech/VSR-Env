"""Client implementations for VSR-Env.

Provides HTTP/WS client (VSREnv) and an in-process client (LocalVSREnv)
for fast local testing.
"""

import asyncio
from typing import Optional
import httpx

from vsr_env.models import VSRAction, VSRObservation, VSRState
from vsr_env.server.vsr_environment import VSREnvironment


class SyncWrapper:
    """Wrapper to make async methods synchronous."""

    def __init__(self, async_client):
        self.async_client = async_client

    def __enter__(self):
        asyncio.get_event_loop().run_until_complete(self.async_client.__aenter__())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.get_event_loop().run_until_complete(
            self.async_client.__aexit__(exc_type, exc_val, exc_tb)
        )

    def reset(self, task_name: str = "delta_hedging", seed: int = 42):
        return asyncio.get_event_loop().run_until_complete(
            self.async_client.reset(task_name, seed)
        )

    def step(self, action: VSRAction):
        return asyncio.get_event_loop().run_until_complete(
            self.async_client.step(action)
        )

    def state(self):
        return asyncio.get_event_loop().run_until_complete(self.async_client.state())


class VSREnv:
    """Async environment client targeting a remote VSR-Env server."""

    def __init__(self, base_url: str):
        if not base_url.startswith("http"):
            base_url = f"http://{base_url}"
        self.base_url = base_url.rstrip("/")
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(base_url=self.base_url)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    def sync(self) -> SyncWrapper:
        """Returns a synchronous version of the client."""
        return SyncWrapper(self)

    async def reset(self, task_name: str = "delta_hedging", seed: int = 42):
        """Reset the remote environment."""
        if not self.client:
            raise RuntimeError("Client must be used as a context manager")

        class ResetResult:
            def __init__(self, obs_dict):
                self.observation = VSRObservation(**obs_dict)
                self.done = False
                self.reward = 0.0

        resp = await self.client.post(
            "/reset", json={"task_name": task_name, "seed": seed}
        )
        resp.raise_for_status()
        data = resp.json()
        return ResetResult(data["observation"])

    async def step(self, action: VSRAction):
        """Step the remote environment."""
        if not self.client:
            raise RuntimeError("Client must be used as a context manager")

        class StepResult:
            def __init__(self, data):
                self.observation = VSRObservation(**data["observation"])
                self.reward = data["reward"]
                self.done = data["done"]
                self.info = data.get("info", {})

        resp = await self.client.post("/step", json=action.model_dump(mode="json"))
        resp.raise_for_status()
        return StepResult(resp.json())

    async def state(self) -> VSRState:
        """Get the current state."""
        if not self.client:
            raise RuntimeError("Client must be used as a context manager")
        resp = await self.client.get("/state")
        resp.raise_for_status()
        return VSRState(**resp.json()["state"])


class LocalVSREnv:
    """In-process environment client matching the VSREnv interface.

    Useful for blazingly fast local testing by bypassing HTTP overhead.
    """

    def __init__(self, **kwargs):
        self.env = VSREnvironment()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def sync(self) -> SyncWrapper:
        return SyncWrapper(self)

    async def reset(self, task_name: str = "delta_hedging", seed: int = 42):
        obs = self.env.reset(task_name=task_name, seed=seed)

        class ResetResult:
            def __init__(self, observation):
                self.observation = observation
                self.done = False
                self.reward = 0.0

        return ResetResult(obs)

    async def step(self, action: VSRAction):
        result = self.env.step(action)

        class StepResult:
            def __init__(self, data):
                self.observation = data["observation"]
                self.reward = data["reward"]
                self.done = data["done"]
                self.info = data.get("info", {})

        return StepResult(result)

    async def state(self) -> VSRState:
        return self.env.state
