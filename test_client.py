import asyncio
from vsr_env.client import LocalVSREnv
from vsr_env.models import VSRAction, TradeDirection


async def main():
    async with LocalVSREnv() as env:
        obs_res = await env.reset("delta_hedging", 42)
        print("Initial Observation generated")

        action = VSRAction(
            selected_strike=4,
            selected_maturity=0,
            direction=TradeDirection.BUY,
            quantity=1.0,
            reasoning="Testing reasoning component",
        )

        step_res = await env.step(action)
        print(f"Reward after step: {step_res.reward}")
        print(f"Expected outcome: {obs_res.observation.expected_outcome}")


asyncio.run(main())
