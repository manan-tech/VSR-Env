"""
Baseline Inference Script for VSR-Env
======================================

This script demonstrates environment usage and establishes baseline scores.
It uses the OpenAI client to interact with the environment.

Environment Variables:
    API_BASE_URL   The API endpoint for the LLM
    HF_TOKEN       Your Hugging Face / API key
    MODEL_NAME     The model identifier to use for inference

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from vsr_env.models import TradeDirection, VSRAction, VSRObservation, VSRState

# Environment configuration
# Groq endpoint (judges will substitute this)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("GROQ_API_KEY")
# Model name for Groq (common models: llama-3.3-70b-versatile, mixtral-8x7b-32768)
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
BENCHMARK = "vsr_env"

# Task configurations (Requirements: 12.2, 12.6)
TASKS = ["iv_reading", "delta_hedging", "arb_capture"]
MAX_STEPS_PER_TASK = {
    "iv_reading": 3,
    "delta_hedging": 5,
    "arb_capture": 8,
}
TASK_SEEDS = {
    "iv_reading": 42,
    "delta_hedging": 123,
    "arb_capture": 456,
}

# LLM configuration
TEMPERATURE = 0.3
MAX_TOKENS = 800

# Success threshold
SUCCESS_SCORE_THRESHOLD = 0.1


# System prompts for each task (Requirement: 12.1)
SYSTEM_PROMPTS = {
    "iv_reading": textwrap.dedent(
        """
        You are an options trader analyzing an implied volatility surface.
        
        Your task is to identify 2 deliberately mispriced options on the IV surface.
        For each mispriced option, determine if it is overpriced or underpriced.
        
        The IV surface is an 8×3 matrix:
        - 8 strikes: [85, 90, 95, 97.5, 100, 102.5, 105, 110]
        - 3 maturities: [30, 90, 180] days
        
        If an option is OVERPRICED, respond with direction "sell" (you sell overpriced things).
        If an option is UNDERPRICED, respond with direction "buy" (you buy underpriced things).
        
        Respond ONLY with a valid JSON object (no markdown, no extra text):
        {"strike_idx": 0, "maturity_idx": 0, "direction": "sell", "quantity": 1.0, "reasoning": "your analysis citing specific IV values and spot price"}
        
        Focus on identifying IV values that deviate from the typical skew and term structure.
        """
    ).strip(),
    
    "delta_hedging": textwrap.dedent(
        """
        You are managing an options portfolio with non-zero delta exposure.
        
        Your objective is to neutralize the portfolio delta to within ±0.05
        while minimizing transaction costs.
        
        If portfolio delta is POSITIVE, SELL calls to reduce it.
        If portfolio delta is NEGATIVE, BUY calls to increase it.
        Choose strikes near ATM (indices 3-5) for best hedging efficiency.
        
        The option chain has:
        - 8 strikes: [85, 90, 95, 97.5, 100, 102.5, 105, 110] (indices 0-7)
        - 3 maturities: [30, 90, 180] days (indices 0-2)
        
        Respond ONLY with a valid JSON object (no markdown, no extra text):
        {"strike_idx": 4, "maturity_idx": 1, "direction": "sell", "quantity": 2.0, "reasoning": "Portfolio delta is 0.55, selling ATM call to reduce by ~0.5. Spot at 100."}
        """
    ).strip(),
    
    "arb_capture": textwrap.dedent(
        """
        You are an options arbitrage trader.
        
        Identify and exploit mispricings on the implied volatility surface.
        Execute trades to capture arbitrage profits while managing portfolio risk.
        Be prepared for regime shifts that may occur mid-episode.
        
        If you detect an OVERPRICED option (IV too high vs neighbors), SELL it.
        If you detect an UNDERPRICED option (IV too low vs neighbors), BUY it.
        Always hedge your delta exposure.
        
        The option chain has:
        - 8 strikes: [85, 90, 95, 97.5, 100, 102.5, 105, 110] (indices 0-7)
        - 3 maturities: [30, 90, 180] days (indices 0-2)
        
        Respond ONLY with a valid JSON object (no markdown, no extra text):
        {"strike_idx": 2, "maturity_idx": 0, "direction": "sell", "quantity": 2.0, "reasoning": "IV at strike 95/1M is 0.27 vs neighbors 0.23 - overpriced by ~4 vol points. Selling. Spot at 98.75, delta is 0.12."}
        """
    ).strip(),
}


def log_start(task: str, env: str, model: str) -> None:
    """Print [START] line to stdout.
    
    Requirements: 12.3
    """
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    """Print [STEP] line to stdout.
    
    Requirements: 12.4
    """
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Print [END] line to stdout.
    
    Requirements: 12.5
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _repair_truncated_json(text: str) -> Optional[str]:
    """Attempt to repair JSON truncated by max_tokens.
    
    Handles:
    - Truncated flat objects: {"strike_idx":4,...,"reasoning":"some long tex
    - Truncated arrays: [{"strike_idx":2,...,"quantity":
    - Truncated nested: {"option1":{"strike_idx":2,...
    """
    # Find the first { which starts an action-like object
    start = text.find('{')
    if start < 0:
        return None
    
    fragment = text[start:]
    
    # If it already ends with }, try as-is
    if fragment.rstrip().endswith('}'):
        return fragment
    
    # Count unmatched braces and check if we're inside a string
    in_string = False
    escape_next = False
    depth = 0
    
    for ch in fragment:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
    
    # Repair: close open string, then close open braces
    repair = fragment
    if in_string:
        repair += '"'
    repair += '}' * depth
    
    return repair


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM response to extract action parameters.
    
    Tries multiple strategies:
    1. Direct JSON parse
    2. Extract from markdown code blocks
    3. Find JSON object in text
    4. Repair truncated JSON (e.g. cut off by max_tokens)
    Returns safe default (hold action) on all failures.
    """
    default = {
        "strike_idx": 0,
        "maturity_idx": 0,
        "direction": "hold",
        "quantity": 0.0,
        "reasoning": "",
    }
    
    if not response_text:
        return default
    
    text = response_text.strip()
    
    # Strategy 1: Direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            # Check if it's a flat action dict (has strike_idx)
            if "strike_idx" in parsed:
                return {**default, **parsed}
            # It's a wrapper like {"option1": {...}, ...} or {"actions": [...]}
            # Extract the first nested dict that has strike_idx
            for val in parsed.values():
                if isinstance(val, dict) and "strike_idx" in val:
                    return {**default, **val}
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and "strike_idx" in item:
                            return {**default, **item}
        elif isinstance(parsed, list):
            # Array response like [{"strike_idx": ...}, ...]
            for item in parsed:
                if isinstance(item, dict) and "strike_idx" in item:
                    return {**default, **item}
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code blocks
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                return {**default, **parsed}
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Find complete JSON object in text
    json_pattern = r"\{[\s\S]*?\}"
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                return {**default, **parsed}
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Repair truncated JSON (max_tokens cut mid-sentence)
    repaired = _repair_truncated_json(text)
    if repaired:
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return {**default, **parsed}
        except json.JSONDecodeError:
            pass
    
    return default


def build_prompt(observation: VSRObservation, step: int) -> str:
    """Build user prompt from observation.
    
    Formats IV surface as table, includes spot price, portfolio Greeks,
    P&L, positions, market sentiment, and last error if present.
    
    Requirements: 12.7
    """
    # Format IV surface as table
    strikes = [85, 90, 95, 97.5, 100, 102.5, 105, 110]
    maturities = [30, 90, 180]
    
    iv_table = "IV Surface (strikes × maturities):\n"
    iv_table += "Strike\\Maturity | " + " | ".join(f"{m}d" for m in maturities) + "\n"
    iv_table += "-" * 50 + "\n"
    
    for i, strike in enumerate(strikes):
        row = f"{strike:>13} |"
        for j in range(3):
            iv_val = observation.iv_surface[i][j]
            row += f" {iv_val:.4f} |"
        iv_table += row + "\n"
    
    # Format portfolio Greeks
    greeks = observation.portfolio_greeks
    greeks_str = (
        f"Delta: {greeks.get('delta', 0.0):.4f}, "
        f"Gamma: {greeks.get('gamma', 0.0):.4f}, "
        f"Vega: {greeks.get('vega', 0.0):.4f}, "
        f"Theta: {greeks.get('theta', 0.0):.4f}"
    )
    
    # Format positions
    positions_str = "None"
    if observation.portfolio_positions:
        positions_str = "\n".join(
            f"  - {pos.get('direction', '?')} {pos.get('quantity', 0)} @ K={pos.get('strike', 0)}, T={pos.get('maturity', 0)}"
            for pos in observation.portfolio_positions
        )
    
    # Build prompt
    prompt = textwrap.dedent(
        f"""
        Step: {step} / {step + observation.steps_remaining}
        Task: {observation.task_name}
        
        {iv_table}
        
        Spot Price: {observation.spot_price:.2f}
        
        Portfolio Greeks: {greeks_str}
        Portfolio P&L: {observation.portfolio_pnl:.2f}
        
        Positions:
        {positions_str}
        
        Market Sentiment: {observation.market_sentiment:.2f}
        """
    ).strip()
    
    # Add last error if present
    if observation.last_action_error:
        prompt += f"\n\nLast Action Error: {observation.last_action_error}"
    
    prompt += "\n\nProvide your action as a JSON object."
    
    return prompt


def get_model_response(
    client: OpenAI, observation: VSRObservation, step: int, task_name: str
) -> Dict[str, Any]:
    """Get action from LLM model.
    
    Builds prompt, calls LLM, parses response.
    Retries up to 2 times on empty response (Groq rate-limit).
    Adds 2s delay between calls to stay under rate limits.
    """
    import time
    import sys

    system_prompt = SYSTEM_PROMPTS.get(task_name, SYSTEM_PROMPTS["iv_reading"])
    user_prompt = build_prompt(observation, step)

    max_retries = 3
    for attempt in range(max_retries):
        # Rate-limit delay (Groq free tier: ~30 req/min)
        time.sleep(2)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = (completion.choices[0].message.content or "").strip()

            if not response_text and attempt < max_retries - 1:
                print(f"  [LLM] Empty response, retrying ({attempt+1}/{max_retries})...", file=sys.stderr)
                time.sleep(3)  # Extra wait before retry
                continue

            print(f"  [LLM] Raw: {response_text[:200]}", file=sys.stderr)
            parsed = parse_llm_response(response_text)
            print(f"  [LLM] Parsed dir={parsed.get('direction')}, strike={parsed.get('strike_idx')}, qty={parsed.get('quantity')}", file=sys.stderr)
            return parsed

        except Exception as exc:
            print(f"  [LLM] ERROR (attempt {attempt+1}): {exc}", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return parse_llm_response("")

    return parse_llm_response("")


def create_action(parsed: Dict[str, Any]) -> VSRAction:
    """Create VSRAction from parsed LLM response.
    
    Maps direction string to TradeDirection enum and ensures types are correct.
    """
    # Map direction string to enum
    direction_str = str(parsed.get("direction") or "hold").lower()
    
    # Handle various direction formats
    if direction_str in ("buy", "overpriced", "underpriced"):
        # Note: iv_reading prompt uses overpriced/underpriced, but we map buy/sell specifically now.
        if direction_str == "underpriced":
            direction = TradeDirection.BUY
        elif direction_str == "overpriced":
            direction = TradeDirection.SELL
        else:
            direction = TradeDirection.BUY
    elif direction_str == "sell":
        direction = TradeDirection.SELL
    else:
        direction = TradeDirection.HOLD
    
    # Fallback to 0 if the LLM output 'null' for strike or maturity
    strike_idx = parsed.get("strike_idx")
    strike_idx = int(strike_idx) if strike_idx is not None else 0
    
    maturity_idx = parsed.get("maturity_idx")
    maturity_idx = int(maturity_idx) if maturity_idx is not None else 0
    
    quantity = parsed.get("quantity")
    quantity = float(quantity) if quantity is not None else 0.0

    return VSRAction(
        selected_strike=strike_idx,
        selected_maturity=maturity_idx,
        direction=direction,
        quantity=quantity,
        reasoning=str(parsed.get("reasoning") or ""),
    )


async def run_task(
    client: OpenAI, env: "VSREnvironment", task_name: str, seed: int
) -> float:
    """Run a single task episode.
    
    Resets environment with fixed seed, loops for max_steps,
    builds prompt, calls LLM, parses response, executes step,
    logs each step, extracts grader score on completion.
    
    Requirements: 12.2, 12.3, 12.4, 12.5
    """
    max_steps = MAX_STEPS_PER_TASK[task_name]
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Reset environment with fixed seed
        observation = env.reset(task_name=task_name, seed=seed)
        
        for step in range(1, max_steps + 1):
            # Get action from LLM
            parsed = get_model_response(client, observation, step, task_name)
            action = create_action(parsed)
            
            # Execute step
            result = env.step(action)
            observation = result["observation"]
            reward_val = float(result["reward"])
            done = result["done"]
            error = observation.last_action_error if hasattr(observation, 'last_action_error') else None
            
            rewards.append(reward_val)
            steps_taken = step
            
            # Format action string for logging
            action_str = f"{action.direction.value}({action.selected_strike},{action.selected_maturity},{action.quantity:.1f})"
            
            # Log step
            log_step(step=step, action=action_str, reward=reward_val, done=done, error=error)
            
            if done:
                break
        
        # Extract grader score from info
        info = result.get("info", {})
        score = info.get("grader_score", sum(rewards) / max(len(rewards), 1))
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        
    except Exception as e:
        # Catastrophic error: score is 0
        print(f"[DEBUG] Task error: {e}", flush=True)
        score = 0.0
    
    finally:
        # Always log end
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return score


async def main() -> None:
    """Main entry point.
    
    Initializes environment, runs all three tasks sequentially,
    prints final summary.
    
    Requirements: 12.2
    """
    # Initialize OpenAI client (Requirement: 12.1)
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Import environment here to avoid circular imports
    from vsr_env.server.vsr_environment import VSREnvironment
    
    # Initialize environment
    env = VSREnvironment()
    
    # Run all three tasks sequentially
    scores = {}
    
    for task_name in TASKS:
        seed = TASK_SEEDS[task_name]
        score = await run_task(client, env, task_name, seed)
        scores[task_name] = score
        print()  # Blank line between tasks
    
    # Print final summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for task_name, score in scores.items():
        print(f"  {task_name}: {score:.2f}")
    print(f"  Average: {sum(scores.values()) / len(scores):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())