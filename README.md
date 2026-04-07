---
title: VSR-Env
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# VSR-Env: Volatility Surface Reasoning Environment

VSR-Env is a reinforcement learning benchmark for options portfolio management and derivatives reasoning, built for the Meta PyTorch OpenEnv Hackathon.

## Overview
Unlike standard numerical optimization environments, VSR-Env acts as a multi-turn logical benchmark. The LLM must observe non-stationary Implied Volatility (IV) surfaces, evaluate complex Greeks (Delta, Gamma, Vega), and make discrete portfolio allocation decisions over multi-step episodes.

## Features Let's
- **5-Tier Adaptive Curriculum**: Ranges from "Easy" Volatility Regime Detection to "Super-Boss" Tier Vega/Gamma Stress tests.
- **Continuous State Formulation**: Underlying models drift using standard Brownian motion but incorporate sudden volatility shocks or earnings crushes.
- **Strict Grading Heuristics**: Adheres to institutional Greek hedging mathematics, utilizing Gaussian boundaries and standard deviation matrices.
- **Diagnostic Telemetry**: Automatically records the complete trajectory history (observations, actions, rewards, reasoning) inside an immutable internal blotter.

## Getting Started

### Installation
Ensure you have `uv` installed, then run:

```bash
uv sync
```

Alternatively via plain pip:
```bash
pip install -e .
```

### Running the Inference Baseline
To run the full 5-tier benchmark evaluating an LLM via the Groq OpenAI compatibility layer:
```bash
export GROQ_API_KEY="your_groq_key"
export MODEL_NAME="llama-3.1-8b-instant"
uv run inference.py
```

### Multi-Mode Server (OpenEnv)
This environment is compliant with the OpenEnv structural guidelines and includes a native FastAPI deployment layer:
```bash
uvicorn vsr_env.server.app:app --host 0.0.0.0 --port 8000
```
Then visit `http://localhost:8000/web` for the manual human-play UI interface.

## Documentation Reference
- `ARCHITECTURE.md` - Core system diagrams and component breakdowns.
- `TASKS.md` - Detailed breakdown of the 5-tiered difficulty progression tasks.
- `REWARDS.md` - Explanations of our deterministic v2 and heuristics reward systems.

## Submission Readiness
This repository has been hardened and validates fully under `openenv validate`. The `Dockerfile` natively builds the entire `uvicorn` stack, ensuring compatibility when pushing to Hugging Face Spaces.
