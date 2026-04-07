# Architecture Diagram: VSR-Env

The VSR-Env consists of three core isolation layers: The **Environment Engine** (Pydantic state schemas, random walks), the **Task Handlers** (grading logic, step overrides), and the **Server API** (OpenEnv FastAPI specification).

## High-Level Flow

```text
[ LLM Agent (inference.py) ]
       |
       | (JSON Action + Reasoning over HTTP)
       v
[ FastAPI Server (server/app.py) ]
       |
       | (Deserializes into VSRAction schema)
       v
[ VSREnvironment (server/vsr_environment.py) ] ----> [ Telemetry Tracker (server/telemetry.py) ]
       |
       | (Dispatches to Current Task)
       v
[ Task Logic (vsr_env/tasks/*.py) ]
   - Modifies State (Spot, Time, IV Surface)
   - Triggers Task-Specific Events (Earnings Crunch, Macro Shocks)
   - Calls Grader (vsr_env/reward/reward_computer.py)
       |
       | (Returns Observation, Reward, Info)
       v
[ Returns to LLM Agent with Diagnostic Grader Analysis injected into Observation ]
```

## Core Components
1. **`VSRState` (models.py)**: The immutable state payload. Stores the portfolio map array, global step counts, accumulated PnL, cash reserves, and Greeks.
2. **`VSRObservation` (models.py)**: The 11-dimensional observation tensor dumped out as ASCII grid tables (for LLM ingestion natively).
3. **`VSREnvironment` (server/vsr_environment.py)**: The canonical orchestration facade compliant with standard Gymnasium syntax (`reset`, `step`).
4. **`inference.py`**: A hardened baseline loop containing "Adaptive Curriculum early stopping" and "Trajectory Diagnostics / Blotters" to teach the LLM in-context mid-episode.
