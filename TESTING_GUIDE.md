# VSR-Env Local Testing Guide

## Quick Start

### 1. Get Groq API Key
1. Go to https://console.groq.com/
2. Sign up/login
3. Navigate to "API Keys"
4. Create a new API key
5. Copy the key (starts with `gsk_...`)

### 2. Set Environment Variables

```bash
# Required: Your Groq API key
export GROQ_API_KEY="gsk_your_actual_key_here"

# Optional: Override defaults
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
```

### 3. Install Dependencies

```bash
pip install -e .
```

### 4. Run Inference

```bash
python inference.py
```

## Configuration Details

### Default Configuration (Already Set)
- **Endpoint**: `https://api.groq.com/openai/v1`
- **Model**: `llama-3.3-70b-versatile` (fast and capable)
- **API Key**: Reads from `GROQ_API_KEY`, `HF_TOKEN`, or `API_KEY` env vars

### Available Groq Models
You can override the model by setting `MODEL_NAME`:

```bash
# Fast and capable (default)
export MODEL_NAME="llama-3.3-70b-versatile"

# Larger context window
export MODEL_NAME="llama-3.1-70b-versatile"

# Faster, smaller model
export MODEL_NAME="llama-3.1-8b-instant"

# Alternative: Mixtral
export MODEL_NAME="mixtral-8x7b-32768"
```

## Understanding the Output

### Output Format
```
[START] task=iv_reading env=vsr_env model=llama-3.3-70b-versatile
[STEP] step=1 action=sell(3,1,2.0) reward=0.50 done=false error=null
[STEP] step=2 action=buy(5,2,1.5) reward=0.30 done=false error=null
[STEP] step=3 action=hold(0,0,0.0) reward=0.00 done=true error=null
[END] success=true steps=3 score=0.75 rewards=0.50,0.30,0.00

[START] task=delta_hedging env=vsr_env model=llama-3.3-70b-versatile
...

============================================================
FINAL SUMMARY
============================================================
  iv_reading: 0.75
  delta_hedging: 0.42
  arb_capture: 0.31
  Average: 0.49
============================================================
```

### What Each Line Means

**[START]**: Beginning of a task
- `task`: Task name (iv_reading, delta_hedging, arb_capture)
- `env`: Environment name (vsr_env)
- `model`: LLM model being used

**[STEP]**: Each action taken
- `step`: Current step number
- `action`: Format is `direction(strike_idx,maturity_idx,quantity)`
  - Example: `sell(3,1,2.0)` = SELL strike index 3, maturity index 1, 2.0 contracts
- `reward`: Per-step reward (0.0-1.0)
- `done`: Whether episode is complete
- `error`: Validation error message or "null"

**[END]**: Task completion
- `success`: true if score >= 0.1
- `steps`: Number of steps taken
- `score`: Final grader score (0.0-1.0)
- `rewards`: Comma-separated list of all per-step rewards

### Score Interpretation

**Scores range from 0.0 to 1.0:**

**IV Reading Task** (Easy - 3 steps):
- 0.0: No correct identifications
- 0.5: 1 of 2 mispricings identified correctly
- 1.0: Both mispricings identified correctly

**Delta Hedging Task** (Medium - 5 steps):
- 0.0: No delta reduction
- 0.3-0.5: Partial delta reduction
- 0.7+: Good delta neutralization
- 1.0: Perfect neutralization with low cost

**Arbitrage Capture Task** (Hard - 8 steps):
- 0.0: No profit or poor risk management
- 0.2-0.4: Some profit with decent risk management
- 0.6+: Good profit with good risk management
- 1.0: Excellent profit with excellent risk management

## Troubleshooting

### Error: "GROQ_API_KEY not set"
```bash
# Check if it's set
echo $GROQ_API_KEY

# Set it
export GROQ_API_KEY="gsk_your_key_here"
```

### Error: "Module not found"
```bash
# Install the package
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Error: "Connection refused" or "API error"
- Check your internet connection
- Verify your API key is valid at https://console.groq.com/
- Check Groq API status: https://status.groq.com/

### Error: "Rate limit exceeded"
- Groq has rate limits on free tier
- Wait a few seconds and try again
- Consider upgrading your Groq account

### Low Scores (All tasks score 0.0)
This is normal! The tasks are challenging:
- The LLM needs to understand options trading
- It needs to parse the IV surface correctly
- It needs to make strategic decisions

**Tips to improve scores:**
- Try different models (llama-3.3-70b-versatile is recommended)
- Adjust temperature (lower = more deterministic)
- The system prompts are already optimized

## For Judges/Evaluators

The judges will run your submission by:

1. Setting their own environment variables:
```bash
export API_BASE_URL="their_endpoint"
export HF_TOKEN="their_api_key"
export MODEL_NAME="their_model"
```

2. Running:
```bash
python inference.py
```

3. Parsing the stdout for scores

Your code should work with any OpenAI-compatible API endpoint.

## Testing Without LLM (Sanity Check)

Test the environment directly without calling the LLM:

```python
from vsr_env.server.vsr_environment import VSREnvironment
from vsr_env.models import VSRAction, TradeDirection

# Create environment
env = VSREnvironment()

# Reset for IV reading task
obs = env.reset("iv_reading", seed=42)
print(f"Spot price: {obs.spot_price:.2f}")
print(f"IV surface shape: {len(obs.iv_surface)}x{len(obs.iv_surface[0])}")

# Take a test action
action = VSRAction(
    selected_strike=3,
    selected_maturity=1,
    direction=TradeDirection.SELL,
    quantity=2.0,
    reasoning="Test action"
)

result = env.step(action)
print(f"Reward: {result['reward'].total:.2f}")
print(f"Done: {result['done']}")
```

## Performance Notes

- Each task takes ~10-30 seconds with Groq (depending on model)
- Total runtime: ~1-3 minutes for all 3 tasks
- Groq is fast compared to other providers
- The environment itself is very fast (< 1ms per step)

## Next Steps

After local testing:
1. Ensure all 3 tasks complete without errors
2. Check that scores are reasonable (> 0.0)
3. Deploy to HuggingFace Spaces
4. Run `openenv validate` to verify compliance
5. Submit to the hackathon!
