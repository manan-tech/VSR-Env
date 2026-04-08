# VSR-Env Quick Start

**Get running in 3 steps.**

---

## ⚡ Prerequisites

- Python 3.10+
- Groq API key (free tier works)

---

## 🚀 3-Step Setup

### 1. Get Groq API Key (Free)

```bash
# Visit: https://console.groq.com/
# Sign up → API Keys → Create Key → Copy it
```

**Why Groq?** Fast inference, free tier available, OpenAI-compatible API.

### 2. Clone & Install

```bash
git clone https://github.com/manan-tech/VSR-Env
cd VSR-Env
pip install -e .
```

### 3. Run Baseline

```bash
export GROQ_API_KEY="gsk_your_key_here"
python inference.py
```

---

## ✅ Expected Output

```
[START] task=vol_regime_detection env=vsr_env model=llama-3.1-8b-instant
[STEP] step=1 action=hold(0,0,0.0) reward=0.80 done=true error=null
[END] success=true steps=1 score=0.80 rewards=0.80

[START] task=delta_hedging env=vsr_env model=llama-3.1-8b-instant
[STEP] step=1 action=sell(4,0,2.0) reward=0.72 done=false error=null
[STEP] step=2 action=sell(4,0,1.0) reward=0.68 done=false error=null
...
[END] success=true steps=5 score=0.75 rewards=0.72,0.68,0.88,0.52,0.65

...

===============================================
FINAL SUMMARY (ADAPTIVE CURRICULUM)
===============================================
  vol_regime_detection: 0.80
  delta_hedging: 0.75
  earnings_vol_crush: 0.68
  gamma_scalping: 0.62
  vega_gamma_stress: 0.71
  Average Completed: 0.71
===============================================
```

---

## 🎛️ Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | Required | API key for Groq LLM endpoint |
| `API_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | `llama-3.1-8b-instant` | Model to use for inference |

### Switching Models

```bash
# Faster model (lower quality)
export MODEL_NAME="llama-3.1-8b-instant"

# Larger model (higher quality)
export MODEL_NAME="llama-3.3-70b-versatile"

# Use Hugging Face endpoint
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
```

---

## 🐳 Docker Deployment

### Build and Run

```bash
docker build -t vsr-env:latest .
docker run -p 8000:8000 vsr-env:latest
```

### Access Server

```bash
# API docs
open http://localhost:8000/docs

# Web UI (if enabled)
open http://localhost:8000/web

# Health check
curl http://localhost:8000/health
```

---

## 🌐 Hugging Face Spaces

Already deployed at: **[MananBansal/VSR-Env](https://huggingface.co/spaces/MananBansal/VSR-Env)**

### Test Remote Endpoint

```python
from vsr_env.client import VSRClient

client = VSRClient(base_url="https://mananbansal-vsr-env.hf.space")
obs = client.reset(task_name="delta_hedging", seed=123)
print(obs.iv_surface)
```

---

## 🔧 Troubleshooting

### "GROQ_API_KEY not set"

```bash
# Check if set
echo $GROQ_API_KEY

# Set it
export GROQ_API_KEY="gsk_..."
```

### "Module not found: vsr_env"

```bash
# Install in development mode
pip install -e .
```

### "Rate limit exceeded"

```bash
# Groq free tier: ~30 requests/minute
# The inference script has built-in throttling (2s sleep between calls)
# If still hitting limits, increase sleep:

# Edit inference.py, line 489:
time.sleep(3)  # Increase from 2 to 3
```

### "Empty LLM response"

The script auto-retries up to 3 times. If persisting:
1. Check API key is valid
2. Try a different model (`llama-3.3-70b-versatile`)
3. Increase `MAX_TOKENS` in inference.py (line 64)

### "JSON decode error"

The script has robust parsing (handles truncated JSON, markdown extraction).
If still failing, check stderr output:
```bash
python inference.py 2>&1 | grep "LLM"
```

---

## 📊 Understanding Output

### [START] line
```
[START] task=delta_hedging env=vsr_env model=llama-3.1-8b-instant
```
- `task`: Which task is running
- `env`: Environment identifier
- `model`: LLM being used

### [STEP] line
```
[STEP] step=1 action=sell(4,0,2.0) reward=0.72 done=false error=null
```
- `step`: Current step number
- `action`: `direction(strike_idx, maturity_idx, quantity)`
- `reward`: Per-step reward [0.0, 1.0]
- `done`: Episode complete?
- `error`: Validation error (if any)

### [END] line
```
[END] success=true steps=5 score=0.75 rewards=0.72,0.68,0.88,0.52,0.65
```
- `success`: Did score ≥ threshold?
- `steps`: Total steps taken
- `score`: Final grader score [0.0, 1.0]
- `rewards`: Per-step reward sequence

---

## 🎯 Running Specific Tasks

### Single Task

```python
# Edit inference.py, line 764:
TASKS = ["delta_hedging"]  # Only run this task
```

### Custom Seed

```python
# Edit inference.py, line 60:
TASK_SEEDS = {
    "delta_hedging": 999,  # Custom seed
    # ...
}
```

### Adjust Max Steps

```python
# Edit inference.py, line 47:
MAX_STEPS_PER_TASK = {
    "delta_hedging": 10,  # Increase from 5
    # ...
}
```

---

## 🧪 Local Development Server

### Start Server

```bash
uvicorn vsr_env.server.app:app --reload --port 8000
```

### Test Endpoints

```bash
# Reset environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "delta_hedging", "seed": 123}'

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "selected_strike": 4,
    "selected_maturity": 0,
    "direction": "sell",
    "quantity": 2.0,
    "reasoning": "Testing"
  }'
```

---

## 📚 Next Steps

1. **Read the documentation**:
   - `TASKS.md` - 5-tier curriculum details
   - `REWARDS.md` - Grading formulas
   - `ARCHITECTURE.md` - System design

2. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

3. **Experiment with models**:
   ```bash
   export MODEL_NAME="llama-3.3-70b-versatile"
   python inference.py
   ```

4. **Deploy your own**:
   ```bash
   # Push to Hugging Face Spaces
   git clone https://huggingface.co/spaces/YOUR_USERNAME/VSR-Env
   cp -r ./* YOUR_USERNAME/VSR-Env/
   cd YOUR_USERNAME/VSR-Env
   git add . && git commit -m "Deploy" && git push
   ```

---

## 🆘 Need Help?

- **GitHub Issues**: [github.com/manan-tech/VSR-Env/issues](https://github.com/manan-tech/VSR-Env/issues)
- **Documentation**: See `*.md` files in repository root
- **Hugging Face Space**: [huggingface.co/spaces/MananBansal/VSR-Env](https://huggingface.co/spaces/MananBansal/VSR-Env)

---

**Ready to test the most challenging quantitative reasoning environment?**

```bash
git clone https://github.com/manan-tech/VSR-Env
cd VSR-Env
export GROQ_API_KEY="gsk_..."
python inference.py
```