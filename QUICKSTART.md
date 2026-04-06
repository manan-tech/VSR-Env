# VSR-Env Quick Start

## 3-Step Setup

### 1. Get Groq API Key
```bash
# Visit: https://console.groq.com/
# Sign up → API Keys → Create Key → Copy it
```

### 2. Set API Key
```bash
export GROQ_API_KEY="gsk_your_key_here"
```

### 3. Run Test
```bash
pip install -e .
python inference.py
```

## That's it! 🎉

You should see output like:
```
[START] task=iv_reading env=vsr_env model=llama-3.3-70b-versatile
[STEP] step=1 action=sell(3,1,2.0) reward=0.50 done=false error=null
...
[END] success=true steps=3 score=0.75 rewards=0.50,0.30,0.00
```

## Configuration (Already Set)

✅ **Endpoint**: `https://api.groq.com/openai/v1`  
✅ **Model**: `llama-3.3-70b-versatile`  
✅ **API Key**: Reads from `GROQ_API_KEY` env var

## Try Different Models

```bash
# Faster model
export MODEL_NAME="llama-3.1-8b-instant"
python inference.py

# Larger context
export MODEL_NAME="llama-3.1-70b-versatile"
python inference.py
```

## Troubleshooting

**"GROQ_API_KEY not set"**
```bash
echo $GROQ_API_KEY  # Should show your key
export GROQ_API_KEY="gsk_..."
```

**"Module not found"**
```bash
pip install -e .
```

**Need help?** See `TESTING_GUIDE.md` for detailed documentation.
