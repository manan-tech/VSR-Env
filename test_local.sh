#!/bin/bash
# Quick test script for VSR-Env with Groq

echo "=========================================="
echo "VSR-Env Local Testing with Groq"
echo "=========================================="
echo ""

# Check if GROQ_API_KEY is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ ERROR: GROQ_API_KEY not set!"
    echo ""
    echo "Please set your Groq API key:"
    echo "  export GROQ_API_KEY='gsk_your_key_here'"
    echo ""
    echo "Get your key at: https://console.groq.com/"
    exit 1
fi

echo "✓ GROQ_API_KEY is set"
echo ""

# Show configuration
echo "Configuration:"
echo "  API Endpoint: ${API_BASE_URL:-https://api.groq.com/openai/v1}"
echo "  Model: ${MODEL_NAME:-llama-3.3-70b-versatile}"
echo ""

# Test the environment directly (without LLM)
echo "Testing environment (without LLM)..."
python -c "
from vsr_env.server.vsr_environment import VSREnvironment
from vsr_env.models import VSRAction, TradeDirection

env = VSREnvironment()
obs = env.reset('iv_reading', 42)
print('✓ Environment reset successful')
print(f'  Spot price: {obs.spot_price:.2f}')
print(f'  IV surface shape: {len(obs.iv_surface)}x{len(obs.iv_surface[0])}')

# Test a step
action = VSRAction(
    selected_strike=3,
    selected_maturity=1,
    direction=TradeDirection.SELL,
    quantity=2.0,
    reasoning='Test'
)
result = env.step(action)
print(f'✓ Step executed successfully')
print(f'  Reward: {result[\"reward\"].total:.2f}')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Environment test failed!"
    echo "Try: pip install -e ."
    exit 1
fi

echo ""
echo "=========================================="
echo "Running inference with Groq LLM..."
echo "=========================================="
echo ""
echo "This will take 1-3 minutes..."
echo ""

# Run inference
python inference.py

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
