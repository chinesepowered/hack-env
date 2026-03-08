#!/bin/bash
set -e

echo "=== Angry Claw H100 Training Setup ==="

# Install training dependencies
echo "Installing training deps..."
pip install unsloth trl datasets 2>&1 | tail -5

# Start the environment server in background
echo "Starting environment server..."
cd /app/env
python -m uvicorn server.app:app --host 0.0.0.0 --port 8001 &
sleep 5

# Verify server
echo "Checking server health..."
curl -s http://localhost:8001/health || echo "Server health check failed"

# Run training
echo "Starting GRPO training on H100..."
cd /app
python training/train.py \
  --model Qwen/Qwen3.5-9B \
  --env-url http://localhost:8001 \
  --epochs 1 \
  --batch-size 4 \
  --output-dir /app/output/red_team_arena

echo "Training complete!"
