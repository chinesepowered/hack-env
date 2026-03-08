# Northflank H100 Training — Operations Guide

## Exec into the Container

```bash
nf exec --project <your-project-id> --service red-team
```

Or via the Northflank web console: **Services → red-team → Exec**.

---

## First-Time Setup (after a fresh container or reboot)

Persistent disk is mounted at `/persist/` — venv and repo survive reboots, but `/tmp/` and `/app/` do not.

### 1. Install system deps (if missing after reboot)
```bash
apt-get install -y git gcc
```

### 2. Clone repo to persistent disk (first time only)
```bash
git clone https://github.com/chinesepowered/hack-env.git /persist/repo
```

### 3. Create venv on persistent disk (first time only)
```bash
python -m venv /persist/trainvenv
/persist/trainvenv/bin/python -m ensurepip
```

### 4. Install dependencies
```bash
/persist/trainvenv/bin/python -m pip install unsloth trl datasets \
  "openenv-core[core]>=0.2.1" fastapi uvicorn requests torch
```

### 5. Install red_team_arena package
```bash
/persist/trainvenv/bin/python -m pip install -e /persist/repo/red_team_arena
```

### 6. Copy training script to /app
```bash
cp /persist/repo/training/train.py /app/training/train.py
```

---

## After a Reboot (container state reset but /persist/ survives)

```bash
apt-get install -y git gcc
cp /persist/repo/training/train.py /app/training/train.py
```

That's it — venv, repo, and model weights all persist.

---

## Start the Environment Server

The HTTP server is only needed if using `EnvHTTPClient`. Training uses `DirectEnvClient` (in-process) so the server is **not required** for training.

If you want it running anyway (e.g. for HF Spaces / API access):
```bash
cd /app/env && uvicorn red_team_arena.server.app:app --host 0.0.0.0 --port 8001 &
```

Check it's up:
```bash
curl http://localhost:8001/health
```

---

## Pull Latest Code + Run Training

```bash
cd /persist/repo && git pull && \
cp /persist/repo/training/train.py /app/training/train.py && \
cp /persist/repo/training/eval_model.py /app/training/eval_model.py && \
TORCHDYNAMO_DISABLE=1 /persist/trainvenv/bin/python /app/training/train.py \
  --model Qwen/Qwen3-4B \
  --env-url http://localhost:8001 \
  --output-dir /persist/output/red_team_arena
```

> `TORCHDYNAMO_DISABLE=1` is required — the container lacks gcc for Triton JIT compilation.

---

## Run Evaluation (Base vs Fine-Tuned)

```bash
cd /persist/repo && git pull && \
cp /persist/repo/training/eval_model.py /app/training/eval_model.py && \
TORCHDYNAMO_DISABLE=1 /persist/trainvenv/bin/python /app/training/eval_model.py \
  --model Qwen/Qwen3-4B \
  --lora-path /persist/output/red_team_arena/final \
  --episodes 10
```

Add `--quiet` to suppress per-step output and show summary only. Use `--episodes 5` for a quick check.

Outputs:
- Per-episode: reward, injection catch rate, false positives, per-step tool calls + reasoning
- Summary table: base vs fine-tuned side-by-side with delta

---

## Training Arguments

| Arg | Default | Notes |
|-----|---------|-------|
| `--model` | `Qwen/Qwen3-4B` | Text-only model required. Qwen3.5 is a VLM and won't work. |
| `--env-url` | `http://localhost:8001` | Ignored by DirectEnvClient but kept for compatibility |
| `--epochs` | `1` | |
| `--batch-size` | `1` | Per-device batch size |
| `--num-generations` | `4` | Must divide evenly into `batch-size * gradient-accum-steps` |
| `--max-seq-length` | `2048` | |
| `--lora-rank` | `16` | |
| `--lr` | `5e-6` | |
| `--dataset-size` | `64` | Number of prompts to generate from env |
| `--output-dir` | `./output/red_team_arena` | Use `/persist/output/...` so weights survive reboot |

---

## Upgrade Dependencies

```bash
/persist/trainvenv/bin/python -m pip install -U unsloth trl
```

---

## Check GPU

```bash
nvidia-smi
```

Expected: NVIDIA H100 80GB HBM3.

---

## Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| `bash: pip: command not found` | Use `python -m pip install ...` |
| `bash: git: command not found` | `apt-get install -y git gcc` |
| `CUDA assert: probability tensor contains inf/nan` | Use `loss_type="dr_grpo"` in GRPOConfig (already set) |
| `generation_batch_size not divisible by num_generations` | `gradient_accumulation_steps` must equal `num_generations` when `batch_size=1` |
| `model has no attribute warnings_issued` | Add `model.warnings_issued = {"estimate_tokens": True}` before GRPOTrainer init (already set) |
| Triton C compiler error | `TORCHDYNAMO_DISABLE=1` prefix (already in run command) |
| Qwen3.5 `compute_3d_position_ids` crash | Qwen3.5 is a VLM — use `Qwen/Qwen3-4B` instead |
| All rewards = 0.0 | Old HTTP client issue — `DirectEnvClient` runs env in-process, bypassing stateless HTTP |

---

## Output

LoRA adapters saved to `--output-dir/final/`. Copy off `/persist/` before container teardown:
```bash
ls /persist/output/red_team_arena/final/
```
