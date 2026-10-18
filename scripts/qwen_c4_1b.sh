#!/bin/bash
# ~1B-token LittleBit-2 run on Qwen3-0.6B (the papers' token budget) on one L40S 48 GB:
# TF32, batch 2 × 2048 tokens without gradient checkpointing, ~1.07 s/step. Auto-resumes after crashes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/little-bit
for attempt in $(seq 1 20); do
  .venv/bin/python train.py --model Qwen/Qwen3-0.6B --init itq --bpw 0.55 --kv-rank-mult 1 \
    --precision tf32 --mse-norm --mse-skip-first 1 \
    --seq-len 2048 --batch-size 2 --lr 8e-5 --steps 245000 \
    --c4-mtokens 1000 --wiki-repeat 4 \
    --eval-every 5000 --eval-batches 20 --ckpt-every 1000 --tag c4-1b \
    >> runs/qwen_c4_1b.log 2>&1 && { echo DONE > runs/qwen_c4_1b_done; exit 0; }
  echo "=== attempt $attempt failed $(date), restarting in 60s" >> runs/qwen_c4_1b.log
  sleep 60
done
