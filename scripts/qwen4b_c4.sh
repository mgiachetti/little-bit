#!/bin/bash
# ~500M-token LittleBit-2 run on Qwen3-4B on one L40S 48 GB (same recipe as qwen_c4_1b.sh).
# Needs gradient checkpointing and 8-bit Adam to fit (~40 GB). Auto-resumes after crashes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/little-bit
for attempt in $(seq 1 20); do
  .venv/bin/python train.py --model Qwen/Qwen3-4B --init itq --bpw 0.55 --kv-rank-mult 1 \
    --precision tf32 --mse-norm --mse-skip-first 1 --grad-ckpt --optim adam8bit \
    --seq-len 2048 --batch-size 2 --lr 8e-5 --steps 122000 \
    --c4-mtokens 500 --wiki-repeat 4 \
    --eval-every 5000 --eval-batches 20 --ckpt-every 1000 --tag c4-500m \
    >> runs/qwen4b_c4.log 2>&1 && { echo DONE > runs/qwen4b_c4_done; exit 0; }
  echo "=== attempt $attempt failed $(date), restarting in 60s" >> runs/qwen4b_c4.log
  sleep 60
done
