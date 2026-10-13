#!/bin/bash
# Long LittleBit-2 run on Qwen3-0.6B: ~110M tokens (C4 + 4x WikiText-2), auto-resumes after crashes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/little-bit
for attempt in $(seq 1 20); do
  .venv/bin/python train.py --model Qwen/Qwen3-0.6B --init itq --bpw 0.55 --kv-rank-mult 1 \
    --grad-ckpt --mse-norm --mse-skip-first 1 \
    --seq-len 2048 --batch-size 1 --grad-accum 2 --lr 8e-5 --steps 26000 \
    --c4-mtokens 100 --wiki-repeat 4 \
    --eval-every 1000 --eval-batches 20 --ckpt-every 250 --tag c4 \
    >> runs/qwen_c4.log 2>&1 && { echo DONE > runs/qwen_c4_done; exit 0; }
  echo "=== attempt $attempt failed $(date), restarting in 60s" >> runs/qwen_c4.log
  sleep 60
done
