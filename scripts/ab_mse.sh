#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--model Qwen/Qwen3-0.6B --init itq --bpw 0.55 --kv-rank-mult 1 --grad-ckpt --batch-size 1 --grad-accum 4 --lr 8e-5 --steps 300 --eval-every 1000 --eval-batches 20 --out runs/ab"
.venv/bin/python train.py $COMMON > runs/ab_paper.log 2>&1
.venv/bin/python train.py $COMMON --mse-norm > runs/ab_msenorm.log 2>&1
echo DONE > runs/ab_done
