---
name: reference-host-hardware
description: Multi-GPU host topology + which GPUs to use for which workload — Phase 3a only used 1 of the 2 L4s
metadata:
  type: reference
---

Agentic-lensing host has 10 GPUs: **8× NVIDIA A16 (16 GB each)** and **2× NVIDIA L4 (24 GB each)**.

The L4s are the most VRAM per GPU (24 GB) and the right default for
deep-learning workloads (PyTorch / JAX). Phase 3a's training script
(`reproductions/huang-2020/05_train_resnet.py`) only used GPU 0 (one L4);
Phase 3b should use **both L4s in parallel** by launching two processes,
each with `CUDA_VISIBLE_DEVICES=0` and `=1`, feeding disjoint slices of
the parent sample. This roughly halves the wall-clock time for
download-bottlenecked sweeps. See [[project-huang-2020-reproduction]]
for the Phase 3a → 3b context.

A16s (16 GB each) are useful for lighter-weight inference, embedding
extraction, or as an alternative for non-DL background work.

Related: [[reference-gigalens-env]] (which says "10 GPUs (8×A16, 2×L4)" —
this memory is the per-GPU-spec elaboration).
