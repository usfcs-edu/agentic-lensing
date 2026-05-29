---
name: reference-host-hardware
description: Multi-GPU host topology + which GPUs to use for which workload — Phase 3a only used 1 of the 2 L4s
metadata:
  type: reference
---

Agentic-lensing host has 10 GPUs: **8× NVIDIA A16 (16 GB each)** and **2× NVIDIA L4 (24 GB each)**.

The L4s are the most VRAM per GPU (24 GB) and the right default for
deep-learning workloads (PyTorch / JAX). Use **both L4s in parallel** by
launching two processes feeding disjoint slices of the work.

**GPU-index gotcha (verified Phase 4, 2026-05-29):** in `nvidia-smi` order
the two L4s are indices **8 and 9** (the eight A16s are 0–7). But CUDA's
default ordering is `FASTEST_FIRST`, which does NOT match `nvidia-smi`, so a
bare `CUDA_VISIBLE_DEVICES=8` can land on a (possibly busy) A16 — this caused
an OOM in Phase 4a. Always export **`CUDA_DEVICE_ORDER=PCI_BUS_ID`** so that
`CUDA_VISIBLE_DEVICES=8` / `=9` reliably select the two L4s. A16s float
between jobs, so check `nvidia-smi` for a free one before grabbing an A16.
See [[project-huang-2020-reproduction]] and [[project-huang-2021-reproduction]].

A16s (16 GB each) are useful for lighter-weight inference, embedding
extraction, or as an alternative for non-DL background work.

Related: [[reference-gigalens-env]] (which says "10 GPUs (8×A16, 2×L4)" —
this memory is the per-GPU-spec elaboration).
