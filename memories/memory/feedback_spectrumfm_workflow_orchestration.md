---
name: feedback-spectrumfm-workflow-orchestration
description: For the SpectrumFM build-out Benson wants Workflow-tool multi-agent orchestration (build→verify), not solo execution
metadata:
  type: feedback
---
On the SpectrumFM Phase-I build-out (2026-06-02), when approving the roadmap plan Benson explicitly directed **"use workflows"** — orchestrate the substantive multi-step work via the Workflow tool (e.g. build → adversarial-verify fan-out), not solo single-agent execution.

**Why:** the build-out is large/multi-workstream and benefits from parallel implementation + adversarial verification of new eval harnesses *before* trusting go/no-go numbers.
**How to apply:** author a Workflow per workstream (implement → adversarially verify → synthesize fixes); keep GPU training/eval runs in the main loop (workflow agents do CPU/reasoning only, to avoid GPU contention). GPU availability varies — confirm what's free before launching: Benson sometimes reserves the **8×A16 for other work**, so the SpectrumFM default is the **2×L4** (PCI_BUS_ID indices 8,9). See [[project-spectrumfm-tooling]], [[reference-host-hardware]], and the roadmap in plans/ (fizzy-soaring-floyd.md).
