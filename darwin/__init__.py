"""DARWIN v2 — evolutionary LoRA-finetuning system.

See ARCHITECTURE.md (repo root) for the ground-truth design spec. This package holds
the controller and shared libraries; training/agent/eval work runs in Linux Docker
containers on remote GPUs (the controller itself stays cross-platform).
"""

__version__ = "2.0.0"
