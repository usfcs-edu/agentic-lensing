"""LensJudge — agentic strong-lens candidate VI grader on the Claude Agent SDK.

Automates the human visual-inspection grading (A/B/C/D) of strong-lens candidates
from the Huang group's ResNet/EfficientNet finders. Named LensJudge (the onboarding
report's "LensAgent"/"VI Pre-grading Agent") to avoid a clash with an existing arXiv
paper. Staged: lean single-agent grader (baseline) -> CNN-gated judge panel ->
spectroscopic pre-grader, all scored in a shared LensBench-VI harness.

Run modules with the repo's ``reproductions/`` dir on ``sys.path``; entry scripts
self-bootstrap that path, so ``python lensjudge/imaging/run_batch.py`` works directly.
"""

__version__ = "0.1.0"
