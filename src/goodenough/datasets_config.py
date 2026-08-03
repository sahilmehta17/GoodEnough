"""
Dataset and split policy. Frozen constants that the freeze script and the
runner both read, so the definition of the experiment lives in one place.

See PREREGISTRATION.md section 11 for the rationale, especially why dev is
drawn from MMLU's validation split rather than carved out of test.
"""

from __future__ import annotations

SEED = 42

# The two primary slices carry the headline claim. Named before the pilot.
MMLU_PRIMARY = ["high_school_geography", "formal_logic"]

# Six exploratory slices, reported with unadjusted intervals.
MMLU_EXPLORATORY = [
    "nutrition",
    "marketing",
    "miscellaneous",
    "college_mathematics",
    "professional_law",
    "high_school_psychology",
]

MMLU_SUBJECTS = MMLU_PRIMARY + MMLU_EXPLORATORY

# Split sizing (PREREGISTRATION.md section 11).
MMLU_MAP_PER_SUBJECT = 100        # capped at available test items
MMLU_ROUTER_PER_SUBJECT = 20      # held out from test after the map block
MMLU_ROUTER_TOTAL_CAP = 150       # pooled router split, capped

GSM8K_DEV_N = 50
GSM8K_MAP_N = 150

# Hugging Face sources. Revisions are recorded into the manifest at freeze time
# so the exact snapshot is reproducible even if the datasets change upstream.
MMLU_HF_PATH = "cais/mmlu"
GSM8K_HF_PATH = "openai/gsm8k"
GSM8K_HF_CONFIG = "main"

# The letters MMLU uses for its four options.
MMLU_CHOICES = ["A", "B", "C", "D"]
