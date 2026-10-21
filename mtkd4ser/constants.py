"""
constants.py – single source of truth for labels and language ids.
"""

LABELS = ["anger", "happiness", "neutral", "sadness"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# language_id values stored in the prepared datasets (1 was reserved for Finnish/FESC).
LANG_IDS = {"en": 0, "fr": 2}
LANG_ID_TO_KEY = {v: k for k, v in LANG_IDS.items()}

SAMPLING_RATE = 16_000
