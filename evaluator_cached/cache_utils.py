import sys
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent
REFERENCE_CACHE_FILE = CACHE_DIR / "reference_embeddings.npz"
EVALUATOR_DIR = CACHE_DIR.parent / "evaluator"
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from audio_features import (  # noqa: E402
    distribution,
    extract_embeddings_with_progress,
    list_audio_files,
)


def load_cached_reference_embeddings(cache_file=REFERENCE_CACHE_FILE):
    cache_file = Path(cache_file)
    if not cache_file.is_file():
        raise FileNotFoundError(
            f"Reference feature cache not found: {cache_file}. "
            "Run extract_reference_features.py <reference_dir> first."
        )

    with np.load(cache_file, allow_pickle=False) as cache:
        if "embeddings" not in cache:
            raise ValueError(f"Invalid reference feature cache: {cache_file}")
        embeddings = np.asarray(cache["embeddings"], dtype=np.float64)

    if embeddings.ndim != 2:
        raise ValueError(f"Invalid reference embeddings shape: {embeddings.shape}")
    if embeddings.shape[0] < 1:
        raise ValueError("Reference feature cache contains no embeddings.")
    return embeddings
