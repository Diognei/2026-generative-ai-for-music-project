import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR_DIR = PROJECT_ROOT / "evaluator"
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from audio_features import (  # noqa: E402
    extract_embeddings_with_progress,
    list_audio_files,
)

CACHE_DIR = Path(__file__).resolve().parent
CACHE_FILE = CACHE_DIR / "reference_embeddings.npz"


def extract_reference_features(reference_dir, cache_file=CACHE_FILE):
    reference_files = list_audio_files(reference_dir)
    print(f"Found {len(reference_files)} reference WAV file(s).", flush=True)

    embeddings = np.concatenate(
        extract_embeddings_with_progress(reference_files, "Loading reference"),
        axis=0,
    )

    cache_file = Path(cache_file)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        embeddings=embeddings,
        reference_dir=str(Path(reference_dir).resolve()),
    )
    print(f"Reference embeddings saved to: {cache_file}", flush=True)
    print(f"Cache shape: {embeddings.shape}", flush=True)
    return embeddings, cache_file


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache spectral embeddings from a reference WAV directory.",
    )
    parser.add_argument("reference_dir", type=Path, help="Reference WAV files.")
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=CACHE_FILE,
        help=f"Optional cache file (default: {CACHE_FILE}).",
    )
    args = parser.parse_args()

    extract_reference_features(args.reference_dir, args.cache_file)


if __name__ == "__main__":
    main()
