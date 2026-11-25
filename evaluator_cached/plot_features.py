import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR_DIR = PROJECT_ROOT / "evaluator"
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from audio_features import (  # noqa: E402
    extract_embeddings_with_progress,
)


CACHE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_FILE = CACHE_DIR / "reference_embeddings.npz"
PLOTED_SAMPLES_PERCENT = 0.2


def load_embeddings(cache_file):
    cache_file = Path(cache_file)
    if not cache_file.is_file():
        raise FileNotFoundError(f"Feature cache not found: {cache_file}")

    with np.load(cache_file, allow_pickle=False) as cache:
        if "embeddings" not in cache:
            raise ValueError(f"Invalid feature cache: {cache_file}")
        embeddings = np.asarray(cache["embeddings"], dtype=np.float64)

    if embeddings.ndim != 2:
        raise ValueError(f"Invalid embeddings shape: {embeddings.shape}")
    if embeddings.shape[0] < 1:
        raise ValueError("Feature cache contains no embeddings.")
    if embeddings.shape[1] < 2:
        raise ValueError("At least two feature dimensions are required for plotting.")
    return embeddings


def sample_fraction(values, fraction, seed):
    values = np.asarray(values)
    if values.shape[0] == 0:
        return values

    sample_count = max(1, int(values.shape[0] * fraction))
    sample_count = min(sample_count, values.shape[0])
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(values.shape[0], size=sample_count, replace=False))
    return values[indices]


def list_wav_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"WAV directory not found: {directory}")

    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def extract_directory_embeddings(directory, label, max_points=None, seed=0):
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"WAV directory not found: {directory}")

    wav_files = list_wav_files(directory)
    if not wav_files:
        print(f"No WAV files found in {directory}; skipping.", flush=True)
        return np.empty((0, 64), dtype=np.float64)

    sampled_wav_files = sample_fraction(wav_files, PLOTED_SAMPLES_PERCENT, seed)
    print(
        f"Found {len(wav_files)} WAV file(s) in {directory}; "
        f"plotting {len(sampled_wav_files)}.",
        flush=True,
    )
    per_file_embeddings = extract_embeddings_with_progress(sampled_wav_files, label)
    embeddings = np.concatenate(per_file_embeddings, axis=0)

    if max_points is not None and embeddings.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(embeddings.shape[0], size=max_points, replace=False)
        embeddings = embeddings[indices]

    return embeddings


def reduce_with_pca(embeddings, max_points=None, seed=0):
    embeddings = np.asarray(embeddings, dtype=np.float64)

    if max_points is not None and embeddings.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(embeddings.shape[0], size=max_points, replace=False)
        embeddings = embeddings[indices]

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ components[:2].T
    return projected


def plot_features(cache_file, waves_dirs, output=None, max_points=None, seed=0):
    loaded_cache_embeddings = load_embeddings(cache_file)
    original_cache_count = loaded_cache_embeddings.shape[0]
    cache_embeddings = sample_fraction(
        loaded_cache_embeddings,
        PLOTED_SAMPLES_PERCENT,
        seed,
    )
    print(
        f"Reference cache: plotting {len(cache_embeddings)} of "
        f"{original_cache_count} embedding frames.",
        flush=True,
    )

    if max_points is not None and cache_embeddings.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(cache_embeddings.shape[0], size=max_points, replace=False)
        cache_embeddings = cache_embeddings[indices]

    groups = [("reference", cache_embeddings, "C0", "All Test Songs")]
    colors = ("C1", "C2", "C3")
    labels = ("Some Validation Songs", "Transformer Generated Songs", "LSTM Generated Songs")

    for directory_index, (directory, color, label) in enumerate(
        zip(waves_dirs, colors, labels),
        start=1,
    ):
        embeddings = extract_directory_embeddings(
            directory,
            f"Loading WAV directory {directory_index}",
            max_points=max_points,
            seed=seed,
        )
        if embeddings.shape[0] > 0:
            groups.append((str(directory), embeddings, color, label))

    feature_count = cache_embeddings.shape[1]
    if any(group_embeddings.shape[1] != feature_count for _, group_embeddings, _, _ in groups):
        raise ValueError("All embedding sets must have the same feature dimension.")

    combined = np.concatenate([group_embeddings for _, group_embeddings, _, _ in groups], axis=0)
    centered = combined - combined.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ components[:2].T

    fig, ax = plt.subplots(figsize=(8, 6))
    offset = 0
    for _, group_embeddings, color, label in groups:
        count = len(group_embeddings)
        group_points = projected[offset:offset + count]
        ax.scatter(
            group_points[:, 0],
            group_points[:, 1],
            s=8,
            alpha=0.55,
            color=color,
            label=label,
        )
        offset += count

    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("Reference and Generated Audio Features (PCA)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output}", flush=True)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot cached and WAV-extracted audio features with 2D PCA.",
    )
    parser.add_argument("cache", type=Path, help="Reference feature cache (.npz).")
    parser.add_argument("waves_dir_1", type=Path, help="First WAV directory.")
    parser.add_argument("waves_dir_2", type=Path, help="Second WAV directory.")
    parser.add_argument("waves_dir_3", type=Path, help="Third WAV directory.")
    parser.add_argument("--output", type=Path, help="Optional image path (for example: features.png).")
    parser.add_argument("--max-points", type=int, help="Maximum number of points per set.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    max_points = args.max_points if args.max_points and args.max_points > 0 else None
    plot_features(
        args.cache,
        (args.waves_dir_1, args.waves_dir_2, args.waves_dir_3),
        args.output,
        max_points,
        args.seed,
    )


if __name__ == "__main__":
    main()
