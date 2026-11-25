import argparse
from pathlib import Path

import numpy as np

from cache_utils import (
    distribution,
    extract_embeddings_with_progress,
    list_audio_files,
    load_cached_reference_embeddings,
)


def kl_divergence(input_distribution, reference_distribution):
    input_distribution = np.clip(input_distribution, 1e-12, None)
    reference_distribution = np.clip(reference_distribution, 1e-12, None)
    return float(
        np.sum(
            input_distribution
            * np.log(input_distribution / reference_distribution)
        )
    )


def calculate_kld(input_dir):
    input_files = list_audio_files(input_dir)
    print(f"Found {len(input_files)} input WAV file(s).", flush=True)

    reference_embeddings = load_cached_reference_embeddings()
    reference_distribution = distribution(reference_embeddings)
    input_embeddings = extract_embeddings_with_progress(
        input_files,
        "Loading input",
    )

    scores = []
    for file_index, (path, embeddings) in enumerate(
        zip(input_files, input_embeddings),
        start=1,
    ):
        print(
            f"Calculating KLD {file_index}/{len(input_files)}: {path}",
            flush=True,
        )
        scores.append(
            kl_divergence(
                distribution(embeddings),
                reference_distribution,
            )
        )
    return np.asarray(scores, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate spectral Kullback-Leibler divergence between WAV directories."
    )
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()

    scores = calculate_kld(args.input_dir)
    print(f"KLD: {scores.mean():.6f} +- {scores.std():.6f}")

if __name__ == "__main__":
    main()
