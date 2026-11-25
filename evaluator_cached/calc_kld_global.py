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
    input_distribution = np.asarray(input_distribution, dtype=np.float64)
    reference_distribution = np.asarray(reference_distribution, dtype=np.float64)

    if input_distribution.shape != reference_distribution.shape:
        raise ValueError("Input and reference distributions have different shapes.")

    input_distribution = np.clip(input_distribution, 1e-12, None)
    reference_distribution = np.clip(reference_distribution, 1e-12, None)

    input_distribution /= input_distribution.sum()
    reference_distribution /= reference_distribution.sum()

    return float(
        np.sum(
            input_distribution
            * np.log(input_distribution / reference_distribution)
        )
    )


def calculate_kld(input_dir):
    input_files = list_audio_files(input_dir)

    print(f"Found {len(input_files)} input WAV file(s).", flush=True)

    if not input_files:
        raise ValueError("No input WAV files found.")

    reference_embeddings = load_cached_reference_embeddings()

    input_embeddings = np.concatenate(
        extract_embeddings_with_progress(
            input_files,
            "Loading input",
        ),
        axis=0,
    )

    print("\nEmbedding statistics:", flush=True)
    print(f"Reference embeddings: {reference_embeddings.shape}", flush=True)
    print(f"Generated embeddings: {input_embeddings.shape}", flush=True)

    reference_distribution = distribution(reference_embeddings)
    input_distribution = distribution(input_embeddings)

    print("\nCalculating global KLD...", flush=True)

    return kl_divergence(
        input_distribution,
        reference_distribution,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Calculate global Kullback-Leibler divergence between WAV directories."
    )

    parser.add_argument("input_dir", type=Path)

    args = parser.parse_args()

    score = calculate_kld(args.input_dir)

    print("\n" + "=" * 40)
    print(f"Global KLD: {score:.6f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
