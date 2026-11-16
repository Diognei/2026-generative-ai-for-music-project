import argparse
from pathlib import Path

import numpy as np

from audio_features import (
    extract_embeddings_with_progress,
    distribution,
    list_audio_files,
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


def calculate_kld(input_dir, reference_dir):
    input_files = list_audio_files(input_dir)
    reference_files = list_audio_files(reference_dir)
    print(f"Found {len(input_files)} input WAV file(s).", flush=True)
    print(f"Found {len(reference_files)} reference WAV file(s).", flush=True)

    reference_embeddings = np.concatenate(
        extract_embeddings_with_progress(
            reference_files,
            "Loading reference",
        ),
        axis=0,
    )
    input_embeddings = extract_embeddings_with_progress(
        input_files,
        "Loading input",
    )
    reference_distribution = distribution(reference_embeddings)

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
    parser.add_argument("reference_dir", type=Path)
    args = parser.parse_args()

    scores = calculate_kld(args.input_dir, args.reference_dir)
    print(f"KLD: {scores.mean():.6f} +- {scores.std():.6f}")

if __name__ == "__main__":
    main()
