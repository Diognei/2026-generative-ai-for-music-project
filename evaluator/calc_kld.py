import argparse
from pathlib import Path

import numpy as np

from audio_features import aggregate_embeddings, distribution, extract_embeddings, list_audio_files


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
    reference_distribution = distribution(aggregate_embeddings(reference_files))
    scores = [
        kl_divergence(
            distribution(extract_embeddings(path)),
            reference_distribution,
        )
        for path in input_files
    ]
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
