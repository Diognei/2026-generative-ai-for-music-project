
import argparse
from pathlib import Path

import numpy as np

from audio_features import (
    extract_embeddings_with_progress,
    list_audio_files,
)


def covariance(embeddings):
    if embeddings.shape[0] < 2:
        raise ValueError(
            "At least two embeddings are required to calculate covariance."
        )
    return np.cov(embeddings, rowvar=False)


def matrix_sqrt(matrix):
    matrix = (matrix + matrix.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def frechet_distance(input_embeddings, reference_embeddings):
    input_mean = input_embeddings.mean(axis=0)
    reference_mean = reference_embeddings.mean(axis=0)

    input_covariance = covariance(input_embeddings)
    reference_covariance = covariance(reference_embeddings)

    covariance_root = matrix_sqrt(input_covariance)

    middle = covariance_root @ reference_covariance @ covariance_root
    covariance_product_root = matrix_sqrt(middle)

    mean_distance = np.sum((input_mean - reference_mean) ** 2)

    covariance_distance = (
        np.trace(input_covariance)
        + np.trace(reference_covariance)
        - 2.0 * np.trace(covariance_product_root)
    )

    return float(max(0.0, mean_distance + covariance_distance))


def calculate_fad(input_dir, reference_dir):
    input_files = list_audio_files(input_dir)
    reference_files = list_audio_files(reference_dir)

    print(f"Found {len(input_files)} input WAV file(s).", flush=True)
    print(f"Found {len(reference_files)} reference WAV file(s).", flush=True)

    if not input_files:
        raise ValueError("No input WAV files found.")

    if not reference_files:
        raise ValueError("No reference WAV files found.")

    # Extract all reference embeddings
    reference_embeddings = np.concatenate(
        extract_embeddings_with_progress(
            reference_files,
            "Loading reference",
        ),
        axis=0,
    )

    # Extract all generated embeddings
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

    if reference_embeddings.shape[1] != input_embeddings.shape[1]:
        raise ValueError("Reference and input embedding dimensions differ.")

    if reference_embeddings.shape[0] < 2 or input_embeddings.shape[0] < 2:
        raise ValueError("At least two embeddings per set are required.")

    # Calculate one global FAD
    print("\nCalculating global FAD...", flush=True)

    score = frechet_distance(
        input_embeddings,
        reference_embeddings,
    )

    return score


def main():
    parser = argparse.ArgumentParser(
        description="Calculate global Fréchet Audio Distance between WAV directories."
    )

    parser.add_argument("input_dir", type=Path)
    parser.add_argument("reference_dir", type=Path)

    args = parser.parse_args()

    score = calculate_fad(args.input_dir, args.reference_dir)

    print("\n" + "=" * 40)
    print(f"Global FAD: {score:.6f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
