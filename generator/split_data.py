import argparse
import random
import shutil
from pathlib import Path


train_percent = 0.8
valid_percent = 0.1
test_percent = 0.1
RANDOM_SEED = 42


def find_midi_files(input_dir, output_dir):
    """Find MIDI files recursively, excluding an output directory inside input."""

    midi_files = []
    output_dir = output_dir.resolve()

    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".mid", ".midi"}:
            continue
        if output_dir == path or output_dir in path.parents:
            continue
        midi_files.append(path)

    return sorted(midi_files)


def split_files(midi_files):
    files = list(midi_files)
    random.Random(RANDOM_SEED).shuffle(files)

    train_count = int(len(files) * train_percent)
    valid_count = int(len(files) * valid_percent)

    train_files = files[:train_count]
    valid_files = files[train_count:train_count + valid_count]
    test_files = files[train_count + valid_count:]
    return train_files, valid_files, test_files


def copy_files(files, split_name, input_dir, output_dir):
    split_dir = output_dir / split_name
    copied_count = 0

    for source in files:
        relative_path = source.relative_to(input_dir)
        destination = split_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_count += 1

    return copied_count


def main():
    parser = argparse.ArgumentParser(
        description="Divide arquivos MIDI em conjuntos de treino, validação e teste."
    )
    parser.add_argument("input_dir", type=Path, help="Diretório com os arquivos MIDI.")
    parser.add_argument("output_dir", type=Path, help="Diretório de saída.")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        parser.error(f"Diretório de entrada não encontrado: {input_dir}")

    midi_files = find_midi_files(input_dir, output_dir)
    if not midi_files:
        parser.error(f"Nenhum arquivo .mid ou .midi encontrado em: {input_dir}")

    train_files, valid_files, test_files = split_files(midi_files)
    counts = {
        "train": copy_files(train_files, "train", input_dir, output_dir),
        "valid": copy_files(valid_files, "valid", input_dir, output_dir),
        "test": copy_files(test_files, "test", input_dir, output_dir),
    }

    print(f"MIDI files found: {len(midi_files)}")
    print(
        "Distribution: "
        f"train={counts['train']}, "
        f"valid={counts['valid']}, "
        f"test={counts['test']}"
    )


if __name__ == "__main__":
    main()