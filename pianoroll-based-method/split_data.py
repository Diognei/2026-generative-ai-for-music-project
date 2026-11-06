
import random
import shutil
from pathlib import Path

TRAIN_PERCENT = 0.8
VALID_PERCENT = 0.1
TEST_PERCENT = 0.1

SEED = 42

SOURCE_DIR = Path("./data/full")
TRAIN_DIR = Path("./data/train")
VALID_DIR = Path("./data/valid")
TEST_DIR = Path("./data/test")


def main():
    total_percent = TRAIN_PERCENT + VALID_PERCENT + TEST_PERCENT

    if abs(total_percent - 1.0) > 1e-6:
        raise ValueError("A soma dos percentuais deve ser igual a 1.0.")

    files = sorted(SOURCE_DIR.rglob("*.npz"))

    if not files:
        print(f"Nenhum arquivo .npz encontrado em {SOURCE_DIR}.")
        return

    random.Random(SEED).shuffle(files)

    total = len(files)
    train_count = int(total * TRAIN_PERCENT)
    valid_count = int(total * VALID_PERCENT)

    train_files = files[:train_count]
    valid_files = files[train_count:train_count + valid_count]
    test_files = files[train_count + valid_count:]

    splits = [
        (train_files, TRAIN_DIR),
        (valid_files, VALID_DIR),
        (test_files, TEST_DIR),
    ]

    for split_files, destination in splits:
        destination.mkdir(parents=True, exist_ok=True)

        for file in split_files:
            relative_path = file.relative_to(SOURCE_DIR)
            target = destination / relative_path

            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), str(target))

    print(f"Total: {total}")
    print(f"Train: {len(train_files)} ({len(train_files) / total:.1%})")
    print(f"Valid: {len(valid_files)} ({len(valid_files) / total:.1%})")
    print(f"Test:  {len(test_files)} ({len(test_files) / total:.1%})")


if __name__ == "__main__":
    main()
