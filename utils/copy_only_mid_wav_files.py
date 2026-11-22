import os
import sys
import shutil

def main(input_dir, output_dir):

    mid_dir = os.path.join(output_dir, "mid")
    wav_dir = os.path.join(output_dir, "wav")

    os.makedirs(mid_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    counter = 0

    for root, _, files in os.walk(input_dir):

        if "generated.mid" in files and "generated.wav" in files:

            mid_src = os.path.join(root, "generated.mid")
            wav_src = os.path.join(root, "generated.wav")

            filename = f"song_{counter:03d}"

            mid_dst = os.path.join(mid_dir, filename + ".mid")
            wav_dst = os.path.join(wav_dir, filename + ".wav")

            shutil.copy2(mid_src, mid_dst)
            shutil.copy2(wav_src, wav_dst)

            print(f"Copiado: {filename}")

            counter += 1

    print("\nConcluído!")
    print(f"Pares encontrados/copied: {counter}")
    print(f"Arquivos MIDI: {mid_dir}")
    print(f"Arquivos WAV: {wav_dir}")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Uso: python collect_generated.py <input_dir> <output_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
