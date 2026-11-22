
import os
import sys
import random
import shutil

NUM_SAMPLES = 100

def main(input_dir, output_dir):

    MID_DIR = os.path.join(input_dir, "mid")
    WAV_DIR = os.path.join(input_dir, "wav")

    OUTPUT_MID = os.path.join(output_dir, "mid")
    OUTPUT_WAV = os.path.join(output_dir, "wav")

    # Verificar diretórios de entrada
    if not os.path.isdir(MID_DIR) or not os.path.isdir(WAV_DIR):
        raise RuntimeError("O input_dir deve conter as pastas mid/ e wav/.")

    # Criar diretórios de saída
    os.makedirs(OUTPUT_MID, exist_ok=True)
    os.makedirs(OUTPUT_WAV, exist_ok=True)

    # Encontrar todos os pares MIDI/WAV
    valid_midis = []

    for root, _, files in os.walk(MID_DIR):
        for filename in files:
            if filename.lower().endswith(".mid"):

                mid_path = os.path.join(root, filename)

                relative_path = os.path.relpath(mid_path, MID_DIR)

                wav_relative = os.path.splitext(relative_path)[0] + ".wav"
                wav_path = os.path.join(WAV_DIR, wav_relative)

                if os.path.isfile(wav_path):
                    valid_midis.append((mid_path, wav_path, relative_path))

    # Verificar quantidade disponível
    if len(valid_midis) < NUM_SAMPLES:
        raise RuntimeError(
            f"Encontrados apenas {len(valid_midis)} pares MIDI/WAV. "
            f"São necessários {NUM_SAMPLES}."
        )

    # Selecionar aleatoriamente
    selected = random.sample(valid_midis, NUM_SAMPLES)

    # Copiar arquivos preservando a estrutura
    for mid_path, wav_path, relative_path in selected:

        mid_dest = os.path.join(OUTPUT_MID, relative_path)

        wav_relative = os.path.splitext(relative_path)[0] + ".wav"
        wav_dest = os.path.join(OUTPUT_WAV, wav_relative)

        os.makedirs(os.path.dirname(mid_dest), exist_ok=True)
        os.makedirs(os.path.dirname(wav_dest), exist_ok=True)

        shutil.copy2(mid_path, mid_dest)
        shutil.copy2(wav_path, wav_dest)

    print("Concluído!")
    print(f"Pares selecionados: {len(selected)}")
    print(f"Arquivos MIDI: {len(selected)}")
    print(f"Arquivos WAV: {len(selected)}")
    print(f"Total de arquivos copiados: {len(selected) * 2}")
    print(f"Destino: {output_dir}")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Uso: python select_pairs.py <input_dir> <output_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
