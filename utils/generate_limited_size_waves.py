
import os
import sys
import wave
import shutil

#MAX_SIZE = 20 * 1024 * 1024  # 20 MB (20 MiB)

MAX_SIZE = 5 * 1024 * 1024  # 5 MB (5 MiB)

def process_wav(input_path, output_path):

    file_size = os.path.getsize(input_path)

    # Se já estiver dentro do limite, apenas copiar
    if file_size <= MAX_SIZE:
        shutil.copy2(input_path, output_path)
        return False

    # Abrir WAV original
    with wave.open(input_path, "rb") as wav_in:

        params = wav_in.getparams()

        # Tamanho de cada frame em bytes
        frame_size = params.nchannels * params.sampwidth

        # Cabeçalho WAV PCM padrão: 44 bytes
        header_size = 44

        # Quantidade máxima de frames para respeitar o limite
        max_frames = (MAX_SIZE - header_size) // frame_size

        # Ler somente o início do áudio
        audio_data = wav_in.readframes(max_frames)

    # Salvar WAV reduzido
    with wave.open(output_path, "wb") as wav_out:
        wav_out.setparams(params)
        wav_out.writeframes(audio_data)

    return True


def main(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    total = 0
    modified = 0
    copied = 0
    errors = 0

    for root, _, files in os.walk(input_dir):

        for filename in files:

            if not filename.lower().endswith(".wav"):
                continue

            input_path = os.path.join(root, filename)

            # Preservar estrutura de subdiretórios
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                was_modified = process_wav(input_path, output_path)

                total += 1

                if was_modified:
                    modified += 1
                    print(f"[REDUZIDO] {relative_path}")
                else:
                    copied += 1
                    print(f"[COPIADO] {relative_path}")

            except (wave.Error, EOFError, OSError) as e:

                errors += 1
                print(f"[ERRO] {relative_path}: {e}")

    print("\n===== RESULTADO =====")
    print(f"Total de arquivos: {total}")
    print(f"Arquivos reduzidos: {modified}")
    print(f"Arquivos apenas copiados: {copied}")
    print(f"Erros: {errors}")
    print(f"Destino: {output_dir}")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Uso: python limit_wav.py <input_dir> <output_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
