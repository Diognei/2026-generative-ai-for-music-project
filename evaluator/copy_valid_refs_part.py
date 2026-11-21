
import os
import random
import shutil

# Diretórios
BASE_DIR = "../output/evaluation"

MID_DIR = os.path.join(BASE_DIR, "set1b_lakh_valid_refs_full", "mid")
WAV_DIR = os.path.join(BASE_DIR, "set1b_lakh_valid_refs_full", "wav")
OUTPUT_DIR = os.path.join(BASE_DIR, "set1_lakh_valid_refs_part")

NUM_SAMPLES = 100

# Encontrar todos os arquivos MIDI que possuem WAV correspondente
valid_midis = []

for root, _, files in os.walk(MID_DIR):
    for filename in files:
        if filename.lower().endswith(".mid"):
            mid_path = os.path.join(root, filename)

            # Obter caminho relativo em relação à pasta mid
            relative_path = os.path.relpath(mid_path, MID_DIR)

            # Construir caminho correspondente na pasta wav
            wav_relative_path = os.path.splitext(relative_path)[0] + ".wav"
            wav_path = os.path.join(WAV_DIR, wav_relative_path)

            if os.path.isfile(wav_path):
                valid_midis.append((mid_path, wav_path, relative_path))

# Verificar quantidade disponível
if len(valid_midis) < NUM_SAMPLES:
    raise RuntimeError(
        f"Encontrados apenas {len(valid_midis)} pares MIDI/WAV. "
        f"São necessários {NUM_SAMPLES}."
    )

# Selecionar aleatoriamente 100 pares
selected = random.sample(valid_midis, NUM_SAMPLES)

# Copiar arquivos preservando a estrutura de diretórios
for mid_path, wav_path, relative_path in selected:

    # Destinos MIDI e WAV
    mid_dest = os.path.join(OUTPUT_DIR, "mid", relative_path)
    wav_relative = os.path.splitext(relative_path)[0] + ".wav"
    wav_dest = os.path.join(OUTPUT_DIR, "wav", wav_relative)

    # Criar subdiretórios
    os.makedirs(os.path.dirname(mid_dest), exist_ok=True)
    os.makedirs(os.path.dirname(wav_dest), exist_ok=True)

    # Copiar arquivos
    shutil.copy2(mid_path, mid_dest)
    shutil.copy2(wav_path, wav_dest)

print(f"Concluído!")
print(f"Pares selecionados: {len(selected)}")
print(f"Arquivos MIDI: {len(selected)}")
print(f"Arquivos WAV: {len(selected)}")
print(f"Total de arquivos copiados: {len(selected) * 2}")
print(f"Destino: {OUTPUT_DIR}")
