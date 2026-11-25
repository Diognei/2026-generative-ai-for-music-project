# Generative AI for Music - Final Project

## Como rodar

Obs.: é necessário ter o docker instalado, ou criar seu próprio virtual environment com as bibliotecas em requirements.txt

Inicializar o container docker:

```bash
./start_container.sh
```

Treinar o gerador de músicas:

```bash
python3 generator/train.py
```

Treinar o baseline (LSTM, mesmos dados, tokens e laço de treino; salva `output/lstm_model.pt`):

```bash
python3 generator/train.py --arch lstm
```

Gerar uma nova música (a arquitetura é lida do checkpoint, então o mesmo comando serve para o baseline):

```bash
python3 generator/generate.py checkpoints/model_best.pt
python3 generator/generate.py checkpoints/lstm_model_best.pt
```

Calcular as métricas simbólicas de diversidade e similaridade (Yang & Lerch, 2020) de uma pasta de MIDIs gerados contra o conjunto de teste:

```bash
python3 evaluator/calc_diversity.py <pasta_de_midis_gerados> data/test --json output/diversity_<sistema>.json
```

Calcular as métricas de áudio:

```bash
python3 evaluator/calc_fad_global.py <target_wav_songs_dir> <reference_wav_songs_dir>
python3 evaluator/calc_kld_global.py <target_wav_songs_dir> <reference_wav_songs_dir>
python3 evaluator/calc_fad.py <target_wav_songs_dir> <reference_wav_songs_dir>
python3 evaluator/calc_kld.py <target_wav_songs_dir> <reference_wav_songs_dir>
```

## Context Window Ablation Study

Execute os comandos abaixo a partir da raiz do repositório para reproduzir o estudo de ablação de janela de contexto (`ctx_256` vs. `ctx_1024`, subconjunto fixo de 20% de treino, 30 épocas, seed 42):

1. Treinar os modelos de ablação (`block_size` 256 vs. 1024):

```bash
python3 generator/train_ablation.py --exp_name ctx_256 --block_size 256 --train_fraction 0.20 --epochs 30
python3 generator/train_ablation.py --exp_name ctx_1024 --block_size 1024 --train_fraction 0.20 --epochs 30
```

2. Gerar `N=100` amostras MIDI por checkpoint (mantém MIDIs brutos por padrão; adicione `--align` para aplicar pós-processamento via `fix_midi()`):

```bash
python3 generator/generate_ablation_batch.py --checkpoint output_ablation/ctx_256/ctx_256_best.pt --output_dir output_ablation/eval_ctx_256/midi --num_samples 100 --seed 42
python3 generator/generate_ablation_batch.py --checkpoint output_ablation/ctx_1024/ctx_1024_best.pt --output_dir output_ablation/eval_ctx_1024/midi --num_samples 100 --seed 42
```

3. Converter os diretórios de MIDI para WAV:

```bash
python3 evaluator/convert_dir_mid_to_wav.py output_ablation/eval_ctx_256/midi output_ablation/eval_ctx_256/wav
python3 evaluator/convert_dir_mid_to_wav.py output_ablation/eval_ctx_1024/midi output_ablation/eval_ctx_1024/wav
python3 evaluator/convert_dir_mid_to_wav.py output_ablation/ref_valid/midi output_ablation/ref_valid/wav
```

4. Calcular FAD e KLD contra o conjunto de referência (`output_ablation/ref_valid/wav`):

```bash
python3 evaluator/calc_fad.py output_ablation/eval_ctx_256/wav output_ablation/ref_valid/wav
python3 evaluator/calc_kld.py output_ablation/eval_ctx_256/wav output_ablation/ref_valid/wav

python3 evaluator/calc_fad.py output_ablation/eval_ctx_1024/wav output_ablation/ref_valid/wav
python3 evaluator/calc_kld.py output_ablation/eval_ctx_1024/wav output_ablation/ref_valid/wav
```
