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
python3 generator/generate.py output/lstm_model.pt
```

Calcular as métricas simbólicas de diversidade e similaridade (Yang & Lerch, 2020) de uma pasta de MIDIs gerados contra o conjunto de teste:

```bash
python3 evaluator/calc_diversity.py <pasta_de_midis_gerados> data/test --json output/diversity_<sistema>.json
```
