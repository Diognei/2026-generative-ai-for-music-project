# MusicGen — versão com relógio global

Esta pasta é uma versão independente do projeto original. Ela mantém as
dependências auxiliares, mas não inclui `data/`, outras pastas `data*` nem
`output/`; coloque os arquivos MIDI de treino em `data/` quando for executar.

As pistas agora são codificadas em um relógio compartilhado. Os eventos são
ordenados por tempo absoluto e usam `TRACK_SELECT_n`, portanto notas
simultâneas de instrumentos diferentes não reiniciam o relógio a cada pista.
O treinamento também usa contexto de 2048 tokens, Transformer de 256
dimensões com 6 camadas e 8 cabeças, máscara de padding e validação por
arquivo MIDI. O melhor checkpoint de validação é salvo em `output/model.pt`.

Para treinar:

```bash
python src/train.py
```

Para gerar:

```bash
python src/generate.py
```

## MusicGen — primeiro experimento

Pipeline mínimo de geração MIDI com Transformer Decoder-only.

## Estrutura

```text
musicgen/
├── data/
│   └── song1.mid
├── output/
├── src/
│   ├── vocabulary.py
│   ├── midi_events.py
│   ├── dataset.py
│   ├── model.py
│   ├── constraints.py
│   ├── train.py
│   └── generate.py
└── requirements.txt
```

## Instalação

```bash
pip install -r requirements.txt
```

Coloque o arquivo MIDI em `data/song1.mid`. O caminho é de arquivo;
não coloque uma barra `/` depois de `.mid`.

## Treinamento

Na raiz do projeto:

```bash
python src/train.py
```

O script procura todos os arquivos `*.mid` e `*.midi` em `data/`.
Para começar com uma única música, deixe somente `song1.mid` nessa pasta.

O checkpoint será salvo em `output/model.pt`.

## Geração

```bash
python src/generate.py
```

O resultado será salvo em uma nova pasta `outX/`, onde `X` é o primeiro
índice disponível (`out1/`, `out2/`, `out3/`, ...). A pasta conterá o MIDI,
o pianoroll em PNG, o áudio WAV e o vídeo MP4 gerados para aquela execução.

Ajuste `TEMPERATURE` em `src/generate.py` para controlar a diversidade.

## Observações

- `BLOCK_SIZE=2048` limita o contexto do Transformer. Músicas maiores são
  divididas em janelas independentes durante o treinamento.
- O experimento usa tempo em segundos quantizado em passos de 10 ms.
  A reconstrução preserva os tempos aproximados das notas, mas não garante
  preservar metadados originais como andamento, fórmulas de compasso e letras.
- A geração usa o plano de pistas de uma referência aleatória e mantém um
  relógio global; a qualidade ainda depende do dataset e da amostragem.
- Este é um baseline experimental com representação multipista mais robusta,
  não uma implementação otimizada para treinamento em larga escala.
