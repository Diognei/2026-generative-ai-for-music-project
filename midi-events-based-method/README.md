# MusicGen — primeiro experimento

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

- `BLOCK_SIZE=512` limita o contexto do Transformer. Músicas maiores são
  divididas em janelas independentes durante o treinamento.
- O experimento usa tempo em segundos quantizado em passos de 10 ms.
  A reconstrução preserva os tempos aproximados das notas, mas não garante
  preservar metadados originais como andamento, fórmulas de compasso e letras.
- A geração inicial é limitada a uma pista. O constrained decoder já mantém
  a lógica de múltiplas pistas configurável, mas a geração multitrack ainda
  precisará de ajustes e testes específicos.
- Este é um baseline didático para validar o pipeline, não uma implementação
  otimizada para treinamento em larga escala.
