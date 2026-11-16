# Generative AI for Music - Final Project

## Como rodar

### Inicializar o container docker 

É necessário ter o docker instalado, ou criar seu próprio virtual environment a partir do requirements.txt

```bash
./start_container.sh
```

### Treinar o gerador de músicas

```bash
python3 generator/train.py
```

### Gerar uma nova música

```bash
python3 generator/generate.py checkpoints/checkpoint_epoch_50.pt
```

## Tarefas

- Tarefa geral: Composição livre (incondicional), a partir do zero ou continuando um trecho inicial (prompt);
  - Treinar um Transformer para aprender a gerar músicas parecidas com as de um dataset de midi amplo: lakh midi dataset.

### Related Work

- [ ] Pesquisar por trabalhos relacionados
  - [ ] Pesquisar 5 artigos do ISMIR que façam algo parecido com o nosso trabalho
  - [ ] Escrever o Related Work citando os artigos e definindo quais as nossas contribuições em relação a eles

### Methodology

- [x] Implementação do método principal
  - [x] Implementar o código do Transformer principal
  - [x] Treiná-lo com o dataset lakh midi dataset inteiro
  - [ ] Definir uma estratégia melhor para inicializar os tokens

### Experiments

- [ ] Implementação dos baselines
  - [ ] Implementar um baseline para comparação
  - [ ] Definir outros baselines que já temos prontos
    
- [ ] Implementação das métricas de avaliação
  - [x] Implementar o código de cálculo do FAD
  - [x] Implementar o código de cálculo do KLD
  - [ ] Pesquisar outra métrica para medir diversidade das músicas geradas

- [ ] Implementar código de geração da tabela de resultados quantitativos
  - [ ] Definir o formato da tabela (métricas vs métodos)
  - [ ] Definir os dados nos quais cada métrica será calculada
  - [ ] Calcular as métricas em todos os dados gerando a tabela
  - [ ] Verificar se é necessário gerar alguma outra tabela

- [ ] Implemenar código de geração da figura de resultados qualitativos
  - [ ] Definir o formato da figura (figura sumarizando as músicas geradas)
  - [ ] Implementar código para geração da figura a partir de uma pasta de músicas geradas

### Paper

- [ ] Escrita do artigo e apresentação
  - [ ] Organizar as seções do artigo
  - [ ] Escrever o artigo
  - [ ] Preparar a apresentação
