# Generative AI for Music - Final Project

## Como rodar

```bash
./start_container.sh
```

```bash
python3 src/train.py
```

```bash
python3 src/generate.py output/checkpoint_epoch_50.pt
```

## Tarefas

- Tarefa geral: Composição livre (incondicional), a partir do zero ou continuando um trecho inicial (prompt);
  - Treinar um Transformer para aprender a gerar músicas parecidas com as de um dataset de midi amplo: lakh midi dataset.

- [x] Implementação do método principal
  - [x] Implementar o código do Transformer principal
  - [x] Treiná-lo com o dataset lakh midi dataset inteiro
  - [ ] Definir uma estratégia melhor para inicializar os tokens

- [ ] Implementação dos baselines
  - [ ] Implementar um baseline para comparação
  - [ ] Definir outros baselines que já temos prontos
    
- [ ] Implementação das métricas de avaliação
  - [ ] FAD
  - [ ] KLD
  - [ ] Pesquisar outra métrica para medir diversidade das músicas geradas
  
- [ ] Pesquisa por trabalhos relacionados
  - [ ] Pesquisar 5 artigos do ISMIR que façam algo parecido com o nosso trabalho
  - [ ] Ler os artigos e definir quais as contribuições do nosso método em relação a cada um deles
    
- [ ] Escrita do artigo e apresentação
  - [ ] Organizar as seções do artigo
  - [ ] Escrever o artigo
  - [ ] Preparar a apresentação
