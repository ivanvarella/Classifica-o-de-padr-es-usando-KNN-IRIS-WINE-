# Plano de Ação Final v2 (Estrutura por Modelo)

## Estrutura do Notebook

### Parte I: Preparação e Ferramentas

#### Etapa 1: Introdução (Markdown)

- Apresentação geral do projeto, datasets e os três modelos que serão investigados.

#### Etapa 2: Configuração do Ambiente, Carga dos Dados e Funções Auxiliares

- Instalação e importação das bibliotecas.
- Carregamento dos datasets Iris e Wine.
- Breve Análise Exploratória de Dados (EDA).
- **Ponto-chave:** Criação de uma **função de avaliação reutilizável**. Para evitar a repetição de código, criaremos uma função (ex: `executar_experimento`) que receberá o modelo, os dados e os parâmetros, e realizará o loop de `N_REPETICOES` para calcular e retornar as métricas. Isso manterá nosso código limpo (princípio DRY - Don't Repeat Yourself).

### Parte II: Análise Individual dos Modelos

#### Etapa 3: Estudo de Caso - k-Nearest Neighbors (k-NN)

- Breve introdução teórica sobre o k-NN (Markdown).
- Definição dos hiperparâmetros a serem testados (a lista de `k`).
- Execução dos experimentos para o k-NN em ambos os datasets, utilizando nossa função auxiliar.
- Agregação dos resultados (média e desvio padrão).
- Análise dos resultados do k-NN:
  - Tabelas de desempenho.
  - Gráficos mostrando o impacto do `k` na performance.
  - Matriz de confusão para a melhor configuração.
- Breve conclusão sobre o desempenho do k-NN (Markdown).

#### Etapa 4: Estudo de Caso - Naive Bayes

- Breve introdução teórica sobre o Naive Bayes (Markdown).
- Execução dos experimentos para o Naive Bayes.
- Agregação dos resultados.
- Análise dos resultados do Naive Bayes:
  - Tabelas de desempenho.
  - Matriz de confusão.
- Breve conclusão sobre o desempenho do Naive Bayes (Markdown).

#### Etapa 5: Estudo de Caso - Árvore de Decisão

- Breve introdução teórica sobre a Árvore de Decisão (Markdown).
- Definição dos hiperparâmetros a serem testados (a lista de `max_depth`).
- Execução dos experimentos para a Árvore de Decisão.
- Agregação dos resultados.
- Análise dos resultados da Árvore de Decisão:
  - Tabelas de desempenho.
  - Gráficos mostrando o impacto do `max_depth`.
  - Matriz de confusão para a melhor configuração.
- Breve conclusão sobre o desempenho da Árvore de Decisão (Markdown).

### Parte III: Síntese e Conclusões Finais

#### Etapa 6: Análise Comparativa Final e Conclusões

- **Junção dos Resultados:** Concatenar os DataFrames de resultados agregados de cada um dos três modelos.
- **Visualização Comparativa:** Criar tabelas e gráficos (ex: gráficos de barras) que coloquem os melhores resultados de cada modelo lado a lado para cada dataset.
- **Discussão Final (Markdown):**
  - Responder qual modelo foi o "vencedor" para cada dataset.
  - Discutir os prós e contras de cada algoritmo com base nos resultados práticos observados (ex: velocidade, interpretabilidade, performance).
  - Apresentar as conclusões gerais do projeto.
