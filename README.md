# Análise Comparativa de Algoritmos de Classificação: k-NN, Naive Bayes e Árvore de Decisão

[![Python](https://img.shields.io/badge/Python-3.12.7-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-red.svg)](https://opensource.org/licenses/)

## 📋 Sobre o Projeto

Este projeto apresenta uma **análise comparativa abrangente** de três algoritmos clássicos de aprendizado de máquina supervisionado para classificação: **k-Nearest Neighbors (k-NN)**, **Gaussian Naive Bayes** e **Decision Tree (Árvore de Decisão)**. O estudo foi desenvolvido como parte da disciplina PPGEP9002 - Inteligência Computacional para Engenharia de Produção da UFRN.

### 🎯 Objetivos

- Comparar o desempenho de três algoritmos de classificação em datasets canônicos
- Analisar o impacto de hiperparâmetros na performance dos modelos
- Avaliar a influência de diferentes proporções de treinamento (60%, 70%, 80%)
- Investigar o efeito da normalização Z-score no desempenho
- Fornecer insights sobre trade-offs entre acurácia, interpretabilidade e eficiência

### 📊 Datasets Utilizados

#### 1. **Iris Dataset** (UCI ML Repository - ID: 53)

- **Descrição**: Dataset clássico de classificação de espécies de íris
- **Amostras**: 150 (50 por classe)
- **Features**: 4 (comprimento e largura de sépalas e pétalas)
- **Classes**: 3 (Iris-setosa, Iris-versicolor, Iris-virginica)
- **Características**: Linearmente separável, bem balanceado

#### 2. **Wine Dataset** (UCI ML Repository - ID: 109)

- **Descrição**: Análise química de vinhos de diferentes cultivares
- **Amostras**: 178
- **Features**: 13 (conteúdo alcoólico, ácido málico, etc.)
- **Classes**: 3 (diferentes cultivares de uva)
- **Características**: Maior dimensionalidade, mais complexo que Iris

## 🛠️ Tecnologias Utilizadas

### **Linguagem de Programação**

- **Python 3.12.7**: Linguagem principal para desenvolvimento e análise de dados

### **Bibliotecas de Análise de Dados**

- **Pandas 2.2.2**: Manipulação e análise de dados estruturados
- **NumPy 1.26.4**: Computação numérica e operações com arrays multidimensionais
- **scikit-learn 1.5.0**: Biblioteca principal para algoritmos de machine learning

### **Visualização**

- **Matplotlib 3.9.0**: Criação de gráficos e visualizações estáticas
- **Seaborn 0.13.2**: Visualizações estatísticas avançadas e estilizadas

### **Ambiente de Desenvolvimento**

- **Jupyter Notebook**: Ambiente interativo para desenvolvimento e análise
- **ucimlrepo 0.0.3**: Acesso direto aos datasets do UCI Machine Learning Repository

### **Gerenciamento de Dependências**

- **uv**: Gerenciador de pacotes Python moderno e rápido

## 🧠 Algoritmos de Classificação Implementados

### 1. **k-Nearest Neighbors (k-NN)**

**Conceito**: Algoritmo não-paramétrico baseado em instâncias que classifica amostras baseado na classe majoritária de seus k vizinhos mais próximos.

**Funcionamento**:

- Armazena todo o conjunto de treinamento
- Para cada amostra de teste, calcula distâncias para todas as amostras de treino
- Seleciona os k vizinhos mais próximos
- Classifica baseado na classe majoritária entre os vizinhos

**Hiperparâmetros Testados**:

- `k ∈ {1, 3, 5, 7, 9}`: Número de vizinhos considerados
- `n_jobs = -1`: Paralelização para aceleração

**Vantagens**:

- Simples de implementar e entender
- Não assume distribuição específica dos dados
- Funciona bem com dados não-lineares

**Desvantagens**:

- Computacionalmente custoso para predição
- Sensível à escala das features
- Pode ser afetado por ruído

### 2. **Gaussian Naive Bayes**

**Conceito**: Classificador probabilístico baseado no Teorema de Bayes com suposição de independência condicional entre features.

**Funcionamento**:

- Calcula probabilidades a posteriori usando o Teorema de Bayes
- Assume que features seguem distribuição Gaussiana
- Aplica suposição "ingênua" de independência entre features
- Classifica baseado na maior probabilidade a posteriori

**Fórmula Principal**:

```
P(Classe|Features) ∝ P(Classe) × ∏ P(Feature_i|Classe)
```

**Vantagens**:

- Extremamente rápido para treinamento e predição
- Funciona bem com poucos dados
- Robusto a features irrelevantes

**Desvantagens**:

- Suposição de independência raramente é verdadeira
- Pode ter performance limitada em problemas complexos

### 3. **Decision Tree (Árvore de Decisão)**

**Conceito**: Algoritmo que cria um modelo de árvore de decisões através de divisões recursivas do espaço de features.

**Funcionamento**:

- Particiona recursivamente o espaço de features
- Cada nó representa uma decisão baseada em uma feature
- Cada folha representa uma classe predita
- Usa critérios como Gini ou Entropia para divisões

**Hiperparâmetros Testados**:

- `max_depth ∈ {3, 5, None}`: Profundidade máxima da árvore
- `random_state = 42`: Semente para reprodutibilidade

**Vantagens**:

- Altamente interpretável
- Não requer normalização
- Captura interações não-lineares
- Funciona com features categóricas e numéricas

**Desvantagens**:

- Propenso a overfitting
- Sensível a pequenas mudanças nos dados
- Pode criar árvores muito complexas

## 📈 Metodologia Experimental

### **Configuração dos Experimentos**

- **Repetições**: 10 execuções por configuração para robustez estatística
- **Divisão dos Dados**: Stratified split para manter proporção das classes
- **Proporções de Treino**: 60%, 70% e 80%
- **Pré-processamento**: Normalização Z-score aplicada aos dados
- **Métricas**: Acurácia, Precisão (macro), Revocação (macro), F1-Score (macro)

### **Pipeline de Avaliação**

1. Divisão estratificada dos dados
2. Normalização Z-score (treinada apenas nos dados de treino)
3. Treinamento do modelo
4. Predição no conjunto de teste
5. Cálculo das métricas de avaliação
6. Agregação estatística (média ± desvio padrão)

## 📊 Resultados Principais

### **Performance Geral por Dataset**

#### **Dataset Iris**

- **Melhor Modelo**: k-NN (k=5) e Árvore de Decisão (max_depth=5)
- **Acurácia Média**: ~97-98%
- **Observação**: Todos os modelos apresentaram excelente performance devido à separabilidade linear

#### **Dataset Wine**

- **Melhor Modelo**: k-NN (k=3-7) e Árvore de Decisão (max_depth=5)
- **Acurácia Média**: ~94-96%
- **Observação**: Maior variabilidade devido à complexidade do dataset

### **Análise de Hiperparâmetros**

#### **k-NN**

- **k=1**: Maior sensibilidade a ruído, menor estabilidade
- **k=3-7**: Performance ótima e estável
- **k=9**: Ligeira degradação em alguns casos

#### **Árvore de Decisão**

- **max_depth=3**: Modelo simples, bom para evitar overfitting
- **max_depth=5**: Balance ideal entre complexidade e generalização
- **max_depth=None**: Risco de overfitting, especialmente com poucos dados

### **Impacto da Normalização**

- **k-NN**: Beneficiou significativamente da normalização Z-score
- **Naive Bayes**: Pouco impacto (já assume distribuição normal)
- **Árvore de Decisão**: Não afetada (algoritmo baseado em divisões)

## 🚀 Como Executar o Projeto

### **Pré-requisitos**

- Python 3.12.7 ou superior
- Gerenciador de pacotes `uv` (recomendado) ou `pip`

### **Instalação no Windows**

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd "Atividade - 24.09.25 - classificação de padrões usando KNN"

# 2. Instale o uv (se não tiver)
pip install uv

# 3. Instale as dependências
uv pip install -r requirements.txt

# 4. Inicie o Jupyter Notebook
jupyter notebook main.ipynb
```

### **Instalação no Linux/macOS**

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd "Atividade - 24.09.25 - classificação de padrões usando KNN"

# 2. Instale o uv (se não tiver)
pip install uv
# ou no macOS com Homebrew:
# brew install uv

# 3. Instale as dependências
uv pip install -r requirements.txt

# 4. Inicie o Jupyter Notebook
jupyter notebook main.ipynb
```

### **Execução Alternativa com pip**

```bash
# Se preferir usar pip ao invés de uv
pip install -r requirements.txt
jupyter notebook main.ipynb
```

### **Estrutura de Execução do Notebook**

1. **Célula 1**: Instalação das dependências (opcional se já instaladas)
2. **Célula 2**: Importação das bibliotecas
3. **Célula 3**: Carregamento e inspeção dos datasets
4. **Célula 4**: Definição das funções auxiliares
5. **Células 5-8**: Experimentos com k-NN
6. **Células 9-11**: Experimentos com Naive Bayes
7. **Células 12-15**: Experimentos com Árvore de Decisão
8. **Células 16-17**: Análise comparativa final

## 📁 Estrutura do Projeto

```
📦 Atividade - 24.09.25 - classificação de padrões usando KNN
├── 📄 main.ipynb                    # Notebook principal com toda a análise
├── 📄 requirements.txt              # Dependências do projeto
├── 📄 README.md                     # Este arquivo
├── 📄 Relatório_v3.md               # Relatório detalhado em Markdown
├── 📄 tree.txt                      # Estrutura de diretórios
└── 📁 data/                         # Resultados dos experimentos
    ├── 📄 knn_raw_results.csv       # Resultados brutos do k-NN
    ├── 📄 nb_raw_results.csv        # Resultados brutos do Naive Bayes
    ├── 📄 dt_raw_results.csv        # Resultados brutos da Árvore de Decisão
    └── 📁 imgs/                     # Gráficos e visualizações
        ├── 📄 comparativo_desempenho_todos_modelos_output.png
        ├── 📄 impacto_k_acuracia_knn_output.png
        ├── 📄 impacto_max_depth_acuracia_dt_output.png
        ├── 📄 matriz_confusao_best_dt_output.png
        ├── 📄 matriz_confusao_best_knn_by_dataset_output.png
        └── 📄 matriz_confusao_best_nb_80porc_output.png

```

## 🔍 Principais Descobertas

### **1. Importância da Normalização**

- O k-NN foi significativamente beneficiado pela normalização Z-score
- A normalização é crucial para algoritmos baseados em distâncias

### **2. Trade-offs entre Modelos**

- **k-NN**: Melhor acurácia, mas computacionalmente custoso
- **Naive Bayes**: Excelente baseline, extremamente rápido
- **Árvore de Decisão**: Balance ideal entre performance e interpretabilidade

### **3. Sensibilidade a Hiperparâmetros**

- k-NN mostrou sensibilidade moderada ao valor de k
- Árvore de Decisão apresentou maior variabilidade com max_depth=None
- Naive Bayes não requer ajuste de hiperparâmetros

### **4. Robustez Estatística**

- 10 repetições foram suficientes para obter resultados estatisticamente robustos
- Desvio padrão baixo indica modelos estáveis

## 🎓 Conclusões

Este estudo demonstrou que **não existe um algoritmo universalmente superior**. A escolha do modelo ideal depende de:

- **Objetivo do projeto**: Máxima acurácia vs. interpretabilidade vs. velocidade
- **Características dos dados**: Dimensionalidade, separabilidade, tamanho
- **Recursos computacionais**: Tempo de treinamento e predição
- **Requisitos de interpretabilidade**: Necessidade de explicar decisões

### **Recomendações por Cenário**

- **Máxima Acurácia**: k-NN com k=5-7
- **Interpretabilidade**: Árvore de Decisão com max_depth=5
- **Velocidade**: Naive Bayes
- **Baseline Rápido**: Naive Bayes
- **Dados Complexos**: k-NN ou Árvore de Decisão

## 👨‍💻 Autor

**Ivan Pedro Varella Albuquerque**  
_Disciplina_: PPGEP9002 - Inteligência Computacional para Engenharia de Produção  
_Professor_: Jose Alfredo Ferreira Costa  
_Instituição_: Universidade Federal do Rio Grande do Norte (UFRN)

## 📚 Referências

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos como parte do programa de mestrado em Engenharia de Produção da UFRN.

---

_Última atualização: Setembro 2025_
