# Projeto de Detecção de Fraude em Transações de Cartão de Crédito

## Descrição

Este projeto implementa um sistema de detecção de fraude em transações de cartão de crédito utilizando técnicas de machine learning. O sistema é capaz de identificar transações fraudulentas com alta precisão e recall, mesmo em um conjunto de dados altamente desbalanceado.

## Estrutura do Projeto

```
fraud_detection_project/
├── data/                  # Diretório para armazenar dados
│   ├── raw/               # Dados brutos
│   └── processed/         # Dados processados
├── models/                # Modelos treinados e transformadores
├── notebooks/             # Jupyter notebooks para análise
├── reports/               # Relatórios, gráficos e resultados
├── src/                   # Código fonte
│   ├── data_preparation.py    # Preparação dos dados
│   ├── feature_transformation.py  # Transformação de features
│   ├── model_training.py     # Treinamento e avaliação de modelos
│   └── utils.py              # Funções utilitárias
├── config/                # Arquivos de configuração
├── main.py                # Script principal
└── README.md              # Este arquivo
```

## Requisitos

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib

Você pode instalar todas as dependências com:

```bash
pip install -r requirements.txt
```

## Como Usar

### 1. Preparação

Coloque os arquivos `fraudTrain.csv` e `fraudTest.csv` no diretório `data/raw/`.

### 2. Execução

Execute o script principal para processar os dados, treinar e avaliar os modelos:

```bash
python main.py
```

Ou execute cada etapa separadamente:

```bash
# Preparação dos dados
python src/data_preparation.py

# Transformação de features
python src/feature_transformation.py

# Treinamento e avaliação de modelos
python src/model_training.py
```

### 3. Notebooks

Explore os notebooks no diretório `notebooks/` para análises detalhadas:

- `01_exploratory_analysis.ipynb`: Análise exploratória dos dados
- `02_feature_engineering.ipynb`: Engenharia de features
- `03_model_evaluation.ipynb`: Avaliação detalhada dos modelos

## Características Principais

### Preparação de Dados

- Tratamento de valores ausentes
- Extração de componentes temporais (hora do dia, dia da semana, mês)
- Cálculo da distância entre cliente e comerciante
- Extração do primeiro e último dígito dos valores de transação
- Cálculo de velocidade entre transações consecutivas
- Detecção de desvios do padrão de gastos
- Criação da coluna "Idade" a partir da data de nascimento

### Transformação de Features

- Normalização apenas das colunas especificadas:
  - 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'distance_km', 'prev_lat', 'prev_long', 'prev_unix_time', 'time_diff_hours'
  
- One-hot encoding apenas para as colunas:
  - 'gender', 'job', 'city'

### Modelagem

- Treinamento de quatro algoritmos:
  - Regressão Logística
  - Random Forest
  - Gradient Boosted Trees
  - XGBoost
  
- Avaliação com métricas apropriadas para dados desbalanceados:
  - AUC-ROC
  - Precisão
  - Recall
  - F1-Score
  - Matriz de Confusão

## Licença

Este projeto é licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.
