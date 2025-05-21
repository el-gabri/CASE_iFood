#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal para executar todo o pipeline de detecção de fraude.
Este script orquestra a execução de todas as etapas do projeto.
"""

import os
import sys
import argparse
import time
import pandas as pd
import shutil

# Adicionar diretório raiz ao path para importar módulos do projeto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import ensure_dir
from src.data_preparation import load_data, check_missing_values, feature_engineering, balance_data
from src.feature_transformation import apply_transformations, save_transformers, prepare_features_for_modeling
from src.model_training import (
    train_logistic_regression, train_random_forest, 
    train_gradient_boosting, train_xgboost,
    evaluate_model, save_model
)

def setup_directories():
    """
    Configura a estrutura de diretórios do projeto
    
    Returns:
    --------
    dict
        Dicionário com os caminhos dos diretórios
    """
    print("Configurando estrutura de diretórios...")
    
    # Definir caminhos
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    notebooks_dir = os.path.join(base_dir, 'notebooks')
    config_dir = os.path.join(base_dir, 'config')
    
    # Criar diretórios
    for directory in [data_dir, raw_dir, processed_dir, models_dir, reports_dir, notebooks_dir, config_dir]:
        ensure_dir(directory)
    
    # Retornar dicionário com caminhos
    return {
        'base_dir': base_dir,
        'data_dir': data_dir,
        'raw_dir': raw_dir,
        'processed_dir': processed_dir,
        'models_dir': models_dir,
        'reports_dir': reports_dir,
        'notebooks_dir': notebooks_dir,
        'config_dir': config_dir
    }

def check_data_files(raw_dir):
    """
    Verifica se os arquivos de dados estão presentes
    
    Parameters:
    -----------
    raw_dir : str
        Diretório com os dados brutos
        
    Returns:
    --------
    bool
        True se os arquivos estão presentes, False caso contrário
    """
    train_path = os.path.join(raw_dir, 'fraudTrain.csv')
    test_path = os.path.join(raw_dir, 'fraudTest.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Erro: Arquivos de dados não encontrados em {raw_dir}")
        print("Por favor, coloque os arquivos 'fraudTrain.csv' e 'fraudTest.csv' no diretório 'data/raw'")
        return False
    
    return True

def run_data_preparation(dirs):
    """
    Executa a etapa de preparação dos dados
    
    Parameters:
    -----------
    dirs : dict
        Dicionário com os caminhos dos diretórios
        
    Returns:
    --------
    tuple
        (DataFrame de treino, DataFrame de teste, colunas categóricas, colunas numéricas para normalizar, colunas numéricas sem normalização)
    """
    print("\n" + "="*80)
    print("ETAPA 1: PREPARAÇÃO DOS DADOS")
    print("="*80)
    
    # Caminhos dos arquivos
    train_path = os.path.join(dirs['raw_dir'], 'fraudTrain.csv')
    test_path = os.path.join(dirs['raw_dir'], 'fraudTest.csv')
    
    # Carregar dados
    train_df, test_df = load_data(train_path, test_path)
    
    # Verificar valores ausentes
    check_missing_values(train_df, test_df)
    
    # Realizar feature engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    # Identificar colunas
    categorical_cols = ['gender', 'job', 'city']
    numeric_cols_to_normalize = [
        'zip', 'lat', 'long', 'city_pop', 'unix_time', 
        'merch_lat', 'merch_long', 'distance_km', 
        'prev_lat', 'prev_long', 'prev_unix_time', 'time_diff_hours'
    ]
    numeric_cols_no_normalize = [
        'amt', 'hour_of_day', 'day_of_week', 'month', 
        'first_digit', 'last_digit', 'transaction_velocity_kmh', 
        'impossible_velocity', 'avg_amt', 'stddev_amt', 
        'amt_zscore', 'amt_is_outlier', 'idade'
    ]
    
    # Balancear dados de treino
    train_df_balanced = balance_data(train_df)
    
    # Salvar dados processados
    train_df_balanced.to_csv(os.path.join(dirs['processed_dir'], 'train_processed.csv'), index=False)
    test_df.to_csv(os.path.join(dirs['processed_dir'], 'test_processed.csv'), index=False)
    
    print("\nDados processados salvos em:")
    print(f"- {os.path.join(dirs['processed_dir'], 'train_processed.csv')}")
    print(f"- {os.path.join(dirs['processed_dir'], 'test_processed.csv')}")
    
    return train_df_balanced, test_df, categorical_cols, numeric_cols_to_normalize, numeric_cols_no_normalize

def run_feature_transformation(train_df, test_df, categorical_cols, numeric_cols_to_normalize, numeric_cols_no_normalize, dirs):
    """
    Executa a etapa de transformação de features
    
    Parameters:
    -----------
    train_df : DataFrame
        DataFrame de treino
    test_df : DataFrame
        DataFrame de teste
    categorical_cols : list
        Lista de colunas categóricas
    numeric_cols_to_normalize : list
        Lista de colunas numéricas para normalizar
    numeric_cols_no_normalize : list
        Lista de colunas numéricas sem normalização
    dirs : dict
        Dicionário com os caminhos dos diretórios
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test, lista de features)
    """
    print("\n" + "="*80)
    print("ETAPA 2: TRANSFORMAÇÃO DE FEATURES")
    print("="*80)
    
    # Aplicar transformações
    train_transformed, test_transformed, transformers = apply_transformations(
        train_df, test_df, categorical_cols, numeric_cols_to_normalize
    )
    
    # Salvar transformadores
    save_transformers(transformers, dirs['models_dir'])
    
    # Preparar features para modelagem
    X_train, y_train, X_test, y_test, feature_cols = prepare_features_for_modeling(
        train_transformed, test_transformed, 
        categorical_cols, numeric_cols_to_normalize, numeric_cols_no_normalize
    )
    
    # Salvar dados transformados
    train_transformed.to_csv(os.path.join(dirs['processed_dir'], 'train_transformed.csv'), index=False)
    test_transformed.to_csv(os.path.join(dirs['processed_dir'], 'test_transformed.csv'), index=False)
    
    # Salvar features e targets para modelagem
    X_train.to_csv(os.path.join(dirs['processed_dir'], 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(dirs['processed_dir'], 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(dirs['processed_dir'], 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(dirs['processed_dir'], 'y_test.csv'), index=False)
    
    # Salvar lista de features
    with open(os.path.join(dirs['processed_dir'], 'feature_cols.txt'), 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    print("\nDados transformados salvos em:")
    print(f"- {os.path.join(dirs['processed_dir'], 'train_transformed.csv')}")
    print(f"- {os.path.join(dirs['processed_dir'], 'test_transformed.csv')}")
    
    return X_train, y_train, X_test, y_test, feature_cols

def run_model_training(X_train, y_train, X_test, y_test, feature_cols, dirs):
    """
    Executa a etapa de treinamento e avaliação de modelos
    
    Parameters:
    -----------
    X_train : DataFrame
        Features de treino
    y_train : array
        Target de treino
    X_test : DataFrame
        Features de teste
    y_test : array
        Target de teste
    feature_cols : list
        Lista de nomes das features
    dirs : dict
        Dicionário com os caminhos dos diretórios
        
    Returns:
    --------
    DataFrame
        DataFrame com resultados da avaliação dos modelos
    """
    print("\n" + "="*80)
    print("ETAPA 3: TREINAMENTO E AVALIAÇÃO DE MODELOS")
    print("="*80)
    
    # Treinar modelos
    models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "Gradient Boosted Trees": train_gradient_boosting(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train)
    }
    
    # Avaliar modelos
    results = []
    for model_name, model in models.items():
        # Avaliar modelo
        model_results = evaluate_model(
            model, X_test, y_test, model_name, 
            feature_cols=feature_cols, 
            output_dir=dirs['reports_dir']
        )
        results.append(model_results)
        
        # Salvar modelo
        save_model(model, model_name, dirs['models_dir'])
    
    # Criar DataFrame com resultados
    results_df = pd.DataFrame(results)
    
    # Salvar resultados em CSV
    results_path = os.path.join(dirs['reports_dir'], 'model_evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResultados salvos em: {results_path}")
    
    # Identificar melhor modelo com base no F1-Score
    best_model_idx = results_df['f1'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'model']
    print(f"\nMelhor modelo baseado no F1-Score: {best_model_name}")
    print(f"F1-Score: {results_df.loc[best_model_idx, 'f1']:.4f}")
    print(f"AUC-ROC: {results_df.loc[best_model_idx, 'auc']:.4f}")
    print(f"Precisão: {results_df.loc[best_model_idx, 'precision']:.4f}")
    print(f"Recall: {results_df.loc[best_model_idx, 'recall']:.4f}")
    
    # Salvar resumo do melhor modelo
    summary_path = os.path.join(dirs['reports_dir'], 'best_model_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("RESUMO DO MODELO DE DETECÇÃO DE FRAUDE\n")
        f.write("=====================================\n\n")
        f.write(f"Melhor modelo: {best_model_name}\n")
        f.write(f"AUC-ROC: {results_df.loc[best_model_idx, 'auc']:.4f}\n")
        f.write(f"Precisão (classe fraude): {results_df.loc[best_model_idx, 'precision']:.4f}\n")
        f.write(f"Recall (classe fraude): {results_df.loc[best_model_idx, 'recall']:.4f}\n")
        f.write(f"F1-Score (classe fraude): {results_df.loc[best_model_idx, 'f1']:.4f}\n")
        f.write(f"Acurácia: {results_df.loc[best_model_idx, 'accuracy']:.4f}\n\n")
        
        f.write("Matriz de Confusão:\n")
        f.write(f"Verdadeiros Positivos: {results_df.loc[best_model_idx, 'tp']}\n")
        f.write(f"Falsos Positivos: {results_df.loc[best_model_idx, 'fp']}\n")
        f.write(f"Verdadeiros Negativos: {results_df.loc[best_model_idx, 'tn']}\n")
        f.write(f"Falsos Negativos: {results_df.loc[best_model_idx, 'fn']}\n\n")
        
        f.write("Interpretação:\n")
        f.write(f"- O modelo consegue identificar {results_df.loc[best_model_idx, 'recall']*100:.1f}% das fraudes\n")
        f.write(f"- Das transações classificadas como fraude, {results_df.loc[best_model_idx, 'precision']*100:.1f}% são realmente fraudes\n")
        f.write(f"- A taxa de falsos positivos é de {results_df.loc[best_model_idx, 'fp']/(results_df.loc[best_model_idx, 'fp']+results_df.loc[best_model_idx, 'tn'])*100:.2f}%\n")
    
    print(f"Resumo do melhor modelo salvo em: {summary_path}")
    
    return results_df

def create_readme(dirs, results_df=None):
    """
    Cria o arquivo README.md com instruções de uso
    
    Parameters:
    -----------
    dirs : dict
        Dicionário com os caminhos dos diretórios
    results_df : DataFrame, optional
        DataFrame com resultados da avaliação dos modelos
    """
    print("\n" + "="*80)
    print("ETAPA 4: CRIAÇÃO DE DOCUMENTAÇÃO")
    print("="*80)
    
    readme_path = os.path.join(dirs['base_dir'], 'README.md')
    
    with open(readme_path, 'w') as f:
        f.write("# Projeto de Detecção de Fraude em Transações de Cartão de Crédito\n\n")
        
        f.write("## Descrição\n\n")
        f.write("Este projeto implementa um sistema de detecção de fraude em transações de cartão de crédito utilizando técnicas de machine learning. O sistema é capaz de identificar transações fraudulentas com alta precisão e recall, mesmo em um conjunto de dados altamente desbalanceado.\n\n")
        
        f.write("## Estrutura do Projeto\n\n")
        f.write("```\n")
        f.write("fraud_detection_project/\n")
        f.write("├── data/                  # Diretório para armazenar dados\n")
        f.write("│   ├── raw/               # Dados brutos\n")
        f.write("│   └── processed/         # Dados processados\n")
        f.write("├── models/                # Modelos treinados e transformadores\n")
        f.write("├── notebooks/             # Jupyter notebooks para análise\n")
        f.write("├── reports/               # Relatórios, gráficos e resultados\n")
        f.write("├── src/                   # Código fonte\n")
        f.write("│   ├── data_preparation.py    # Preparação dos dados\n")
        f.write("│   ├── feature_transformation.py  # Transformação de features\n")
        f.write("│   ├── model_training.py     # Treinamento e avaliação de modelos\n")
        f.write("│   └── utils.py              # Funções utilitárias\n")
        f.write("├── config/                # Arquivos de configuração\n")
        f.write("├── main.py                # Script principal\n")
        f.write("└── README.md              # Este arquivo\n")
        f.write("```\n\n")
        
        f.write("## Requisitos\n\n")
        f.write("- Python 3.8+\n")
        f.write("- pandas\n")
        f.write("- numpy\n")
        f.write("- scikit-learn\n")
        f.write("- xgboost\n")
        f.write("- matplotlib\n")
        f.write("- seaborn\n")
        f.write("- joblib\n\n")
        
        f.write("Você pode instalar todas as dependências com:\n\n")
        f.write("```bash\n")
        f.write("pip install -r requirements.txt\n")
        f.write("```\n\n")
        
        f.write("## Como Usar\n\n")
        f.write("### 1. Preparação\n\n")
        f.write("Coloque os arquivos `fraudTrain.csv` e `fraudTest.csv` no diretório `data/raw/`.\n\n")
        
        f.write("### 2. Execução\n\n")
        f.write("Execute o script principal para processar os dados, treinar e avaliar os modelos:\n\n")
        f.write("```bash\n")
        f.write("python main.py\n")
        f.write("```\n\n")
        
        f.write("Ou execute cada etapa separadamente:\n\n")
        f.write("```bash\n")
        f.write("# Preparação dos dados\n")
        f.write("python src/data_preparation.py\n\n")
        f.write("# Transformação de features\n")
        f.write("python src/feature_transformation.py\n\n")
        f.write("# Treinamento e avaliação de modelos\n")
        f.write("python src/model_training.py\n")
        f.write("```\n\n")
        
        f.write("### 3. Notebooks\n\n")
        f.write("Explore os notebooks no diretório `notebooks/` para análises detalhadas:\n\n")
        f.write("- `01_exploratory_analysis.ipynb`: Análise exploratória dos dados\n")
        f.write("- `02_feature_engineering.ipynb`: Engenharia de features\n")
        f.write("- `03_model_evaluation.ipynb`: Avaliação detalhada dos modelos\n\n")
        
        if results_df is not None:
            f.write("## Resultados\n\n")
            f.write("### Desempenho dos Modelos\n\n")
            f.write("| Modelo | AUC-ROC | Precisão | Recall | F1-Score |\n")
            f.write("|--------|---------|----------|--------|----------|\n")
            
            for _, row in results_df.iterrows():
                f.write(f"| {row['model']} | {row['auc']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |\n")
            
            # Identificar melhor modelo com base no F1-Score
            best_model_idx = results_df['f1'].idxmax()
            best_model_name = results_df.loc[best_model_idx, 'model']
            
            f.write("\n### Melhor Modelo\n\n")
            f.write(f"O modelo **{best_model_name}** apresentou o m
(Content truncated due to size limit. Use line ranges to read in chunks)