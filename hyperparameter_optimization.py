#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para hiperparametrização de modelos e validação walk-forward.

Este módulo implementa a otimização de hiperparâmetros para cada modelo
e realiza a validação walk-forward para avaliação temporal robusta.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
from tqdm import tqdm

# Bibliotecas de machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
)
import xgboost as xgb

# Configurações de visualização
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style='whitegrid')


def optimize_logistic_regression(X_train, y_train, cv=3, n_jobs=-1):
    """
    Otimiza hiperparâmetros para o modelo de Regressão Logística.
    
    Parameters:
    -----------
    X_train : DataFrame
        Features de treino
    y_train : array
        Target de treino
    cv : int
        Número de folds para validação cruzada
    n_jobs : int
        Número de jobs para paralelização
        
    Returns:
    --------
    best_model : LogisticRegression
        Melhor modelo encontrado
    best_params : dict
        Melhores hiperparâmetros encontrados
    """
    print("Otimizando hiperparâmetros para Regressão Logística...")
    
    # Definir modelo base
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    # Definir grade de hiperparâmetros
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    # Realizar busca em grade
    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring='f1',
        cv=cv,
        n_jobs=n_jobs,
        verbose=1
    )
    
    # Treinar modelo com diferentes combinações de hiperparâmetros
    grid_search.fit(X_train, y_train)
    
    # Obter melhor modelo e hiperparâmetros
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Melhores hiperparâmetros: {best_params}")
    print(f"Melhor F1-Score: {grid_search.best_score_:.4f}")
    
    return best_model, best_params


def optimize_random_forest(X_train, y_train, cv=3, n_jobs=-1):
    """
    Otimiza hiperparâmetros para o modelo Random Forest.
    
    Parameters:
    -----------
    X_train : DataFrame
        Features de treino
    y_train : array
        Target de treino
    cv : int
        Número de folds para validação cruzada
    n_jobs : int
        Número de jobs para paralelização
        
    Returns:
    --------
    best_model : RandomForestClassifier
        Melhor modelo encontrado
    best_params : dict
        Melhores hiperparâmetros encontrados
    """
    print("Otimizando hiperparâmetros para Random Forest...")
    
    # Definir modelo base
    model = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=n_jobs
    )
    
    # Definir grade de hiperparâmetros
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Usar RandomizedSearchCV para otimização (mais eficiente para muitos hiperparâmetros)
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=20,  # Número de combinações a testar
        scoring='f1',
        cv=cv,
        random_state=42,
        n_jobs=n_jobs,
        verbose=1
    )
    
    # Treinar modelo com diferentes combinações de hiperparâmetros
    random_search.fit(X_train, y_train)
    
    # Obter melhor modelo e hiperparâmetros
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print(f"Melhores hiperparâmetros: {best_params}")
    print(f"Melhor F1-Score: {random_search.best_score_:.4f}")
    
    return best_model, best_params


def optimize_gradient_boosting(X_train, y_train, cv=3, n_jobs=-1):
    """
    Otimiza hiperparâmetros para o modelo Gradient Boosted Trees.
    
    Parameters:
    -----------
    X_train : DataFrame
        Features de treino
    y_train : array
        Target de treino
    cv : int
        Número de folds para validação cruzada
    n_jobs : int
        Número de jobs para paralelização
        
    Returns:
    --------
    best_model : GradientBoostingClassifier
        Melhor modelo encontrado
    best_params : dict
        Melhores hiperparâmetros encontrados
    """
    print("Otimizando hiperparâmetros para Gradient Boosted Trees...")
    
    # Definir modelo base
    model = GradientBoostingClassifier(
        random_state=42
    )
    
    # Definir grade de hiperparâmetros
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Usar RandomizedSearchCV para otimização (mais eficiente para muitos hiperparâmetros)
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=20,  # Número de combinações a testar
        scoring='f1',
        cv=cv,
        random_state=42,
        n_jobs=n_jobs,
        verbose=1
    )
    
    # Treinar modelo com diferentes combinações de hiperparâmetros
    random_search.fit(X_train, y_train)
    
    # Obter melhor modelo e hiperparâmetros
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print(f"Melhores hiperparâmetros: {best_params}")
    print(f"Melhor F1-Score: {random_search.best_score_:.4f}")
    
    return best_model, best_params


def optimize_xgboost(X_train, y_train, cv=3, n_jobs=-1):
    """
    Otimiza hiperparâmetros para o modelo XGBoost.
    
    Parameters:
    -----------
    X_train : DataFrame
        Features de treino
    y_train : array
        Target de treino
    cv : int
        Número de folds para validação cruzada
    n_jobs : int
        Número de jobs para paralelização
        
    Returns:
    --------
    best_model : XGBClassifier
        Melhor modelo encontrado
    best_params : dict
        Melhores hiperparâmetros encontrados
    """
    print("Otimizando hiperparâmetros para XGBoost...")
    
    # Calcular scale_pos_weight para lidar com desbalanceamento
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    # Definir modelo base
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=n_jobs
    )
    
    # Definir grade de hiperparâmetros
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Usar RandomizedSearchCV para otimização (mais eficiente para muitos hiperparâmetros)
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=20,  # Número de combinações a testar
        scoring='f1',
        cv=cv,
        random_state=42,
        n_jobs=n_jobs,
        verbose=1
    )
    
    # Treinar modelo com diferentes combinações de hiperparâmetros
    random_search.fit(X_train, y_train)
    
    # Obter melhor modelo e hiperparâmetros
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print(f"Melhores hiperparâmetros: {best_params}")
    print(f"Melhor F1-Score: {random_search.best_score_:.4f}")
    
    return best_model, best_params


def walk_forward_validation(df, feature_cols, target_col='is_fraud', n_splits=5, test_size=0.2):
    """
    Realiza validação walk-forward para avaliação temporal robusta.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame com features e target, ordenado por tempo
    feature_cols : list
        Lista de colunas de features
    target_col : str
        Nome da coluna target
    n_splits : int
        Número de splits para validação
    test_size : float
        Proporção do conjunto de teste
        
    Returns:
    --------
    results : dict
        Dicionário com resultados da validação
    """
    print("Realizando validação walk-forward...")
    
    # Garantir que o DataFrame está ordenado por tempo
    if 'transaction_timestamp' in df.columns:
        df = df.sort_values('transaction_timestamp')
    elif 'unix_time' in df.columns:
        df = df.sort_values('unix_time')
    
    # Extrair features e target
    X = df[feature_cols]
    y = df[target_col]
    
    # Criar splits temporais
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(df) * test_size))
    
    # Inicializar modelos
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, penalty='l2', solver='liblinear', 
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_split=2, 
            min_samples_leaf=1, max_features='sqrt', bootstrap=True, 
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosted Trees': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, 
            min_samples_split=2, min_samples_leaf=1, subsample=1.0, 
            max_features=None, random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, 
            min_child_weight=1, gamma=0, subsample=1.0, 
            colsample_bytree=1.0, objective='binary:logistic', 
            scale_pos_weight=np.sum(y == 0) / np.sum(y == 1), 
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
    }
    
    # Inicializar dicionário para armazenar resultados
    results = {model_name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []} 
               for model_name in models.keys()}
    
    # Realizar validação walk-forward
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nSplit {i+1}/{n_splits}")
        
        # Dividir dados em treino e teste
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Treinar e avaliar cada modelo
        for model_name, model in models.items():
            print(f"Treinando {model_name}...")
            
            # Treinar modelo
            model.fit(X_train, y_train)
            
            # Fazer previsões
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            # Armazenar resultados
            results[model_name]['accuracy'].append(accuracy)
            results[model_name]['precision'].append(precision)
            results[model_name]['recall'].append(recall)
            results[model_name]['f1'].append(f1)
            results[model_name]['auc'].append(auc)
            
            # Imprimir métricas
            print(f"  Acurácia: {accuracy:.4f}")
            print(f"  Precisão: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC-ROC: {auc:.4f}")
    
    # Calcular médias das métricas
    for model_name in results.keys():
        for metric in results[model_name].keys():
            results[model_name][f'mean_{metric}'] = np.mean(results[model_name][metric])
    
    # Imprimir resultados finais
    print("\nResultados finais (média das métricas):")
    for model_name in results.keys():
        print(f"\n{model_name}:")
        print(f"  Acurácia: {results[model_name]['mean_accuracy']:.4f}")
        print(f"  Precisão: {results[model_name]['mean_precision']:.4f}")
        print(f"  Recall: {results[model_name]['mean_recall']:.4f}")
        print(f"  F1-Score: {results[model_name]['mean_f1']:.4f}")
        print(f"  AUC-ROC: {results[model_name]['mean_auc']:.4f}")
    
    return results


def plot_walk_forward_results(results, output_dir=None):
    """
    Plota os resultados da validação walk-forward.
    
    Parameters:
    -----------
    results : dict
        Dicionário com resultados da validação
    output_dir : str
        Diretório para salvar os gráficos
    """
    # Criar diretório para salvar os gráficos
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Métricas para plotar
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Plotar cada métrica
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # Plotar resultados para cada modelo
        for model_name in results.keys():
            plt.plot(range(1, len(results[model_name][metric]) + 1), 
                     results[model_name][metric], 
                     marker='o', 
                     label=f"{model_name} (média: {results[model_name][f'mean_{metric}']:.4f})")
        
        # Configurar gráfico
        plt.xlabel('Split', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f'Resultados da Validação Walk-Forward - {metric.capitalize()}', fontsize=14)
        plt.xticks(range(1, len(results[list(results.keys())[0]][metric]) + 1))
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Salvar gráfico
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f'walk_forward_{metric}.png'), 
                        dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # Plotar comparação das médias das métricas
    plt.figure(figsize=(14, 10))
    
    # Configurar largura das barras
    bar_width = 0.15
    index = np.arange(len(results))
    
    # Plotar barras para cada métrica
    for i, metric in enumerate(metrics):
        values = [results[model_name][f'mean_{metric}'] for model_name in results.keys()]
        plt.bar(index + i*bar_width, values, bar_width, label=metric.upper())
    
    # Configurar eixos e legendas
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.title('Comparação de Métricas por Modelo (Validação Walk-Forward)', fontsize=14)
    plt.xticks(index + bar_width * (len(metrics) - 1) / 2, results.keys())
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for i, metric in enumerate(metrics):
        values = [results[model_name][f'mean_{metric}'] for model_name in results.keys()]
        for j, value in enumerate(values):
            plt.text(j + i*bar_width, value + 0.01, f'{value:.3f}', 
                    ha='center', va='bottom', rotation=90, fontsize=10)
    
    # Salvar gráfico
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'walk_forward_comparison.png'), 
                    dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def save_hyperparameter_results(models_dict, output_dir):
    """
    Salva os resultados da otimização de hiperparâmetros.
    
    Parameters:
    -----------
    models_dict : dict
        Dicionário com modelos otimizados e seus hiperparâmetros
    output_dir : str
        Diretório para salvar os resultados
    """
    # Criar diretório para salvar os resultados
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar resultados em arquivo de texto
    with open(os.path.join(output_dir, 'hyperparameter_optimization_results.txt'), 'w') as f:
        f.write("RESULTADOS DA OTIMIZAÇÃO DE HIPERPARÂMETROS\n")
        f.write("=========================================\n\n")
        
        for model_name, (model, params) in models_dict.items():
            f.write(f"{model_name}:\n")
            f.write("-" * len(model_name) + "\n")
            
            for param, value in params.items():
                f.write(f"  {param}: {value}\n")
            
            f.write("\n")
    
    # Salvar modelos otimizados
    for model_name, (model, _) in models_dict.items():
        joblib.dump(model, os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_optimized.joblib"))
    
    print(f"Resultados da otimização de hiperparâmetros salvos em: {output_dir}")


def main(data_dir, models_dir, reports_dir):
    """
    Função principal para executar otimização de hiperparâmetros e validação walk-forward.
    
    Parameters:
    -----------
    data_dir : str
        Diretório com os dados
    models_dir : str
        Diretório para salvar os modelos
    reports_dir : str
        Diretório para salvar os relatórios
    """
    # Criar diretórios se não existirem
    for directory in [data_dir, models_dir, reports_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Caminhos para os arquivos de dados processados
    processed_dir = os.path.join(data_dir, 'processed')
    train_path = os.path.join(processed_dir, 'train_transformed.csv')
    
    # Verificar se os arquivos existem
    if not os.path.exists(train_path):
        print(f"Erro: Arquivo de dados processados não encontrado em {processed_dir}")
        print("Execute o script de preparação de dados primeiro.")
        return
    
    # Carregar dados
    print("Carregando dados processados...")
    train_df = pd.read_csv(train_path)
    
    # Carregar lista de features
    feature_cols_path = os.path.join(processed_dir, 'feature_cols.txt')
    if os.path.exists(feature_cols_path):
        with open(feature_cols_path, 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]
    else:
        # Identificar colunas de features (todas exceto a target)
        feature_cols = [col for col in train_df.columns if col != 'is_fraud']
    
    # Extrair features e target
    X = train_df[feature_cols]
    y = train_df['is_fraud']
    
    print(f"Conjunto de dados: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"Distribuição da classe alvo: {np.bincount(y)}")
    
    # 1. Otimização de hiperparâmetros para cada modelo
    print("\n" + "="*80)
    print("OTIMIZAÇÃO DE HIPERPARÂMETROS")
    print("="*80)
    
    # Usar TimeSeriesSplit para validação cruzada temporal
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Otimizar cada modelo
    lr_model, lr_params = optimize_logistic_regression(X, y, cv=tscv)
    rf_model, rf_params = optimize_random_forest(X, y, cv=tscv)
    gb_model, gb_params = optimize_gradient_boosting(X, y, cv=tscv)
    xgb_model, xgb_params = optimize_xgboost(X, y, cv=tscv)
    
    # Armazenar modelos otimizados e seus hiperparâmetros
    optimized_models = {
        'Logistic Regression': (lr_model, lr_params),
        'Random Forest': (rf_model, rf_params),
        'Gradient Boosted Trees': (gb_model, gb_params),
        'XGBoost': (xgb_model, xgb_params)
    }
    
    # Salvar resultados da otimização de hiperparâmetros
    save_hyperparameter_results(optimized_models, models_dir)
    
    # 2. Validação walk-forward
    print("\n" + "="*80)
    print("VALIDAÇÃO WALK-FORWARD")
    print("="*80)
    
    # Realizar validação walk-forward
    wf_results = walk_forward_validation(train_df, feature_cols)
    
    # Plotar resultados da validação walk-forward
    plot_walk_forward_results(wf_results, reports_dir)
    
    # Salvar resultados da validação walk-forward
    with open(os.path.join(reports_dir, 'walk_forward_results.txt'), 'w') as f:
        f.write("RESULTADOS DA VALIDAÇÃO WALK-FORWARD\n")
        f.write("==================================\n\n")
        
        for model_name in wf_results.keys():
            f.write(f"{model_name}:\n")
            f.write("-" * len(model_name) + "\n")
            
            f.write(f"  Acurácia: {wf_results[model_name]['mean_accuracy']:.4f}\n")
            f.write(f"  Precisão: {wf_results[model_name]['mean_precision']:.4f}\n")
            f.write(f"  Recall: {wf_results[model_name]['mean_recall']:.4f}\n")
            f.write(f"  F1-Score: {wf_results[model_name]['mean_f1']:.4f}\n")
            f.write(f"  AUC-ROC: {wf_results[model_name]['mean_auc']:.4f}\n\n")
    
    print(f"Resultados da validação walk-forward salvos em: {reports_dir}")
    
    # Identificar melhor modelo com base no F1-Score médio
    best_model_name = max(wf_results.keys(), key=lambda k: wf_results[k]['mean_f1'])
    best_model = optimized_models[best_model_name][0]
    
    print("\n" + "="*80)
    print("MELHOR MODELO")
    print("="*80)
    print(f"Melhor modelo baseado na validação walk-forward: {best_model_name}")
    print(f"F1-Score médio: {wf_results[best_model_name]['mean_f1']:.4f}")
    print(f"AUC-ROC médio: {wf_results[best_model_name]['mean_auc']:.4f}")
    
    # Salvar melhor modelo
    best_model_path = os.path.join(models_dir, 'best_model_walk_forward.joblib')
    joblib.dump(best_model, best_model_path)
    print(f"Melhor modelo salvo em: {best_model_path}")
    
    # Criar resumo do melhor modelo
    summary_path = os.path.join(reports_dir, 'best_model_walk_forward_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("RESUMO DO MELHOR MODELO (VALIDAÇÃO WALK-FORWARD)\n")
        f.write("=============================================\n\n")
        f.write(f"Melhor modelo: {best_model_name}\n")
        f.write(f"AUC-ROC médio: {wf_results[best_model_name]['mean_auc']:.4f}\n")
        f.write(f"Precisão média (classe fraude): {wf_results[best_model_name]['mean_precision']:.4f}\n")
        f.write(f"Recall médio (classe fraude): {wf_results[best_model_name]['mean_recall']:.4f}\n")
        f.write(f"F1-Score médio (classe fraude): {wf_results[best_model_name]['mean_f1']:.4f}\n")
        f.write(f"Acurácia média: {wf_results[best_model_name]['mean_accuracy']:.4f}\n\n")
        
        f.write("Hiperparâmetros otimizados:\n")
        for param, value in optimized_models[best_model_name][1].items():
            f.write(f"  {param}: {value}\n")
    
    print(f"Resumo do melhor modelo salvo em: {summary_path}")


if __name__ == "__main__":
    # Definir caminhos
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    
    # Executar função principal
    main(data_dir, models_dir, reports_dir)
