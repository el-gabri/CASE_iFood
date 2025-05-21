#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para treinamento e avaliação de modelos de machine learning para detecção de fraude.
Inclui implementações de Regressão Logística, Random Forest, Gradient Boosted Trees e XGBoost.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
import xgboost as xgb
import os
import sys
import joblib
import time

# Adicionar diretório raiz ao path para importar módulos do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    plot_confusion_matrix, plot_feature_importance, 
    plot_roc_curve, plot_precision_recall_curve, ensure_dir
)

def load_modeling_data(data_dir):
    """
    Carrega os dados preparados para modelagem
    
    Parameters:
    -----------
    data_dir : str
        Diretório com os dados processados
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test, lista de features)
    """
    print("Carregando dados para modelagem...")
    
    # Carregar features e targets
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    # Carregar lista de features
    with open(os.path.join(data_dir, 'feature_cols.txt'), 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    print(f"Dados carregados: X_train={X_train.shape}, X_test={X_test.shape}")
    
    return X_train, y_train, X_test, y_test, feature_cols

def train_logistic_regression(X_train, y_train):
    """
    Treina um modelo de Regressão Logística
    
    Parameters:
    -----------
    X_train : DataFrame
        Features de treino
    y_train : array
        Target de treino
        
    Returns:
    --------
    LogisticRegression
        Modelo treinado
    """
    print("\nTreinando Regressão Logística...")
    start_time = time.time()
    
    # Inicializar e treinar o modelo
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Tempo de treinamento: {training_time:.2f} segundos")
    
    return model

def train_random_forest(X_train, y_train):
    """
    Treina um modelo Random Forest
    
    Parameters:
    -----------
    X_train : DataFrame
        Features de treino
    y_train : array
        Target de treino
        
    Returns:
    --------
    RandomForestClassifier
        Modelo treinado
    """
    print("\nTreinando Random Forest...")
    start_time = time.time()
    
    # Inicializar e treinar o modelo
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Tempo de treinamento: {training_time:.2f} segundos")
    
    return model

def train_gradient_boosting(X_train, y_train):
    """
    Treina um modelo Gradient Boosted Trees
    
    Parameters:
    -----------
    X_train : DataFrame
        Features de treino
    y_train : array
        Target de treino
        
    Returns:
    --------
    GradientBoostingClassifier
        Modelo treinado
    """
    print("\nTreinando Gradient Boosted Trees...")
    start_time = time.time()
    
    # Inicializar e treinar o modelo
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        max_features=None,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Tempo de treinamento: {training_time:.2f} segundos")
    
    return model

def train_xgboost(X_train, y_train):
    """
    Treina um modelo XGBoost
    
    Parameters:
    -----------
    X_train : DataFrame
        Features de treino
    y_train : array
        Target de treino
        
    Returns:
    --------
    XGBClassifier
        Modelo treinado
    """
    print("\nTreinando XGBoost...")
    start_time = time.time()
    
    # Inicializar e treinar o modelo
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),  # Balanceamento de classes
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Tempo de treinamento: {training_time:.2f} segundos")
    
    return model

def evaluate_model(model, X_test, y_test, model_name, feature_cols=None, output_dir=None):
    """
    Avalia o modelo e retorna métricas
    
    Parameters:
    -----------
    model : object
        Modelo treinado
    X_test : DataFrame
        Features de teste
    y_test : array
        Target de teste
    model_name : str
        Nome do modelo
    feature_cols : list, optional
        Lista de nomes das features
    output_dir : str, optional
        Diretório para salvar visualizações
        
    Returns:
    --------
    dict
        Dicionário com métricas de avaliação
    """
    print(f"\nAvaliando modelo: {model_name}")
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    # Calcular matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Imprimir métricas
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_score:.4f}")
    
    print("\nMatriz de Confusão:")
    print(f"Verdadeiros Positivos (TP): {tp}")
    print(f"Falsos Positivos (FP): {fp}")
    print(f"Verdadeiros Negativos (TN): {tn}")
    print(f"Falsos Negativos (FN): {fn}")
    
    # Visualizações
    if output_dir:
        ensure_dir(output_dir)
        
        # Matriz de confusão
        plot_confusion_matrix(
            cm, 
            title=f'Matriz de Confusão - {model_name}',
            save_path=os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
        )
        
        # Importância das features (para modelos baseados em árvores)
        if hasattr(model, 'feature_importances_') and feature_cols:
            importance_df = plot_feature_importance(
                feature_cols,
                model.feature_importances_,
                title=f'Importância das Features - {model_name}',
                save_path=os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
            )
            
            # Salvar importância das features em CSV
            importance_df.to_csv(
                os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_feature_importance.csv'),
                index=False
            )
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plot_roc_curve(
            fpr, tpr, roc_auc,
            title=f'Curva ROC - {model_name}',
            save_path=os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_roc_curve.png')
        )
        
        # Curva Precision-Recall
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        plot_precision_recall_curve(
            recall_curve, precision_curve, pr_auc,
            title=f'Curva Precision-Recall - {model_name}',
            save_path=os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_pr_curve.png')
        )
    
    # Salvar resultados em dicionário
    results = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_score,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }
    
    return results

def save_model(model, model_name, models_dir):
    """
    Salva o modelo treinado
    
    Parameters:
    -----------
    model : object
        Modelo treinado
    model_name : str
        Nome do modelo
    models_dir : str
        Diretório para salvar o modelo
    """
    # Garantir que o diretório existe
    ensure_dir(models_dir)
    
    # Salvar modelo
    model_path = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}.joblib')
    joblib.dump(model, model_path)
    print(f"Modelo '{model_name}' salvo em: {model_path}")

def main():
    """
    Função principal para treinamento e avaliação de modelos
    """
    # Definir caminhos
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    
    # Carregar dados
    X_train, y_train, X_test, y_test, feature_cols = load_modeling_data(data_dir)
    
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
            output_dir=reports_dir
        )
        results.append(model_results)
        
        # Salvar modelo
        save_model(model, model_name, models_dir)
    
    # Criar DataFrame com resultados
    results_df = pd.DataFrame(results)
    
    # Salvar resultados em CSV
    results_path = os.path.join(reports_dir, 'model_evaluation_results.csv')
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
    summary_path = os.path.join(reports_dir, 'best_model_summary.txt')
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
    print("\nTreinamento e avaliação de modelos concluídos com sucesso!")

if __name__ == "__main__":
    main()
