#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de utilidades para o projeto de detecção de fraude.
Contém funções auxiliares utilizadas em diferentes partes do projeto.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import math
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula a distância Haversine entre dois pontos em km
    
    Parameters:
    -----------
    lat1, lon1 : float
        Coordenadas do primeiro ponto (latitude, longitude)
    lat2, lon2 : float
        Coordenadas do segundo ponto (latitude, longitude)
        
    Returns:
    --------
    float
        Distância em quilômetros
    """
    # Converter graus para radianos
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Fórmula de Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Raio da Terra em quilômetros
    
    return c * r

def calculate_age(dob_str):
    """
    Calcula a idade a partir da data de nascimento
    
    Parameters:
    -----------
    dob_str : str
        Data de nascimento no formato 'YYYY-MM-DD'
        
    Returns:
    --------
    int
        Idade em anos
    """
    try:
        # Converter string para datetime
        dob = datetime.strptime(dob_str, '%Y-%m-%d')
        
        # Data de referência (usamos a data mais recente do conjunto de dados)
        # Assumindo que a data mais recente é 2023-01-01
        reference_date = datetime(2023, 1, 1)
        
        # Calcular idade
        age = reference_date.year - dob.year
        
        # Ajustar se ainda não fez aniversário neste ano
        if (reference_date.month, reference_date.day) < (dob.month, dob.day):
            age -= 1
            
        return age
    except:
        return np.nan

def plot_confusion_matrix(cm, title='Matriz de Confusão', cmap=plt.cm.Blues, save_path=None):
    """
    Plota a matriz de confusão
    
    Parameters:
    -----------
    cm : array
        Matriz de confusão
    title : str
        Título do gráfico
    cmap : matplotlib colormap
        Mapa de cores
    save_path : str, optional
        Caminho para salvar a figura
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=['Não Fraude', 'Fraude'],
                yticklabels=['Não Fraude', 'Fraude'])
    plt.title(title)
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figura salva em: {save_path}")
    
    plt.show()

def plot_feature_importance(feature_names, importances, title='Importância das Features', save_path=None):
    """
    Plota a importância das features
    
    Parameters:
    -----------
    feature_names : list
        Lista com os nomes das features
    importances : array
        Array com os valores de importância
    title : str
        Título do gráfico
    save_path : str, optional
        Caminho para salvar a figura
    """
    # Criar DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Ordenar por importância
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plotar
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figura salva em: {save_path}")
    
    plt.show()
    
    return importance_df

def plot_roc_curve(fpr, tpr, roc_auc, title='Curva ROC', save_path=None):
    """
    Plota a curva ROC
    
    Parameters:
    -----------
    fpr : array
        Taxa de falsos positivos
    tpr : array
        Taxa de verdadeiros positivos
    roc_auc : float
        Área sob a curva ROC
    title : str
        Título do gráfico
    save_path : str, optional
        Caminho para salvar a figura
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(title)
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figura salva em: {save_path}")
    
    plt.show()

def plot_precision_recall_curve(recall, precision, pr_auc, title='Curva Precision-Recall', save_path=None):
    """
    Plota a curva Precision-Recall
    
    Parameters:
    -----------
    recall : array
        Valores de recall
    precision : array
        Valores de precisão
    pr_auc : float
        Área sob a curva Precision-Recall
    title : str
        Título do gráfico
    save_path : str, optional
        Caminho para salvar a figura
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figura salva em: {save_path}")
    
    plt.show()

def ensure_dir(directory):
    """
    Garante que o diretório existe, criando-o se necessário
    
    Parameters:
    -----------
    directory : str
        Caminho do diretório
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Diretório criado: {directory}")
