#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para aplicação de normalização e one-hot encoding conforme regras específicas.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import sys
import joblib

# Adicionar diretório raiz ao path para importar módulos do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ensure_dir

def apply_transformations(train_df, test_df, categorical_cols, numeric_cols_to_normalize):
    """
    Aplica normalização e one-hot encoding conforme regras específicas
    
    Parameters:
    -----------
    train_df : DataFrame
        DataFrame de treino
    test_df : DataFrame
        DataFrame de teste
    categorical_cols : list
        Lista de colunas categóricas para aplicar one-hot encoding
    numeric_cols_to_normalize : list
        Lista de colunas numéricas para normalizar
        
    Returns:
    --------
    tuple
        (DataFrame de treino transformado, DataFrame de teste transformado, 
         dicionário com transformadores)
    """
    print("\nAplicando normalização e one-hot encoding...")
    
    # Criar cópias para não modificar os originais
    train_transformed = train_df.copy()
    test_transformed = test_df.copy()
    
    # Dicionário para armazenar transformadores
    transformers = {}
    
    # 1. Normalização de colunas numéricas
    if numeric_cols_to_normalize:
        print(f"Normalizando colunas numéricas: {numeric_cols_to_normalize}")
        
        # Inicializar e ajustar o scaler
        scaler = StandardScaler()
        scaler.fit(train_df[numeric_cols_to_normalize])
        
        # Transformar dados de treino e teste
        train_scaled = scaler.transform(train_df[numeric_cols_to_normalize])
        test_scaled = scaler.transform(test_df[numeric_cols_to_normalize])
        
        # Converter arrays para DataFrames
        train_scaled_df = pd.DataFrame(
            train_scaled, 
            columns=[f"{col}_scaled" for col in numeric_cols_to_normalize],
            index=train_df.index
        )
        
        test_scaled_df = pd.DataFrame(
            test_scaled, 
            columns=[f"{col}_scaled" for col in numeric_cols_to_normalize],
            index=test_df.index
        )
        
        # Concatenar com os DataFrames originais
        train_transformed = pd.concat([train_transformed, train_scaled_df], axis=1)
        test_transformed = pd.concat([test_transformed, test_scaled_df], axis=1)
        
        # Armazenar o scaler
        transformers['scaler'] = scaler
    
    # 2. One-hot encoding de colunas categóricas
    if categorical_cols:
        print(f"Aplicando one-hot encoding nas colunas: {categorical_cols}")
        
        # Inicializar e ajustar o encoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(train_df[categorical_cols])
        
        # Transformar dados de treino e teste
        train_encoded = encoder.transform(train_df[categorical_cols])
        test_encoded = encoder.transform(test_df[categorical_cols])
        
        # Obter nomes das colunas codificadas
        encoded_feature_names = []
        for i, col in enumerate(categorical_cols):
            categories = encoder.categories_[i]
            for cat in categories:
                encoded_feature_names.append(f"{col}_{cat}")
        
        # Converter arrays para DataFrames
        train_encoded_df = pd.DataFrame(
            train_encoded, 
            columns=encoded_feature_names,
            index=train_df.index
        )
        
        test_encoded_df = pd.DataFrame(
            test_encoded, 
            columns=encoded_feature_names,
            index=test_df.index
        )
        
        # Concatenar com os DataFrames transformados
        train_transformed = pd.concat([train_transformed, train_encoded_df], axis=1)
        test_transformed = pd.concat([test_transformed, test_encoded_df], axis=1)
        
        # Armazenar o encoder
        transformers['encoder'] = encoder
    
    return train_transformed, test_transformed, transformers

def save_transformers(transformers, output_dir):
    """
    Salva os transformadores para uso posterior
    
    Parameters:
    -----------
    transformers : dict
        Dicionário com transformadores
    output_dir : str
        Diretório para salvar os transformadores
    """
    print("\nSalvando transformadores...")
    
    # Garantir que o diretório existe
    ensure_dir(output_dir)
    
    # Salvar cada transformador
    for name, transformer in transformers.items():
        transformer_path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(transformer, transformer_path)
        print(f"Transformador '{name}' salvo em: {transformer_path}")

def prepare_features_for_modeling(train_df, test_df, categorical_cols, numeric_cols_to_normalize, numeric_cols_no_normalize):
    """
    Prepara as features para modelagem, incluindo seleção de colunas relevantes
    
    Parameters:
    -----------
    train_df : DataFrame
        DataFrame de treino transformado
    test_df : DataFrame
        DataFrame de teste transformado
    categorical_cols : list
        Lista de colunas categóricas que foram codificadas
    numeric_cols_to_normalize : list
        Lista de colunas numéricas que foram normalizadas
    numeric_cols_no_normalize : list
        Lista de colunas numéricas que não foram normalizadas
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test, lista de features)
    """
    print("\nPreparando features para modelagem...")
    
    # Identificar colunas codificadas (one-hot encoding)
    encoded_cols = []
    for col in train_df.columns:
        for cat_col in categorical_cols:
            if col.startswith(f"{cat_col}_"):
                encoded_cols.append(col)
    
    # Identificar colunas normalizadas
    scaled_cols = [f"{col}_scaled" for col in numeric_cols_to_normalize]
    
    # Combinar todas as features para modelagem
    feature_cols = encoded_cols + scaled_cols + numeric_cols_no_normalize
    
    # Remover a coluna alvo se estiver nas features
    if 'is_fraud' in feature_cols:
        feature_cols.remove('is_fraud')
    
    print(f"Total de features para modelagem: {len(feature_cols)}")
    
    # Separar features e target
    X_train = train_df[feature_cols]
    y_train = train_df['is_fraud']
    
    X_test = test_df[feature_cols]
    y_test = test_df['is_fraud']
    
    return X_train, y_train, X_test, y_test, feature_cols

def main():
    """
    Função principal para aplicação de normalização e one-hot encoding
    """
    # Definir caminhos
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    models_dir = os.path.join(base_dir, 'models')
    
    # Caminhos dos arquivos de entrada
    train_path = os.path.join(processed_dir, 'train_processed.csv')
    test_path = os.path.join(processed_dir, 'test_processed.csv')
    
    # Verificar se os arquivos existem
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Erro: Arquivos de entrada não encontrados.")
        print(f"Verifique se os arquivos estão em: {processed_dir}")
        return
    
    # Carregar dados processados
    print("Carregando dados processados...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Definir colunas para transformação
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
    
    # Aplicar transformações
    train_transformed, test_transformed, transformers = apply_transformations(
        train_df, test_df, categorical_cols, numeric_cols_to_normalize
    )
    
    # Salvar transformadores
    save_transformers(transformers, models_dir)
    
    # Preparar features para modelagem
    X_train, y_train, X_test, y_test, feature_cols = prepare_features_for_modeling(
        train_transformed, test_transformed, 
        categorical_cols, numeric_cols_to_normalize, numeric_cols_no_normalize
    )
    
    # Salvar dados transformados
    train_transformed.to_csv(os.path.join(processed_dir, 'train_transformed.csv'), index=False)
    test_transformed.to_csv(os.path.join(processed_dir, 'test_transformed.csv'), index=False)
    
    # Salvar features e targets para modelagem
    X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    
    # Salvar lista de features
    with open(os.path.join(processed_dir, 'feature_cols.txt'), 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    print("\nNormalização e one-hot encoding concluídos com sucesso!")
    print(f"Dados transformados salvos em: {processed_dir}")
    print(f"Transformadores salvos em: {models_dir}")

if __name__ == "__main__":
    main()
