#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para preparação e transformação dos dados para o projeto de detecção de fraude.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import math
import os
import sys

# Adicionar diretório raiz ao path para importar módulos do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import haversine_distance, calculate_age, ensure_dir

def load_data(train_path, test_path):
    """
    Carrega os dados de treino e teste
    
    Parameters:
    -----------
    train_path : str
        Caminho para o arquivo de treino
    test_path : str
        Caminho para o arquivo de teste
        
    Returns:
    --------
    tuple
        (DataFrame de treino, DataFrame de teste)
    """
    print("Carregando dados de treino e teste...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Número de linhas no conjunto de treino: {len(train_df)}")
    print(f"Número de linhas no conjunto de teste: {len(test_df)}")
    
    return train_df, test_df

def check_missing_values(train_df, test_df):
    """
    Verifica valores ausentes nos conjuntos de dados
    
    Parameters:
    -----------
    train_df : DataFrame
        DataFrame de treino
    test_df : DataFrame
        DataFrame de teste
        
    Returns:
    --------
    tuple
        (DataFrame com contagem de valores ausentes no treino, DataFrame com contagem de valores ausentes no teste)
    """
    print("\nVerificando valores ausentes...")
    
    # Verificar valores ausentes no conjunto de treino
    train_missing = train_df.isnull().sum()
    train_missing = train_missing[train_missing > 0]
    
    # Verificar valores ausentes no conjunto de teste
    test_missing = test_df.isnull().sum()
    test_missing = test_missing[test_missing > 0]
    
    if len(train_missing) > 0:
        print("\nValores ausentes no conjunto de treino:")
        print(train_missing)
    else:
        print("Não há valores ausentes no conjunto de treino.")
        
    if len(test_missing) > 0:
        print("\nValores ausentes no conjunto de teste:")
        print(test_missing)
    else:
        print("Não há valores ausentes no conjunto de teste.")
    
    return train_missing, test_missing

def feature_engineering(df):
    """
    Realiza feature engineering no DataFrame
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame para aplicar feature engineering
        
    Returns:
    --------
    DataFrame
        DataFrame com novas features
    """
    print("\nRealizando feature engineering...")
    
    # Criar cópia para não modificar o original
    df_new = df.copy()
    
    # 1. Extrair componentes temporais
    print("Extraindo componentes temporais...")
    df_new['transaction_timestamp'] = pd.to_datetime(df_new['trans_date_trans_time'])
    df_new['hour_of_day'] = df_new['transaction_timestamp'].dt.hour
    df_new['day_of_week'] = df_new['transaction_timestamp'].dt.dayofweek + 1  # 1=Segunda, 7=Domingo
    df_new['month'] = df_new['transaction_timestamp'].dt.month
    
    # Converter timestamp para unix time (segundos desde 1970-01-01)
    df_new['unix_time'] = df_new['transaction_timestamp'].astype(int) // 10**9
    
    # 2. Calcular distância entre cliente e comerciante
    print("Calculando distância entre cliente e comerciante...")
    
    # Aplicar função de distância vetorizada
    df_new['distance_km'] = haversine_distance(
        df_new['lat'].values, df_new['long'].values,
        df_new['merch_lat'].values, df_new['merch_long'].values
    )
    
    # 3. Extrair primeiro e último dígito do valor da transação
    print("Extraindo primeiro e último dígito do valor da transação...")
    df_new['amt_str'] = df_new['amt'].astype(str)
    
    # Primeiro dígito (não zero)
    df_new['first_digit'] = df_new['amt_str'].str.replace(r'^0+', '').str[0].astype(int)
    
    # Último dígito
    df_new['last_digit'] = df_new['amt_str'].str[-1].astype(int)
    
    # 4. Calcular velocidade entre transações consecutivas do mesmo cartão
    print("Calculando velocidade entre transações consecutivas...")
    
    # Ordenar transações por cartão e timestamp
    df_new = df_new.sort_values(['cc_num', 'transaction_timestamp'])
    
    # Agrupar por cartão
    grouped = df_new.groupby('cc_num')
    
    # Obter localização e timestamp da transação anterior
    df_new['prev_lat'] = grouped['lat'].shift(1)
    df_new['prev_long'] = grouped['long'].shift(1)
    df_new['prev_unix_time'] = grouped['unix_time'].shift(1)
    
    # Calcular tempo entre transações em horas
    df_new['time_diff_hours'] = (df_new['unix_time'] - df_new['prev_unix_time']) / 3600
    
    # Calcular distância entre transações consecutivas
    mask = (~df_new['prev_lat'].isna()) & (~df_new['prev_long'].isna())
    df_new.loc[mask, 'transaction_distance_km'] = haversine_distance(
        df_new.loc[mask, 'lat'].values, 
        df_new.loc[mask, 'long'].values,
        df_new.loc[mask, 'prev_lat'].values, 
        df_new.loc[mask, 'prev_long'].values
    )
    
    # Calcular velocidade (km/h)
    mask = (mask) & (df_new['time_diff_hours'] > 0)
    df_new.loc[mask, 'transaction_velocity_kmh'] = (
        df_new.loc[mask, 'transaction_distance_km'] / df_new.loc[mask, 'time_diff_hours']
    )
    
    # 5. Flag para velocidades fisicamente impossíveis (> 1000 km/h)
    df_new['impossible_velocity'] = np.where(df_new['transaction_velocity_kmh'] > 1000, 1, 0)
    
    # 6. Calcular desvios do padrão de gastos por cartão
    print("Calculando desvios do padrão de gastos...")
    
    # Calcular média e desvio padrão dos gastos por cartão
    card_stats = df_new.groupby('cc_num')['amt'].agg(['mean', 'std']).reset_index()
    card_stats.columns = ['cc_num', 'avg_amt', 'stddev_amt']
    
    # Juntar com o dataframe original
    df_new = pd.merge(df_new, card_stats, on='cc_num', how='left')
    
    # Calcular z-score do valor da transação
    df_new['amt_zscore'] = np.where(
        (df_new['stddev_amt'].notna()) & (df_new['stddev_amt'] > 0),
        (df_new['amt'] - df_new['avg_amt']) / df_new['stddev_amt'],
        0
    )
    
    # Flag para valores atípicos (|z-score| > 3)
    df_new['amt_is_outlier'] = np.where(np.abs(df_new['amt_zscore']) > 3, 1, 0)
    
    # 7. Calcular idade a partir da data de nascimento
    print("Calculando idade a partir da data de nascimento...")
    df_new['idade'] = df_new['dob'].apply(calculate_age)
    
    return df_new

def identify_columns(df):
    """
    Identifica colunas categóricas e numéricas no DataFrame
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame para identificar colunas
        
    Returns:
    --------
    tuple
        (lista de colunas categóricas, lista de colunas numéricas)
    """
    print("\nIdentificando colunas categóricas e numéricas...")
    
    # Colunas categóricas específicas para one-hot encoding
    categorical_cols = ['gender', 'job', 'city']
    
    # Colunas numéricas para normalização
    numeric_cols_to_normalize = [
        'zip', 'lat', 'long', 'city_pop', 'unix_time', 
        'merch_lat', 'merch_long', 'distance_km', 
        'prev_lat', 'prev_long', 'prev_unix_time', 'time_diff_hours'
    ]
    
    # Colunas numéricas que não precisam de normalização
    numeric_cols_no_normalize = [
        'is_fraud', 'amt', 'hour_of_day', 'day_of_week', 'month', 
        'first_digit', 'last_digit', 'transaction_velocity_kmh', 
        'impossible_velocity', 'avg_amt', 'stddev_amt', 
        'amt_zscore', 'amt_is_outlier', 'idade'
    ]
    
    # Todas as colunas numéricas
    numeric_cols = numeric_cols_to_normalize + numeric_cols_no_normalize
    
    print(f"Colunas categóricas para one-hot encoding: {categorical_cols}")
    print(f"Colunas numéricas para normalização: {numeric_cols_to_normalize}")
    print(f"Colunas numéricas sem normalização: {numeric_cols_no_normalize}")
    
    return categorical_cols, numeric_cols_to_normalize, numeric_cols_no_normalize

def balance_data(df, target_col='is_fraud', ratio=10):
    """
    Balanceia os dados usando undersampling da classe majoritária
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame para balancear
    target_col : str
        Nome da coluna alvo
    ratio : int
        Proporção desejada entre classes (majoritária:minoritária)
        
    Returns:
    --------
    DataFrame
        DataFrame balanceado
    """
    print("\nBalanceando dados...")
    
    # Separar classes
    fraud_df = df[df[target_col] == 1]
    non_fraud_df = df[df[target_col] == 0]
    
    fraud_count = len(fraud_df)
    non_fraud_count = len(non_fraud_df)
    
    print(f"Transações fraudulentas: {fraud_count}")
    print(f"Transações não fraudulentas: {non_fraud_count}")
    print(f"Proporção original: 1:{non_fraud_count/fraud_count:.1f}")
    
    # Definir tamanho da amostra para a classe majoritária
    sample_size = min(fraud_count * ratio, non_fraud_count)
    
    # Realizar undersampling da classe majoritária
    non_fraud_sample = non_fraud_df.sample(n=sample_size, random_state=42)
    
    # Combinar para criar conjunto balanceado
    balanced_df = pd.concat([fraud_df, non_fraud_sample])
    
    print(f"Transações não fraudulentas após undersampling: {len(non_fraud_sample)}")
    print(f"Nova proporção: 1:{len(non_fraud_sample)/fraud_count:.1f}")
    print(f"Total de transações após balanceamento: {len(balanced_df)}")
    
    return balanced_df

def save_processed_data(train_df, test_df, output_dir):
    """
    Salva os dados processados
    
    Parameters:
    -----------
    train_df : DataFrame
        DataFrame de treino processado
    test_df : DataFrame
        DataFrame de teste processado
    output_dir : str
        Diretório para salvar os dados processados
    """
    print("\nSalvando dados processados...")
    
    # Garantir que o diretório existe
    ensure_dir(output_dir)
    
    # Salvar dados
    train_path = os.path.join(output_dir, 'train_processed.csv')
    test_path = os.path.join(output_dir, 'test_processed.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Dados de treino salvos em: {train_path}")
    print(f"Dados de teste salvos em: {test_path}")

def main():
    """
    Função principal para preparação e transformação dos dados
    """
    # Definir caminhos
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    
    # Caminhos dos arquivos de entrada
    train_path = os.path.join(data_dir, 'raw', 'fraudTrain.csv')
    test_path = os.path.join(data_dir, 'raw', 'fraudTest.csv')
    
    # Verificar se os arquivos existem
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Erro: Arquivos de entrada não encontrados.")
        print(f"Verifique se os arquivos estão em: {os.path.join(data_dir, 'raw')}")
        return
    
    # Carregar dados
    train_df, test_df = load_data(train_path, test_path)
    
    # Verificar valores ausentes
    check_missing_values(train_df, test_df)
    
    # Realizar feature engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    # Identificar colunas
    categorical_cols, numeric_cols_to_normalize, numeric_cols_no_normalize = identify_columns(train_df)
    
    # Balancear dados de treino
    train_df_balanced = balance_data(train_df)
    
    # Salvar dados processados
    save_processed_data(train_df_balanced, test_df, processed_dir)
    
    print("\nPreparação e transformação dos dados concluída com sucesso!")

if __name__ == "__main__":
    main()
