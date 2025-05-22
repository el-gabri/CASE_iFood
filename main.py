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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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


def caract_df(df: pd.DataFrame):
    """
    Características básicas de um dataframe.

    :param df: Dataframe a ser análisado
    :return: None
    """
    print('Nº linhas: {}\nNº colunas: {}'.format(df.shape[0], df.shape[1]))
    print('Nº linhas duplicadas: {}'.format(df.duplicated().sum()))
    print('\nNº de vazios*:')
    for col in df.columns:
        # n_na = pd.isna(df[col]).sum()
        n_na = df[col].isnull().sum()
        if n_na > 1:
            print('\t{}: {} - {}%'.format(col,
                                          n_na,
                                          round(100 * n_na / df.shape[0], 2)))
    print('(*) Antes do processamento.')
    return


def check_catg(serie: pd.Series, s=7):
    """
    Checa se existe alguma observação fora do padrão, só funciona para séries de números de mesmo tamanho

    :param s:
    :param serie: Série que será checada a procura de valoresfora do padrão.
    :return: None
    """
    cond1 = serie.apply(lambda x: False if pd.isna(x) else True if len(x) != s else False)
    cond2 = serie.apply(
        lambda x: False if pd.isna(x) else True if ''.join([num for num in x if not num.isnumeric()]) else False)

    print(cond1.sum())
    print(cond2.sum())
    return cond1, cond2


def barplot(serie: pd.Series, c='Green'):
    """
    Função automática de criação de sns.barplot.
    :param serie: Série que origininará o gráfico.
    :param c: Cor do gráfico.
    :return: None.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Contabilizando Na's: substitui NaN por 'Faltante'
    serie = serie.replace(np.nan, 'Faltante', regex=True).copy()

    # Gerando dados para plot
    df = serie.value_counts().reset_index()
    # Force column names so that we know exactly what they are:
    # The first column will be the category (named 'index') and the second column is the frequency.
    df.columns = ['index', serie.name]
    # Now we can safely cast types:
    df = df.astype({'index': str, serie.name: int})

    # Separando dados missings
    miss = df[df['index'] == 'Faltante']

    # Criando categoria "Outros"
    df = df[df['index'] != 'Faltante'].reset_index(drop=True)
    size_outros = 0
    if df.shape[0] > 10:
        size_outros = df.shape[0] - 6
        df_top = df.head(5).copy()
        # Sum the rest of the values
        sum_outros = df.drop([0, 1, 2, 3, 4])[serie.name].sum()
        df_outros = pd.DataFrame({'index': ['Outros'], serie.name: [sum_outros]})
        df = pd.concat([df_top, df_outros])

    df = pd.concat([df, miss]).reset_index(drop=True)

    # Display dados (para debug)
    df_display = df.copy()
    df_display.columns = [serie.name, 'Freq']
    if size_outros > 0:
        print("Mostrando as maiores categorias de {}".format(df_display.shape[0] + size_outros))

    # Setup Plot
    palette = {catg: c if (catg != 'Outros') and (catg != 'Faltante') else ('gray' if catg == 'Outros' else 'black')
               for catg in df['index']}
    altura = 4 if df.shape[0] < 16 else int(len(df['index']) / 4)
    plt.figure(figsize=(14, altura))

    # Plot
    sns.barplot(x=serie.name, y='index', data=df, palette=palette)
    for y, x in enumerate(df[serie.name]):
        dif = df[serie.name].max() * 0.005
        percent = 100 * x / df[serie.name].sum()

        # Posição das legendas
        if x > 7 * dif:
            plt.annotate(x, xy=(x - dif, y), ha='right', va='center', color='white')
            plt.annotate('{:.2f}%'.format(percent), xy=(x + dif, y), ha='left', va='center', color='black')
        else:
            plt.annotate('{} - {:.2f}%'.format(x, percent), xy=(x + dif, y), ha='left', va='center', color='black')

    # Layout
    plt.xlim(0, df[serie.name].max() * 1.1)
    plt.title('Frequência da coluna {}'.format(serie.name))
    plt.xlabel('Frequência')
    plt.ylabel(serie.name.title().replace('_', ' '))
    plt.show()


def time_plot(serie: pd.Series, c='Green', stats=True):
    """
    Função de criação automatica de séries temporais e análise.
    :param serie: Série que origininará o gráfico
    :param c: Cor do gráfico
    :param stats: Indicador se traz ou não as estatísticas/plots estatísticos da série temporal
    :return: None
    """

    # Garantir que a série esteja em datetime
    if serie.dtype != 'datetime64[ns]':
        serie = serie.astype(str)
        serie = pd.to_datetime(serie, format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Contabilizando Na's e mostrando o range
    print('Número de missings: {}'.format(pd.isna(serie).sum()))
    print('Range da série: {} - {}'.format(serie.min().strftime('%d/%b/%Y'),
                                           serie.max().strftime('%d/%b/%Y')))

    dias = (serie.max() - serie.min()).days
    if dias // 365 == 1:
        print('\t- {}Ano {}Meses {}dias'.format(dias // 365, (dias % 365) // 30, (dias % 365) % 30))
    else:
        print('\t- {}Anos {}Meses {}dias'.format(dias // 365, (dias % 365) // 30, (dias % 365) % 30))

    # Destrinchando a série em dia, mês e ano
    serie_dia = serie.dt.strftime('%d/%m/%Y').copy()
    serie_mes = serie.dt.strftime('%b/%Y').copy()
    serie_ano = serie.dt.strftime('%Y').copy()

    # Para os dados diários:
    df_dia = serie_dia.value_counts().rename(serie.name)
    df_dia.index.name = "day"  # define um nome diferente para o índice
    df_dia = df_dia.reset_index()
    df_dia['date'] = pd.to_datetime(df_dia['day'], format='%d/%m/%Y')
    df_dia['media movel (30)'] = df_dia[serie.name].rolling(window=30).mean()
    df_dia.sort_values('date', inplace=True)

    # Para os dados mensais:
    df_mes = serie_mes.value_counts().rename(serie.name)
    df_mes.index.name = "month"  # define um nome diferente para o índice
    df_mes = df_mes.reset_index()
    df_mes['date'] = pd.to_datetime(df_mes['month'], format='%b/%Y')
    df_mes.sort_values('date', inplace=True)

    # Para os dados anuais:
    df_ano = serie_ano.value_counts().rename(serie.name)
    df_ano.index.name = "year"  # define um nome diferente para o índice
    df_ano = df_ano.reset_index()
    df_ano['date'] = pd.to_datetime(df_ano['year'], format='%Y')
    df_ano.sort_values('date', inplace=True)

    # Plots
    fig, ax = plt.subplots(3, figsize=(15, 8))

    sns.lineplot(x='date', y=serie.name, color=c, data=df_dia, ax=ax[0])

    sns.barplot(x='month', y=serie.name, color=c, data=df_mes, ax=ax[1])
    for x, y in enumerate(df_mes[serie.name]):
        ax[1].annotate(y, xy=(x, y), ha='center', va='bottom')

    sns.barplot(x='year', y=serie.name, color=c, data=df_ano, ax=ax[2])
    for x, y in enumerate(df_ano[serie.name]):
        ax[2].annotate(y, xy=(x, y), ha='center', va='bottom')

    # Setup dos plots
    ax[0].set_title('Distribuição de {} por dia'.format(serie.name))
    ax[1].set_title('Distribuição de {} por mes'.format(serie.name))
    ax[2].set_title('Distribuição de {} por ano'.format(serie.name))

    ax[0].set_xlabel('Dias')
    ax[1].set_xlabel('Meses')
    ax[2].set_xlabel('Anos')

    ax[0].set_ylabel('Frequência')
    ax[1].set_ylabel('Frequência')
    ax[2].set_ylabel('Frequência')

    ax[0].set_ylim(0, df_dia[serie.name].max() * 1.2)
    ax[1].set_ylim(0, df_mes[serie.name].max() * 1.2)
    ax[2].set_ylim(0, df_ano[serie.name].max() * 1.2)

    plt.tight_layout()
    plt.show()

    if stats:
        # Estacionaridade: Teste Dickey-Fuller
        p_value = sm.tsa.stattools.adfuller(df_dia[serie.name])[1]
        print('\n\n' + 27 * '##' + ' Estatísticas ' + 27 * '##' + '\n')

        # Plots estatísticos
        plt.figure(figsize=(15, 8))
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        ax3 = plt.subplot2grid((2, 2), (1, 1))

        sns.lineplot(x='date', y=serie.name, color=c, data=df_dia, label='Dados', ax=ax1)
        sns.lineplot(x='date', y='media movel (30)', color='firebrick',
                     data=df_dia, label='Média movel (30 dias)', ax=ax1)

        plot_acf(df_dia[serie.name], ax=ax2)
        plot_pacf(df_dia[serie.name], ax=ax3)

        ax1.set_title('Análise da série temporal {}\n Dickey-fuler: p={:.5f}'.format(serie.name, p_value))

        plt.tight_layout()
        plt.show()

    return


def iqr(serie: pd.Series, multiplicador=1.5):
    """
    Análise de outliers da série numérica

    :param serie: Série de valores numéricos
    :param multiplicador: Intervalo do que é considerado aceitavel, sugestão 3.0
    :return: série sem outliers
    """
    # Valores outliers
    # q1, q3 = np.quantile(serie, [0.25, 0.75])
    # IQR = (q3 - q1) * multiplicador
    # limit_lower = q1 - IQR if q1 > IQR else 0
    # limit_upper = q3 + IQR
    factor = multiplicador
    limit_upper = serie.mean() + serie.std() * factor
    limit_lower = serie.mean() - serie.std() * factor

    # Outliers
    outliers = [x for x in serie if (x > limit_upper) | (x < limit_lower)]
    in_limits = [x if (x <= limit_upper) & (x >= limit_lower) else np.nan for x in serie]

    print('Número de outliers (excluídos): {} ({}% do total)'.format(len(outliers),
                                                                     round(len(outliers) * 100 / len(serie),
                                                                           2)))
    print('Número de registros considerados: {}'.format(len(serie) - len(outliers)))

    return pd.Series(in_limits, name=serie.name)


def numeric_plot(serie: pd.Series, c='Green', outliers=True, mult=1.5):
    """
    Análise de dados numéricos.

    :param serie: Série a ser analisada
    :param c: cor do gráfico
    :param outliers: Indicador dos outliers
    :param mult: Intervalo do que é considerado aceitavel, sugestão 1.5 ou 2.5.
    :return: None
    """

    # Outliers
    if outliers:
        serie = iqr(serie.copy(), multiplicador=mult)

    serie = serie.loc[pd.notna(serie)].copy()
    df = serie.describe().reset_index()
    df = pd.concat([df, pd.DataFrame({'index': ['skewness', 'Kurtosis'],
                                      serie.name: [skew(serie), kurtosis(serie)]})])

    df.columns = ['', 'Valor']
    df.set_index('', inplace=True)

    # Plot
    fig, ax = plt.subplots(2, figsize=(15, 6), sharex=True)

    violin = sns.violinplot(x=serie, color=c, inner=None, ax=ax[0])
    plt.setp(violin.collections, alpha=.3)
    sns.boxplot(x=serie, color=c, ax=ax[0])

    sns.histplot(x=serie, color=c, ax=ax[1])

    # Plot layout
    ax[0].set_xlabel('Valor')
    ax[1].set_xlabel('Valor')

    ax[0].set_ylabel('{}'.format(serie.name.title().replace('_', ' ')))
    ax[1].set_ylabel('Frequência')

    ax[0].set_title('Distribuição da variável {}'.format(serie.name.title().replace('_', ' ')))
    ax[1].set_title('Distribuição da variável {}'.format(serie.name.title().replace('_', ' ')))

    plt.tight_layout()
    plt.show()
    return
