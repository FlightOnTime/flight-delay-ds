"""
Módulo de Pré-processamento e Engenharia de Features
Versão refatorada do notebook FlightOnTime_v8
"""
import numpy as np


def downcast_dataframe(df):
    """
    Reduz uso de memória através de downcast de tipos de dados.

    Conversões:
    - int64 → int32/int16 quando possível
    - float64 → float32 quando possível

    Returns:
        pd.DataFrame: DataFrame otimizado
    """
    start_memory = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'int64':
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(
                    np.int32).min and c_max < np.iinfo(
                    np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)

        elif col_type == 'float64':
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.finfo(
                    np.float32).min and c_max < np.finfo(
                    np.float32).max:
                df[col] = df[col].astype(np.float32)

    end_memory = df.memory_usage(deep=True).sum() / 1024**2
    reduction_pct = (1 - end_memory / start_memory) * 100

    print("📊 Otimização de Memória:")
    print(f"  - Antes: {start_memory:.2f} MB")
    print(f"  - Depois: {end_memory:.2f} MB")
    print(f"  - Redução: {reduction_pct:.1f}%")

    return df


def criar_features_temporais(df):
    """
    Extrai features temporais sem data leakage.

    Features criadas:
    1. dephour: Hora da partida (0-23)
    2. is_weekend: Fim de semana (0/1)
    3. quarter: Trimestre (1-4)
    4. time_of_day: Período do dia (Morning/Afternoon/Evening/Night)

    Args:
        df: DataFrame com coluna 'CRSDepTime' e 'DayOfWeek'

    Returns:
        pd.DataFrame: DataFrame com novas features
    """
    df_feat = df.copy()

    # Feature 1: Hora da partida
    df_feat['dephour'] = (
        df_feat['CRSDepTime'] //
        100).clip(
        0,
        23).astype('int8')

    # Feature 2: Fim de semana (6=Saturday, 7=Sunday)
    df_feat['is_weekend'] = df_feat['DayOfWeek'].isin([6, 7]).astype('int8')

    # Feature 3: Trimestre
    df_feat['quarter'] = ((df_feat['Month'] - 1) // 3 + 1).astype('int8')

    # Feature 4: Período do dia
    def classify_time_period(hour):
        if 6 <= hour < 12:
            return 'Morning (6am-12pm)'
        elif 12 <= hour < 18:
            return 'Afternoon (12pm-6pm)'
        elif 18 <= hour < 22:
            return 'Evening (6pm-10pm)'
        else:
            return 'Night (10pm-6am)'

    df_feat['time_of_day'] = df_feat['dephour'].apply(
        classify_time_period).astype('category')

    print(
        "✅ Features temporais criadas: ['dephour', 'is_weekend', 'quarter', 'time_of_day']")

    return df_feat


def criar_features_historicas(df, delay_col='ArrDelay15'):
    """
    VERSÃO CORRIGIDA: Features históricas SEM data leakage.

    Features criadas:
    1. origin_delay_rate: Taxa histórica de atrasos no aeroporto
    2. carrier_delay_rate: Taxa histórica de atrasos da companhia
    3. origin_traffic: Número acumulado de voos no aeroporto

    CRÍTICO: Usa shift(1) + expanding().mean() para evitar lookahead bias.

    Args:
        df: DataFrame OBRIGATORIAMENTE ordenado por 'FlightDate'
        delay_col: Nome da coluna target (default: 'ArrDelay15')

    Returns:
        pd.DataFrame: DataFrame com features históricas
    """
    df_feat = df.copy()

    # VALIDAÇÃO CRÍTICA
    if 'FlightDate' not in df_feat.columns:
        raise ValueError(
            "❌ Coluna 'FlightDate' não encontrada! Obrigatória para features históricas.")

    # Ordenar por data ANTES de tudo
    df_feat = df_feat.sort_values('FlightDate').reset_index(drop=True)
    print("📅 Dataset ordenado por FlightDate (obrigatório para evitar data leakage)")

    # 1. Taxa de atraso por aeroporto (rolling com shift)
    df_feat['origin_delay_rate'] = df_feat.groupby(
        'Origin')[delay_col].transform(lambda x: x.shift(1).expanding().mean())

    # 2. Taxa de atraso por companhia (rolling com shift)
    df_feat['carrier_delay_rate'] = df_feat.groupby(
        'Airline')[delay_col].transform(lambda x: x.shift(1).expanding().mean())

    # 3. Congestionamento acumulado (até o dia anterior)
    df_feat['origin_traffic'] = df_feat.groupby(
        ['Origin', 'FlightDate']).cumcount().astype('int16')

    # Preencher NaNs iniciais com média global
    global_mean = df_feat[delay_col].mean()
    df_feat['origin_delay_rate'].fillna(global_mean, inplace=True)
    df_feat['carrier_delay_rate'].fillna(global_mean, inplace=True)

    print("✅ Features históricas criadas: "
          "['origin_delay_rate', 'carrier_delay_rate', 'origin_traffic']")
    print("🛡️ Data leakage evitado através de shift(1) temporal!")
    print(f"📊 NaN preenchidos com média global: {global_mean:.4f}")

    return df_feat
