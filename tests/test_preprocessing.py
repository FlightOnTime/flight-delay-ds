"""
Testes Unitários para Módulo de Pré-processamento
"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    criar_features_temporais,
    criar_features_historicas,
    downcast_dataframe
)


class TestCriarFeaturesTemporais:
    """Testes para criar_features_temporais"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """DataFrame de exemplo para testes"""
        return pd.DataFrame({
            'CRSDepTime': [830, 1230, 1815, 2345],
            'DayOfWeek': [1, 6, 7, 3],
            'Month': [1, 4, 7, 12]
        })
    
    def test_dephour_extraction(self, sample_dataframe):
        """Testa extração correta da hora"""
        df_result = criar_features_temporais(sample_dataframe)
        
        assert 'dephour' in df_result.columns
        assert list(df_result['dephour']) == [8, 12, 18, 23]
        assert df_result['dephour'].dtype == 'int8'
    
    def test_is_weekend_flag(self, sample_dataframe):
        """Testa flag de fim de semana"""
        df_result = criar_features_temporais(sample_dataframe)
        
        assert 'is_weekend' in df_result.columns
        assert list(df_result['is_weekend']) == [0, 1, 1, 0]  # 6 e 7 são fim de semana
        assert df_result['is_weekend'].dtype == 'int8'
    
    def test_quarter_calculation(self, sample_dataframe):
        """Testa cálculo de trimestre"""
        df_result = criar_features_temporais(sample_dataframe)
        
        assert 'quarter' in df_result.columns
        assert list(df_result['quarter']) == [1, 2, 3, 4]
        assert df_result['quarter'].dtype == 'int8'
    
    def test_time_of_day_classification(self, sample_dataframe):
        """Testa classificação de período do dia"""
        df_result = criar_features_temporais(sample_dataframe)
        
        assert 'time_of_day' in df_result.columns
        expected = [
            'Morning (6am-12pm)',
            'Afternoon (12pm-6pm)',
            'Evening (6pm-10pm)',
            'Night (10pm-6am)'
        ]
        assert list(df_result['time_of_day']) == expected
    
    def test_dephour_clip_boundaries(self):
        """Testa que dephour é limitado entre 0 e 23"""
        df = pd.DataFrame({
            'CRSDepTime': [0, 100, 2500, 9999],  # Valores extremos
            'DayOfWeek': [1, 1, 1, 1],
            'Month': [1, 1, 1, 1]
        })
        
        df_result = criar_features_temporais(df)
        assert df_result['dephour'].min() >= 0
        assert df_result['dephour'].max() <= 23


class TestCriarFeaturesHistoricas:
    """Testes para criar_features_historicas"""
    
    @pytest.fixture
    def sample_dataframe_temporal(self):
        """DataFrame temporal para testes"""
        return pd.DataFrame({
            'FlightDate': pd.date_range('2024-01-01', periods=5),
            'Origin': ['JFK', 'JFK', 'LAX', 'JFK', 'LAX'],
            'Airline': ['AA', 'AA', 'DL', 'AA', 'DL'],
            'ArrDelay15': [1, 0, 1, 1, 0]
        })
    
    def test_requires_flight_date(self):
        """Testa que exige coluna FlightDate"""
        df = pd.DataFrame({'Origin': ['JFK'], 'Airline': ['AA']})
        
        with pytest.raises(ValueError, match="FlightDate"):
            criar_features_historicas(df)
    
    def test_origin_delay_rate_calculation(self, sample_dataframe_temporal):
        """Testa cálculo de taxa de atraso por origem"""
        df_result = criar_features_historicas(sample_dataframe_temporal)
        
        assert 'origin_delay_rate' in df_result.columns
        assert not df_result['origin_delay_rate'].isna().all()
    
    def test_carrier_delay_rate_calculation(self, sample_dataframe_temporal):
        """Testa cálculo de taxa de atraso por companhia"""
        df_result = criar_features_historicas(sample_dataframe_temporal)
        
        assert 'carrier_delay_rate' in df_result.columns
        assert not df_result['carrier_delay_rate'].isna().all()
    
    def test_origin_traffic_calculation(self, sample_dataframe_temporal):
        """Testa cálculo de tráfego acumulado"""
        df_result = criar_features_historicas(sample_dataframe_temporal)
        
        assert 'origin_traffic' in df_result.columns
        assert df_result['origin_traffic'].dtype == 'int16'
    
    def test_no_data_leakage_shift(self, sample_dataframe_temporal):
        """Testa que primeiro valor é NaN (devido ao shift)"""
        df_result = criar_features_historicas(sample_dataframe_temporal)
        
        # Primeiro valor deve ser preenchido com média global (não NaN após fillna)
        assert not df_result['origin_delay_rate'].iloc[0] == df_result['ArrDelay15'].iloc[0]
    
    def test_dataframe_ordering(self, sample_dataframe_temporal):
        """Testa que DataFrame é ordenado por FlightDate"""
        df_shuffled = sample_dataframe_temporal.sample(frac=1, random_state=42)
        df_result = criar_features_historicas(df_shuffled)
        
        assert df_result['FlightDate'].is_monotonic_increasing


class TestDowncastDataframe:
    """Testes para downcast_dataframe"""
    
    def test_int64_to_int32_conversion(self):
        """Testa conversão de int64 para int32"""
        df = pd.DataFrame({
            'col1': np.array([1, 2, 3], dtype='int64')
        })
        
        df_result = downcast_dataframe(df)
        assert df_result['col1'].dtype == 'int32'
    
    def test_float64_to_float32_conversion(self):
        """Testa conversão de float64 para float32"""
        df = pd.DataFrame({
            'col1': np.array([1.5, 2.5, 3.5], dtype='float64')
        })
        
        df_result = downcast_dataframe(df)
        assert df_result['col1'].dtype == 'float32'
    
    def test_memory_reduction(self):
        """Testa se há redução de memória"""
        df = pd.DataFrame({
            'int_col': np.array(range(1000), dtype='int64'),
            'float_col': np.array(range(1000), dtype='float64')
        })
        
        memory_before = df.memory_usage(deep=True).sum()
        df_result = downcast_dataframe(df)
        memory_after = df_result.memory_usage(deep=True).sum()
        
        assert memory_after < memory_before