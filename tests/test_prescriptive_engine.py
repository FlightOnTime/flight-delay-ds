"""
Testes Unitários para o Motor Prescritivo
"""
import numpy as np
import pytest

from src.prescriptive_engine import gerar_output_prescritivo


class TestGerarOutputPrescritivo:
    """Testes para a função gerar_output_prescritivo"""

    def test_output_atrasado_alta_confianca(self):
        """Testa predição de voo atrasado com alta confiança"""
        # Arrange
        y_pred = np.array([1])  # Atrasado
        y_proba = np.array([0.85])  # Alta probabilidade
        feature_importance = {
            "dephour": 0.273,
            "carrier_delay_rate": 0.141,
            "time_of_day": 0.135
        }

        # Act
        resultado = gerar_output_prescritivo(
            y_pred, y_proba, feature_importance, top_n=3)

        # Assert
        assert len(resultado) == 1
        assert resultado[0]["previsao"] == "Atrasado"
        assert resultado[0]["probabilidade_atraso"] == 0.85
        # CORRIGIDO: 0.85 >= 0.75
        assert resultado[0]["confianca"] == "Muito Alta"
        assert len(resultado[0]["principais_fatores"]) == 3
        assert len(resultado[0]["recomendacoes"]) == 5
        assert "⚠️" in resultado[0]["recomendacoes"][0]

    def test_output_pontual_moderada_confianca(self):
        """Testa predição de voo pontual com confiança moderada"""
        # Arrange
        y_pred = np.array([0])  # Pontual
        y_proba = np.array([0.35])  # Probabilidade baixa de atraso
        feature_importance = {
            "dephour": 0.273,
            "carrier_delay_rate": 0.141,
            "time_of_day": 0.135
        }

        # Act
        resultado = gerar_output_prescritivo(
            y_pred, y_proba, feature_importance, top_n=3)

        # Assert
        assert resultado[0]["previsao"] == "Pontual"
        assert resultado[0]["probabilidade_atraso"] == 0.35
        # CORRIGIDO: 1-0.35=0.65 >= 0.60
        assert resultado[0]["confianca"] == "Alta"
        assert "✅" in resultado[0]["recomendacoes"][0]

    def test_classificacao_confianca(self):
        """Testa classificação correta de níveis de confiança"""
        feature_importance = {"dephour": 0.273}

        # Muito Alta (>= 0.75)
        resultado = gerar_output_prescritivo(
            np.array(
                [1]), np.array(
                [0.80]), feature_importance)
        assert resultado[0]["confianca"] == "Muito Alta"

        # Alta (>= 0.60)
        resultado = gerar_output_prescritivo(
            np.array(
                [1]), np.array(
                [0.65]), feature_importance)
        assert resultado[0]["confianca"] == "Alta"

        # Moderada (>= 0.50)
        resultado = gerar_output_prescritivo(
            np.array(
                [1]), np.array(
                [0.55]), feature_importance)
        assert resultado[0]["confianca"] == "Moderada"

        # Baixa (< 0.50)
        resultado = gerar_output_prescritivo(
            np.array(
                [1]), np.array(
                [0.45]), feature_importance)
        assert resultado[0]["confianca"] == "Baixa"

    def test_multiplas_predicoes(self):
        """Testa predição em lote (múltiplos voos)"""
        # Arrange
        y_pred = np.array([1, 0, 1])
        y_proba = np.array([0.75, 0.30, 0.55])
        feature_importance = {"dephour": 0.273, "Distance": 0.087}

        # Act
        resultado = gerar_output_prescritivo(
            y_pred, y_proba, feature_importance, top_n=2)

        # Assert
        assert len(resultado) == 3
        assert resultado[0]["previsao"] == "Atrasado"
        assert resultado[1]["previsao"] == "Pontual"
        assert resultado[2]["previsao"] == "Atrasado"
        assert all(len(r["principais_fatores"]) == 2 for r in resultado)

    def test_principais_fatores_ordenados(self):
        """Testa se os principais fatores estão ordenados por importância"""
        # Arrange
        y_pred = np.array([1])
        y_proba = np.array([0.60])
        feature_importance = {
            "Distance": 0.087,
            "dephour": 0.273,
            "carrier_delay_rate": 0.141
        }

        # Act
        resultado = gerar_output_prescritivo(
            y_pred, y_proba, feature_importance, top_n=3)

        # Assert
        fatores = resultado[0]["principais_fatores"]
        assert "dephour" in fatores[0]  # Maior importância primeiro
        assert "carrier_delay_rate" in fatores[1]
        assert "Distance" in fatores[2]

    def test_formato_probabilidade(self):
        """Testa se a probabilidade é arredondada corretamente"""
        # Arrange
        y_pred = np.array([1])
        y_proba = np.array([0.558123456])
        feature_importance = {"dephour": 0.273}

        # Act
        resultado = gerar_output_prescritivo(
            y_pred, y_proba, feature_importance)

        # Assert
        assert resultado[0]["probabilidade_atraso"] == 0.558
        assert isinstance(resultado[0]["probabilidade_atraso"], float)


class TestValidacaoEntrada:
    """Testes de validação de entrada"""

    def test_arrays_tamanhos_diferentes_erro(self):
        """Testa erro quando arrays têm tamanhos diferentes"""
        with pytest.raises(IndexError):
            y_pred = np.array([1, 0])
            y_proba = np.array([0.75])  # Tamanho diferente
            feature_importance = {"dephour": 0.273}
            gerar_output_prescritivo(y_pred, y_proba, feature_importance)

    def test_feature_importance_vazio(self):
        """Testa comportamento com feature importance vazio"""
        y_pred = np.array([1])
        y_proba = np.array([0.75])
        feature_importance = {}

        resultado = gerar_output_prescritivo(
            y_pred, y_proba, feature_importance)
        assert resultado[0]["principais_fatores"] == []
