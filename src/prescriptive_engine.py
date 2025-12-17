"""
Motor Prescritivo - Gera recomendações acionáveis para companhias aéreas
Baseado em Mosqueira et al. (2024)
"""
import numpy as np
from typing import List, Dict


def gerar_output_prescritivo(
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    feature_importance_dict: Dict[str, float],
    top_n: int = 3
) -> List[Dict]:
    """
    Gera output JSON no formato prescritivo.

    Estrutura de saída:
    {
        "indice_voo": int,
        "previsao": "Atrasado" | "Pontual",
        "probabilidade_atraso": float,
        "confianca": "Muito Alta" | "Alta" | "Moderada" | "Baixa",
        "principais_fatores": ["feature: X% de importância", ...],
        "recomendacoes": ["ação1", "ação2", ...]
    }

    Args:
        y_pred: Array com predições (0=Pontual, 1=Atrasado)
        y_proba: Array com probabilidades [0.0 - 1.0]
        feature_importance_dict: {feature_name: importance}
        top_n: Número de features mais importantes para mostrar

    Returns:
        List[Dict]: Lista de predições prescritivas
    """
    outputs = []

    # Top features globais (ordenadas por importância)
    top_features = sorted(
        feature_importance_dict.items(),
        key=lambda x: x[1],  # Corrigido para x[1] para obter o valor da importância
        reverse=True
    )[:top_n]

    for i in range(len(y_pred)):
        pred = y_pred[i]
        prob = y_proba[i]

        # Determinar rótulo e confiança
        if pred == 1:
            previsao = "Atrasado"
            confianca_value = prob
        else:
            previsao = "Pontual"
            confianca_value = 1 - prob

        # Classificar confiança
        if confianca_value >= 0.75:
            confianca = "Muito Alta"
        elif confianca_value >= 0.60:
            confianca = "Alta"
        elif confianca_value >= 0.50:
            confianca = "Moderada"
        else:
            confianca = "Baixa"

        # Principais fatores
        principais_fatores = [
            f"{feat}: {imp * 100:.1f}% de importância"
            for feat, imp in top_features
        ]

        # Recomendações baseadas em predição
        if pred == 1:  # Atrasado
            recomendacoes = [
                "⚠️ Reclassificar voo como potencialmente atrasado",
                "📢 Notificar passageiros com conexões (>2h)",
                "🎯 Antecipar boarding em 10-15 minutos",
                "🚪 Reservar gate alternativo",
                "🔧 Realizar pré-voo com margem de tempo"
            ]
        else:  # Pontual
            recomendacoes = [
                "✅ Manter agendamento normal",
                "🟢 Prioridade operacional normal",
                "⏰ Estimativa: Decolagem no horário"
            ]

        output_json = {
            "indice_voo": i,
            "previsao": previsao,
            "probabilidade_atraso": float(round(prob, 3)),
            "confianca": confianca,
            "principais_fatores": principais_fatores,
            "recomendacoes": recomendacoes
        }

        outputs.append(output_json)

    return outputs
