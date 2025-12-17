"""
FlightOnTime API - Sistema de Predição de Atrasos de Voos
Versão: 7.0
"""
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model_utils import (load_encoders, load_feature_names, load_metadata,
                             load_model)
# Importar módulos locais
from src.preprocessing import (criar_features_historicas,
                               criar_features_temporais)
from src.prescriptive_engine import gerar_output_prescritivo

# ============================
# CONFIGURAÇÃO GLOBAL
# ============================
app = FastAPI(
    title="FlightOnTime API",
    description="API de Predição de Atrasos de Voos com Recomendações Prescritivas",
    version="7.0"
)

# Carregar artefatos uma vez ao iniciar (performance)
print("🔄 Carregando modelo e artefatos...")
MODEL = load_model("models/randomforest_v7_final.pkl")
ENCODERS = load_encoders("models/label_encoders_v7.pkl")
METADATA = load_metadata("models/metadata_v7.json")
FEATURE_NAMES = load_feature_names("models/feature_names_v7.json")
THRESHOLD = METADATA["optimal_threshold"]

# Feature importance do modelo (para output prescritivo)
FEATURE_IMPORTANCE = dict(zip(
    FEATURE_NAMES["todas"],
    MODEL.feature_importances_
))

print(f"✅ API inicializada! Threshold: {THRESHOLD:.4f}")


# ============================
# SCHEMAS DE ENTRADA/SAÍDA
# ============================
class FlightInput(BaseModel):
    """
    Schema de entrada - Dados do voo para predição.
    
    Formato JSON padrão (flexível para adaptação futura).
    Campos obrigatórios baseados nas features do modelo.
    """
    # Features categóricas
    Airline: str = Field(..., example="AA", description="Código da companhia aérea (ex: AA, DL, UA)")
    Origin: str = Field(..., example="JFK", description="Aeroporto de origem (código IATA)")
    Dest: str = Field(..., example="LAX", description="Aeroporto de destino (código IATA)")
    
    # Features temporais
    Month: int = Field(..., ge=1, le=12, example=12, description="Mês do voo (1-12)")
    DayOfWeek: int = Field(..., ge=1, le=7, example=2, description="Dia da semana (1=Segunda, 7=Domingo)")
    CRSDepTime: int = Field(..., ge=0, le=2359, example=1830, description="Hora programada de partida (HHMM)")
    
    # Features numéricas
    Distance: int = Field(..., gt=0, example=2475, description="Distância do voo em milhas")
    
    # Features históricas (opcionais - calculadas internamente se não fornecidas)
    origin_delay_rate: Optional[float] = Field(None, example=0.21, description="Taxa histórica de atraso do aeroporto de origem")
    carrier_delay_rate: Optional[float] = Field(None, example=0.18, description="Taxa histórica de atraso da companhia")
    origin_traffic: Optional[int] = Field(None, example=150, description="Tráfego do aeroporto de origem")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Airline": "AA",
                "Origin": "JFK",
                "Dest": "LAX",
                "Month": 12,
                "DayOfWeek": 2,
                "CRSDepTime": 1830,
                "Distance": 2475
            }
        }


class PredictionOutput(BaseModel):
    """Schema de saída - Predição com recomendações prescritivas"""
    previsao: str = Field(..., description="'Atrasado' ou 'Pontual'")
    probabilidade_atraso: float = Field(..., description="Probabilidade de atraso (0.0 - 1.0)")
    confianca: str = Field(..., description="'Muito Alta', 'Alta', 'Moderada' ou 'Baixa'")
    principais_fatores: List[str] = Field(..., description="Top 3 features mais importantes")
    recomendacoes: List[str] = Field(..., description="Ações operacionais recomendadas")


# ============================
# FUNÇÕES AUXILIARES
# ============================
def processar_features(flight_data: FlightInput) -> pd.DataFrame:
    """
    Processa dados de entrada e cria features necessárias.
    
    Args:
        flight_data: Dados do voo (JSON)
    
    Returns:
        DataFrame com features prontas para predição
    """
    # Converter para DataFrame
    df = pd.DataFrame([flight_data.dict()])
    
    # Criar features temporais
    df = criar_features_temporais(df)
    
    # Usar features históricas fornecidas ou valores padrão
    if flight_data.origin_delay_rate is None:
        df['origin_delay_rate'] = METADATA['metrics']['recall']  # Fallback: média global
    if flight_data.carrier_delay_rate is None:
        df['carrier_delay_rate'] = METADATA['metrics']['recall']
    if flight_data.origin_traffic is None:
        df['origin_traffic'] = 100  # Valor padrão moderado
    
    return df


def aplicar_encoders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica LabelEncoders nas features categóricas.
    
    Args:
        df: DataFrame com features categóricas
    
    Returns:
        DataFrame com features encoded
    """
    df_encoded = df.copy()
    
    for col in FEATURE_NAMES["categoricas"]:
        if col in df_encoded.columns and col in ENCODERS:
            try:
                # Tratar valores desconhecidos (não vistos no treino)
                known_classes = set(ENCODERS[col].classes_)
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: x if x in known_classes else ENCODERS[col].classes_[0]
                )
                df_encoded[col] = ENCODERS[col].transform(df_encoded[col])
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Erro ao encodar coluna '{col}': {str(e)}"
                )
    
    return df_encoded


# ============================
# ENDPOINTS DA API
# ============================
@app.get("/")
def root():
    """Endpoint raiz - Informações da API"""
    return {
        "message": "FlightOnTime API v7.0",
        "status": "online",
        "model_version": METADATA["version"],
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info"
        }
    }


@app.get("/health")
def health_check():
    """Health check - Verifica se a API está funcionando"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "encoders_loaded": ENCODERS is not None,
        "threshold": THRESHOLD
    }


@app.get("/model/info")
def model_info():
    """Retorna informações do modelo treinado"""
    return {
        "version": METADATA["version"],
        "timestamp": METADATA["timestamp"],
        "metrics": METADATA["metrics"],
        "business_metrics": METADATA["business_metrics"],
        "optimal_threshold": THRESHOLD,
        "features": {
            "total": len(FEATURE_NAMES["todas"]),
            "numericas": FEATURE_NAMES["numericas"],
            "categoricas": FEATURE_NAMES["categoricas"]
        }
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(flight_data: FlightInput):
    """
    Endpoint principal - Predição de atraso com recomendações prescritivas.
    
    Args:
        flight_data: Dados do voo (JSON)
    
    Returns:
        Predição com probabilidade e recomendações
    """
    try:
        # 1. Processar features
        df = processar_features(flight_data)
        
        # 2. Aplicar encoders
        df_encoded = aplicar_encoders(df)
        
        # 3. Selecionar apenas features do modelo (na ordem correta)
        X = df_encoded[FEATURE_NAMES["todas"]]
        
        # 4. Fazer predição
        y_proba = MODEL.predict_proba(X)[:, 1]  # Probabilidade da classe "Atrasado"
        y_pred = (y_proba >= THRESHOLD).astype(int)
        
        # 5. Gerar output prescritivo
        output = gerar_output_prescritivo(
            y_pred=y_pred,
            y_proba=y_proba,
            feature_importance_dict=FEATURE_IMPORTANCE,
            top_n=3
        )[0]  # Pegar primeira predição (batch size = 1)
        
        # 6. Formatar resposta
        return PredictionOutput(
            previsao=output["previsao"],
            probabilidade_atraso=output["probabilidade_atraso"],
            confianca=output["confianca"],
            principais_fatores=output["principais_fatores"],
            recomendacoes=output["recomendacoes"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar predição: {str(e)}"
        )


@app.post("/predict/batch")
def predict_batch(flights: List[FlightInput]):
    """
    Predição em lote - Processa múltiplos voos de uma vez.
    
    Args:
        flights: Lista de dados de voos
    
    Returns:
        Lista de predições
    """
    try:
        results = []
        for flight in flights:
            result = predict(flight)
            results.append(result.dict())
        
        return {"predictions": results, "total": len(results)}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar batch: {str(e)}"
        )


# ============================
# EXECUÇÃO LOCAL
# ============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)