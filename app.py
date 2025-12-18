"""
FlightOnTime API - Sistema de Predição de Atrasos de Voos
Versão: 7.0
"""
from pydantic import BaseModel, Field, validator
import re
from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.model_utils import (load_encoders, load_feature_names, load_metadata,
                             load_model)
# Importar módulos locais
from src.preprocessing import criar_features_temporais
from src.prescriptive_engine import gerar_output_prescritivo

# ============================
# CONFIGURAÇÃO GLOBAL
# ============================
app = FastAPI(
    title="FlightOnTime API",
    description="API de Predição de Atrasos de Voos com Recomendações Prescritivas",
    version="7.0",
    docs_url="/v1/docs",            # Swagger UI
    redoc_url="/v1/redoc",          # ReDoc
    openapi_url="/v1/openapi.json"  # OpenAPI JSON
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
# ============================
# NOVO SCHEMA COM VALIDAÇÕES
# ============================


class FlightInput(BaseModel):
    """
    Schema de entrada - Dados do voo para predição.
    Inclui validações rigorosas conforme regras de negócio.
    """
    # Features categóricas
    Airline: str = Field(...,
                         example="AA",
                         description="Código da companhia aérea (2 letras maiúsculas)")
    Origin: str = Field(..., example="JFK",
                        description="Aeroporto de origem (3 letras maiúsculas)")
    Dest: str = Field(..., example="LAX",
                      description="Aeroporto de destino (3 letras maiúsculas)")

    # Features temporais
    Month: int = Field(..., ge=1, le=12, example=12,
                       description="Mês do voo (1-12)")
    DayOfWeek: int = Field(..., ge=1, le=7, example=2,
                           description="Dia da semana (1=Segunda, 7=Domingo)")
    CRSDepTime: int = Field(...,
                            ge=0,
                            le=2359,
                            example=1830,
                            description="Hora programada de partida (HHMM)")

    # Features numéricas
    Distance: float = Field(...,
                            gt=0,
                            le=10000.0,
                            example=2475.0,
                            description="Distância do voo em milhas (máx 10000)")

    # Features históricas (opcionais)
    origin_delay_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        example=0.21,
        description="Taxa histórica de atraso do aeroporto")
    carrier_delay_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        example=0.18,
        description="Taxa histórica de atraso da companhia")
    origin_traffic: Optional[int] = Field(
        None,
        ge=0,
        le=100000,
        example=150,
        description="Tráfego do aeroporto")

    # --- VALIDAÇÕES PERSONALIZADAS ---

    @validator('Airline')
    def validate_airline(cls, v):
        # Lista de carriers válidos (hardcoded para exemplo, ideal carregar de
        # arquivo)
        VALID_CARRIERS = [
            "AA",
            "DL",
            "UA",
            "WN",
            "B6",
            "AS",
            "NK",
            "F9",
            "G4",
            "HA"]

        if not re.match(r'^[A-Z]{2}$', v):
            raise ValueError("Airline code must be 2 uppercase letters")
        if v not in VALID_CARRIERS:
            raise ValueError(
                f"Invalid carrier '{v}'. Allowed: {VALID_CARRIERS}")
        return v

    @validator('Origin', 'Dest')
    def validate_airports(cls, v):
        if not re.match(r'^[A-Z]{3}$', v):
            raise ValueError(f"Airport code '{v}' must be 3 uppercase letters")
        # Nota: Validação contra lista completa de 362 aeroportos pode ser pesada aqui,
        # mas a validação de formato já barra "XX" ou minúsculas.
        return v

    @validator('CRSDepTime')
    def validate_time_format(cls, v):
        hours = v // 100
        minutes = v % 100
        if not (0 <= hours <= 23 and 0 <= minutes <= 59):
            raise ValueError(f"Invalid time format: {v} (must be HHMM)")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "Airline": "AA",
                "Origin": "JFK",
                "Dest": "LAX",
                "Month": 12,
                "DayOfWeek": 2,
                "CRSDepTime": 1830,
                "Distance": 2475.0
            }
        }


class PredictionOutput(BaseModel):
    """Schema de saída - Predição com recomendações prescritivas"""
    previsao: str = Field(..., description="'Atrasado' ou 'Pontual'")
    probabilidade_atraso: float = Field(...,
                                        description="Probabilidade de atraso (0.0 - 1.0)")
    confianca: str = Field(...,
                           description="'Muito Alta', 'Alta', 'Moderada' ou 'Baixa'")
    principais_fatores: List[str] = Field(...,
                                          description="Top 3 features mais importantes")
    recomendacoes: List[str] = Field(...,
                                     description="Ações operacionais recomendadas")


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
        # Fallback: média global
        df['origin_delay_rate'] = METADATA['metrics']['recall']
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
                    lambda x: x if x in known_classes else ENCODERS[col].classes_[0])
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
@app.get("/v1/health")
def health_check():
    """Health check - Verifica se a API está funcionando"""
    model_loaded = MODEL is not None
    encoders_loaded = ENCODERS is not None

    return {
        "status": "healthy" if (
            model_loaded and encoders_loaded) else "unhealthy",
        "model_loaded": model_loaded,
        "encoders_loaded": encoders_loaded,
        "model_version": METADATA.get(
            "version",
            "v7"),
        "api_version": "7.0",
        "threshold": THRESHOLD,
        "timestamp": datetime.utcnow().isoformat() +
        "Z",
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
        # Probabilidade da classe "Atrasado"
        y_proba = MODEL.predict_proba(X)[:, 1]
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
