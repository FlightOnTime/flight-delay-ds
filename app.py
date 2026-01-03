# ========================================
# FLIGHTONTIME API v2.0
# Sistema de Previsão de Atrasos de Voos
# ========================================

import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# ========================================
# INICIALIZAR FASTAPI
# ========================================
app = FastAPI(
    title="FlightOnTime API",
    description="Sistema de Previsão de Atrasos de Voos com ML",
    version="2.0"
)

# ========================================
# CONFIGURAÇÃO DE PATHS
# ========================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'randomforest_v7_final.pkl'
ENCODERS_PATH = BASE_DIR / 'models' / 'label_encoders_v7.pkl'
THRESHOLD_PATH = BASE_DIR / 'models' / 'optimal_threshold_v2.txt'
METADATA_PATH = BASE_DIR / 'models' / 'metadata_v7.json'

# ========================================
# CARREGAR MODELO E ARTEFATOS
# ========================================
try:
    print("🔄 Carregando modelo e artefatos...")

    # Carregar modelo
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modelo carregado: {type(model).__name__}")

    # Carregar encoders
    encoders = joblib.load(ENCODERS_PATH)
    print(f"✅ Encoders carregados: {len(encoders)} variáveis categóricas")

    # Carregar threshold otimizado
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, 'r') as f:
            OPTIMAL_THRESHOLD = float(f.read().strip())
        print(f"✅ Threshold otimizado carregado: {OPTIMAL_THRESHOLD:.3f}")
    else:
        OPTIMAL_THRESHOLD = 0.409  # Fallback
        print(f"⚠️ Usando threshold default: {OPTIMAL_THRESHOLD:.3f}")

    # Carregar metadados (opcional)
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        version = metadata.get('version', 'N/A')
        roc_auc = metadata.get('metrics', {}).get('roc_auc', 'N/A')
        print(f"📊 Modelo v{version} | ROC-AUC: {roc_auc}")
    else:
        print("⚠️ Arquivo de metadados não encontrado.")

    print("=" * 60)
    print("🚀 API PRONTA NA PORTA 8000")
    print("=" * 60)

except Exception as e:
    print(f"❌ ERRO CRÍTICO ao inicializar: {e}")
    # Não damos raise aqui para permitir que a API suba e reporte erro no /health,
    # mas em produção o ideal seria falhar o deploy.
    model = None
    encoders = {}
    OPTIMAL_THRESHOLD = 0.5
    metadata = {}


# ========================================
# SCHEMAS
# ========================================
class FlightRequest(BaseModel):
    airline: str
    origin: str
    dest: str
    distance: float
    day_of_week: int
    flight_date: str
    crs_dep_time: int
    origin_delay_rate: float
    carrier_delay_rate: float
    origin_traffic: int

    @field_validator('origin_delay_rate', 'carrier_delay_rate')
    @classmethod
    def validate_rates(cls, v, info):
        if not 0 <= v <= 1:
            raise ValueError(f'{info.field_name} deve estar entre 0 e 1')
        return v

    @field_validator('distance')
    @classmethod
    def validate_distance(cls, v):
        if v <= 0:
            raise ValueError('Distance deve ser positiva')
        return v

    @field_validator('day_of_week')
    @classmethod
    def validate_day(cls, v):
        if not 1 <= v <= 7:
            raise ValueError('DayOfWeek deve estar entre 1 e 7')
        return v


# ========================================
# FUNÇÕES AUXILIARES
# ========================================
def get_time_of_day(h):
    if 6 <= h < 12:
        return 'Morning'
    elif 12 <= h < 18:
        return 'Afternoon'
    elif 18 <= h < 22:
        return 'Evening'
    else:
        return 'Night'


# ========================================
# ENDPOINTS
# ========================================
@app.post("/predict")
def predict_flight_delay(request: FlightRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado corretamente.")

    try:
        # 1. Tratamento de Data e Hora
        try:
            flight_date = datetime.strptime(request.flight_date, "%Y-%m-%d")
            month = flight_date.month
            quarter = (month - 1) // 3 + 1

            dep_time_str = str(request.crs_dep_time).zfill(4)
            hour = int(dep_time_str[:2])
        except ValueError:
            raise HTTPException(status_code=400, detail="Data ou horário inválido.")

        is_weekend = 1 if request.day_of_week >= 6 else 0
        time_of_day = get_time_of_day(hour)

        # 2. Construir Dicionário de Features (Voltando para 'dephour')
        features_dict = {
            'Airline': request.airline,
            'Origin': request.origin,
            'Dest': request.dest,
            'Distance': request.distance,
            'Month': month,
            'DayOfWeek': request.day_of_week,
            'dephour': hour,  # <--- Voltamos para o nome que o erro exigiu
            'quarter': quarter,
            'is_weekend': is_weekend,
            'time_of_day': time_of_day,
            'origin_delay_rate': request.origin_delay_rate,
            'carrier_delay_rate': request.carrier_delay_rate,
            'origin_traffic': request.origin_traffic
        }

        X = pd.DataFrame([features_dict])

        # Ordem EXATA conforme feature_names_v7.json
        cols_order = [
            'Month', 'DayOfWeek', 'dephour', 'is_weekend', 'quarter',
            'Distance', 'origin_delay_rate', 'carrier_delay_rate', 'origin_traffic',
            'Airline', 'Origin', 'Dest', 'time_of_day'
        ]
        # Garante que o DataFrame tenha as colunas nesta ordem antes da predição
        X = X[cols_order]

        # Codificação segura para evitar Erro 500 com novos nomes
        categorical_cols = ['Airline', 'Origin', 'Dest', 'time_of_day']
        for col in categorical_cols:
            if col in encoders:
                val = X.at[0, col]
                # Verifica se o valor existe no treinamento
                if val in encoders[col].classes_:
                    X[col] = encoders[col].transform([val])[0]
                else:
                    # Valor padrão para dados desconhecidos
                    X[col] = -1

        # Predição
        probability = model.predict_proba(X)[0][1]

        # Aplicar threshold otimizado
        prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0
        result = "Atrasado" if prediction == 1 else "Pontual"

        return {
            "prediction": result,
            "probability_delay": round(float(probability), 4),
            "features_used": features_dict,
            "threshold_used": OPTIMAL_THRESHOLD
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {str(e)}")


@app.get("/health")
async def health_check():
    """Verifica se a API está funcionando"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": OPTIMAL_THRESHOLD,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
async def model_info():
    """Retorna informações sobre o modelo em uso"""
    info = {
        "model_type": type(model).__name__ if model else "None",
        "threshold": OPTIMAL_THRESHOLD,
        "threshold_rationale": "Otimizado para custo-benefício (FN=$200, FP=$50)",
        "expected_metrics": {
            "recall": "38.3%",
            "precision": "23.9%",
            "accuracy": "69.9%",
            "roc_auc": "0.625"
        },
        "prediction_distribution": {
            "expected_delayed": "~26%",
            "expected_on_time": "~74%"
        }
    }

    # Adicionar metadados se existirem
    if metadata:
        info['model_version'] = metadata.get('version', 'N/A')
        info['trained_date'] = metadata.get('timestamp', 'N/A')

    return info


@app.get("/")
async def root():
    return {
        "message": "FlightOnTime API v2.0",
        "status": "operational",
        "docs": "http://127.0.0.1:8000/docs"
    }


if __name__ == "__main__":
    # MUDANÇA IMPORTANTE: Porta 8000 para evitar conflito com Spring Boot/Tomcat na 8080
    uvicorn.run(app, host="0.0.0.0", port=8000)