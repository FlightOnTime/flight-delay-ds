
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from api.schemas.flights import FlightRequest

app = FastAPI(
    title="FlightOnTime API",
    description="Sistema de Previsão de Atrasos de Voos com ML (Auto-Lookup)",
    version="2.1"
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'randomforest_v7_final.pkl'
ENCODERS_PATH = BASE_DIR / 'models' / 'label_encoders_v7.pkl'
THRESHOLD_PATH = BASE_DIR / 'models' / 'optimal_threshold_v2.txt'
METADATA_PATH = BASE_DIR / 'models' / 'metadata_v7.json'
LOOKUP_PATH = BASE_DIR / 'models' / 'lookup_tables.json'


try:
    print("🔄 Inicializando API v2.1...")
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)

    if os.path.exists(LOOKUP_PATH):
        with open(LOOKUP_PATH, 'r') as f:
            lookup_tables = json.load(f)
        print(f"✅ Lookup Tables carregadas ({len(lookup_tables.get('origin_delay_rate', []))} aeroportos)")
    else:
        print("⚠️ Lookup Tables não encontradas! Usando defaults globais.")
        lookup_tables = {"defaults": {"origin_delay_rate": 0.2, "carrier_delay_rate": 0.2, "origin_traffic": 500}}

    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, 'r') as f:
            OPTIMAL_THRESHOLD = float(f.read().strip())
    else:
        OPTIMAL_THRESHOLD = 0.409

    print("🚀 API PRONTA NA PORTA 8000")

except Exception as e:
    print(f"❌ ERRO CRÍTICO: {e}")
    model = None
    lookup_tables = {}
    OPTIMAL_THRESHOLD = 0.5


def get_time_of_day(h):
    if 6 <= h < 12:
        return 'Morning'
    elif 12 <= h < 18:
        return 'Afternoon'
    elif 18 <= h < 22:
        return 'Evening'
    else:
        return 'Night'

@app.get("/")
async def root():
    return {"ok": True}


@app.get("/health")
async def health():
    
    model_obj = globals().get('model', None)
    encoders_obj = globals().get('encoders', None)
    lookup_obj = globals().get('lookup_tables', None)
    threshold = globals().get('OPTIMAL_THRESHOLD', None)

    model_loaded = model_obj is not None and hasattr(model_obj, "predict_proba")
    encoders_loaded = isinstance(encoders_obj, dict) and len(encoders_obj) > 0
    lookup_loaded = isinstance(lookup_obj, dict) and len(lookup_obj) > 0
    threshold_ok = isinstance(threshold, (float, int))

    origin_count = len(lookup_obj.get('origin_delay_rate', {})) if isinstance(lookup_obj, dict) else 0
    carrier_count = len(lookup_obj.get('carrier_delay_rate', {})) if isinstance(lookup_obj, dict) else 0

    return {
        "ok": bool(model_loaded and encoders_loaded),
        "model": {
            "loaded": bool(model_loaded),
            "has_predict_proba": bool(model_obj is not None and hasattr(model_obj, "predict_proba"))
        },
        "encoders": {
            "loaded": bool(encoders_loaded),
            "count": len(encoders_obj) if isinstance(encoders_obj, dict) else 0
        },
        "lookup_tables": {
            "loaded": bool(lookup_loaded),
            "origin_count": origin_count,
            "carrier_count": carrier_count
        },
        "optimal_threshold": float(threshold) if threshold_ok else None,
        "model_file_exists": MODEL_PATH.exists()
    }
    

@app.post("/predict")
async def predict_flight_delay(request: FlightRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo indisponível")

    try:
        
        try:
            flight_date = datetime.strptime(request.flight_date, "%Y-%m-%d")
            month = flight_date.month
            quarter = (month - 1) // 3 + 1
            hour = int(str(request.crs_dep_time).zfill(4)[:2])
        except ValueError:
            raise HTTPException(status_code=400, detail="Data ou horário inválido")

        defaults = lookup_tables.get("defaults", {})

        origin_rate = lookup_tables.get("origin_delay_rate", {}).get(
            request.origin, defaults.get("origin_delay_rate", 0.195)
        )
        carrier_rate = lookup_tables.get("carrier_delay_rate", {}).get(
            request.airline, defaults.get("carrier_delay_rate", 0.205)
        )
        traffic = lookup_tables.get("origin_traffic", {}).get(
            request.origin, defaults.get("origin_traffic", 450)
        )

        features = {
            'Airline': request.airline,
            'Origin': request.origin,
            'Dest': request.dest,
            'Distance': request.distance,
            'Month': month,
            'DayOfWeek': request.day_of_week,
            'dephour': hour,
            'quarter': quarter,
            'is_weekend': 1 if request.day_of_week >= 6 else 0,
            'time_of_day': get_time_of_day(hour),
            'origin_delay_rate': origin_rate,
            'carrier_delay_rate': carrier_rate,
            'origin_traffic': traffic
        }

        X = pd.DataFrame([features])

        for col in ['Airline', 'Origin', 'Dest', 'time_of_day']:
            if col in encoders:
                val = X.at[0, col]
                X[col] = encoders[col].transform([val])[0] if val in encoders[col].classes_ else -1

        cols_order = [
            'Month', 'DayOfWeek', 'dephour', 'is_weekend', 'quarter',
            'Distance', 'origin_delay_rate', 'carrier_delay_rate', 'origin_traffic',
            'Airline', 'Origin', 'Dest', 'time_of_day'
        ]
        X = X[cols_order]

        proba = model.predict_proba(X)[0][1]
        prediction = 1 if proba >= OPTIMAL_THRESHOLD else 0

        return {
            "prediction": "Atrasado" if prediction == 1 else "Pontual",
            "probability_delay": round(float(proba), 4),
            "recommendation": "Alerta: Alto risco operacional" if prediction == 1 else "Operação normal",
            "internal_metrics": {
                "historical_origin_risk": origin_rate,
                "historical_carrier_risk": carrier_rate
            }
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
