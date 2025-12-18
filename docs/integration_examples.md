# 📡 Exemplos de Integração - FlightOnTime API

## ✅ Exemplo 1: Predição de Voo Típico (Sucesso)

### **Request - Exemplo 1**

```
POST http://localhost:8000/v1/predict
Content-Type: application/json

{
  "carrier": "AA",
  "origin": "JFK",
  "dest": "LAX",
  "day_of_week": 3,
  "crs_dep_time": 1430,
  "distance": 2475.0,
  "origin_delay_rate": 0.21,
  "carrier_delay_rate": 0.18,
  "origin_traffic": 45000
}
```

### **Response - Exemplo 1** (200 OK)

```
{
  "previsao": "Atrasado",
  "probabilidade_atraso": 0.73,
  "confianca": "Alta",
  "principais_fatores": [
    "distance: 25.3% de importância",
    "origin_delay_rate: 18.7% de importância",
    "crs_dep_time: 15.2% de importância",
    "carrier_delay_rate": 12.1% de importância",
    "origin_traffic: 8.9% de importância"
  ],
  "recomendacoes": [
    "⏰ Aeroporto JFK tem histórico de 21% de atrasos",
    "🛫 Embarque com 30min de antecedência adicional",
    "📱 Ative notificações de status do voo"
  ]
}
```

---

## ❌ Exemplo 2: Carrier Inválido (Erro 400)

### **Request - Exemplo 2**

```
POST http://localhost:8000/v1/predict
Content-Type: application/json

{
  "carrier": "XX",
  "origin": "JFK",
  "dest": "LAX",
  "day_of_week": 3,
  "crs_dep_time": 1430,
  "distance": 2475.0,
  "origin_delay_rate": 0.21,
  "carrier_delay_rate": 0.18,
  "origin_traffic": 45000
}
```

### **Response - Exemplo 2** (400 Bad Request)

```
{
  "error": {
    "code": "INVALID_CARRIER",
    "message": "Carrier code 'XX' is not valid",
    "field": "carrier",
    "allowed_values": ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9", "G4", "HA"]
  },
  "timestamp": "2025-12-18T16:00:00Z",
  "path": "/v1/predict"
}
```

---

## ❌ Exemplo 3: Distance Fora do Range (Erro 400)

### **Request - Exemplo 3**

```
POST http://localhost:8000/v1/predict
Content-Type: application/json

{
  "carrier": "AA",
  "origin": "JFK",
  "dest": "LAX",
  "day_of_week": 3,
  "crs_dep_time": 1430,
  "distance": 15000.0
}
```

### **Response** (400 Bad Request)

```
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Distance must be between 0 and 10000 miles",
    "field": "distance",
    "provided_value": 15000.0,
    "valid_range": {
      "min": 0.0,
      "max": 10000.0
    }
  },
  "timestamp": "2025-12-18T16:00:00Z",
  "path": "/v1/predict"
}
```
