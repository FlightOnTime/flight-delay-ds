# 📏 Regras de Validação - FlightOnTime API

```markdown
# 📏 Regras de Validação - FlightOnTime API

Este documento define os limites operacionais (ranges) aceitos pelo modelo preditivo da FlightOnTime API. Valores fora destes intervalos devem retornar erro `400 Bad Request`.

---

## 🔢 Campos Numéricos

### **1. `distance` (Distância do voo em milhas)**
- **Tipo:** `float`
- **Range:** `0.0` a `10000.0` milhas
- **Típico:** `200.0` a `2500.0` milhas
- **Validação Python:**
```python
if not (0.0 <= distance <= 10000.0):
    raise ValidationError("Distance must be between 0 and 10,000 miles")

```

## **2. `origin_delay_rate` (Taxa de atraso do aeroporto de origem)**

* **Tipo:** `float`
* **Range:** `0.0` a `1.0` (0% a 100%)
* **Típico:** `0.10` a `0.30` (10% a 30%)
* **Valor Default:** `0.20`
* **Validação Python:**

```python
if not (0.0 <= origin_delay_rate <= 1.0):
    raise ValidationError("origin_delay_rate must be between 0.0 and 1.0")

```

## **3. `carrier_delay_rate` (Taxa de atraso da companhia)**

* **Tipo:** `float`
* **Range:** `0.0` a `1.0`
* **Típico:** `0.10` a `0.30`
* **Valor Default:** `0.20`
* **Validação Python:**

```python
if not (0.0 <= carrier_delay_rate <= 1.0):
    raise ValidationError("carrier_delay_rate must be between 0.0 and 1.0")

```

### **4. `origin_traffic` (Tráfego acumulado do aeroporto)**

* **Tipo:** `int`
* **Range:** `0` a `100000` voos/mês
* **Típico:** `1000` a `50000` voos/mês
* **Valor Default:** `10000`
* **Validação Python:**

```python
if not (0 <= origin_traffic <= 100000):
    raise ValidationError("origin_traffic must be between 0 and 100,000")

```

### **5. `day_of_week` (Dia da semana)**

* **Tipo:** `int`
* **Range:** `1` a `7` (1=Segunda, 7=Domingo)
* **Validação Python:**

```python
if not (1 <= day_of_week <= 7):
    raise ValidationError("day_of_week must be between 1 (Monday) and 7 (Sunday)")

```

### **6. `crs_dep_time` (Horário programado de partida)**

* **Tipo:** `int`
* **Range:** `0` a `2359` (formato HHMM)
* **Exemplos:** `830` (08:30), `1430` (14:30)
* **Validação Python:**

```python
if not (0 <= crs_dep_time <= 2359):
    raise ValidationError("crs_dep_time must be between 0 and 2359")

# Validação de formato de hora
hours = crs_dep_time // 100
minutes = crs_dep_time % 100
if not (0 <= hours <= 23 and 0 <= minutes <= 59):
    raise ValidationError("crs_dep_time invalid time format (HHMM)")

```

---

## 🔤 Campos Categóricos

## **1. `carrier` (Código IATA da companhia aérea)**

* **Tipo:** `string` (2 letras maiúsculas)
* **Valores Válidos:** `["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9", "G4", "HA"]`

## **2. `origin` (Código IATA do aeroporto de origem)**

* **Tipo:** `string` (3 letras maiúsculas)
* **Valores Válidos:** Ver arquivo `valid_airports.json`.

## **3. `dest` (Código IATA do aeroporto de destino)**

* **Tipo:** `string` (3 letras maiúsculas)
* **Valores Válidos:** Ver arquivo `valid_airports.json`.
* **Nota:** Atualmente, a lista de destinos válidos é a mesma que a de origens.

---

## 📊 Tabela Resumo

| Campo | Tipo | Min | Max | Obrigatório | Default |
| --- | --- | --- | --- | --- | --- |
| `carrier` | string | — | — | ✅ Sim | — |
| `origin` | string | — | — | ✅ Sim | — |
| `dest` | string | — | — | ✅ Sim | — |
| `day_of_week` | int | 1 | 7 | ✅ Sim | — |
| `crs_dep_time` | int | 0 | 2359 | ✅ Sim | — |
| `distance` | float | 0.0 | 10000.0 | ✅ Sim | — |
| `origin_delay_rate` | float | 0.0 | 1.0 | ❌ Não | 0.20 |
| `carrier_delay_rate` | float | 0.0 | 1.0 | ❌ Não | 0.20 |
| `origin_traffic` | int | 0 | 100000 | ❌ Não | 10000 |

---

## 🛡️ Exemplo de Implementação (Pydantic)

```python
from pydantic import BaseModel, Field, validator
import re

class FlightData(BaseModel):
    # Campos obrigatórios
    carrier: str = Field(..., pattern=r'^[A-Z]{2}$')
    origin: str = Field(..., pattern=r'^[A-Z]{3}$')
    dest: str = Field(..., pattern=r'^[A-Z]{3}$')
    day_of_week: int = Field(..., ge=1, le=7)
    crs_dep_time: int = Field(..., ge=0, le=2359)
    distance: float = Field(..., ge=0.0, le=10000.0)
    
    # Campos opcionais
    origin_delay_rate: float = Field(default=0.20, ge=0.0, le=1.0)
    carrier_delay_rate: float = Field(default=0.20, ge=0.0, le=1.0)
    origin_traffic: int = Field(default=10000, ge=0, le=100000)

    @validator('crs_dep_time')
    def validate_time_format(cls, v):
        hours = v // 100
        minutes = v % 100
        if not (0 <= hours <= 23 and 0 <= minutes <= 59):
            raise ValueError(f"Invalid time format: {v} (must be HHMM)")
        return v

```
