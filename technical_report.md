# 📊 Relatório Técnico - FlightOnTime

## 1. Resumo Executivo

**Objetivo**: Desenvolver um modelo de Machine Learning para prever atrasos de voos domésticos nos EUA com pelo menos 60% de recall.

**Resultado**: Modelo Random Forest com **ROC-AUC de 0.663** e **Recall de 64.1%**, superando a meta estabelecida.

---

## 2. Metodologia Detalhada

### 2.1 Coleta e Preparação dos Dados

**Dataset Original**:
- 11,408,131 voos (Jan/2023 - Dez/2024)
- 21 colunas brutas do BTS
- Taxa de atraso: 20.4% (threshold 15 minutos)

**Limpeza de Dados**:
1. Remoção de voos cancelados/desviados
2. Filtragem de aeroportos principais (top 53)
3. Tratamento de valores inválidos em CRS_DEP_TIME
4. Ordenação temporal para evitar data leakage

### 2.2 Feature Engineering

**Features Temporais** (6):
- `dep_hour`: Hora da partida (0-23)
- `DAY_OF_WEEK`: Dia da semana (1-7)
- `is_weekend`: Flag final de semana
- `MONTH`: Mês (1-12)
- `quarter`: Trimestre (1-4)
- `time_of_day`: Período (Morning/Afternoon/Evening/Night)

**Features de Rota** (4):
- `route`: Origem_Destino
- `route_frequency`: Frequência acumulada da rota
- `DISTANCE`: Distância em milhas
- `distance_category`: Short/Medium/Long

**Features Históricas** (4):
- `origin_delay_rate`: Taxa de atraso do aeroporto (mês anterior)
- `carrier_delay_rate`: Taxa de atraso da companhia (mês anterior)
- `origin_traffic`: Volume acumulado do aeroporto
- `carrier`: Código da companhia aérea

**Features Categóricas** (2):
- `Origin`: Aeroporto de origem
- `Dest`: Aeroporto de destino

**Total**: 16 features preditivas

### 2.3 Split Temporal

Para evitar data leakage, utilizamos split temporal:
- **Treino**: 9,126,504 voos (80%) - Jan/2023 até Ago/2024
- **Teste**: 2,281,627 voos (20%) - Ago/2024 até Dez/2024

Rationale: Simula cenário real onde modelo prevê futuro usando apenas passado.

### 2.4 Modelagem

**Modelos Testados**:
1. Logistic Regression (baseline)
2. Random Forest (modelo final)

**Hiperparâmetros do Random Forest**:
- `n_estimators`: 100
- `max_depth`: 15
- `min_samples_split`: 100
- `min_samples_leaf`: 50
- `class_weight`: 'balanced'
- `random_state`: 42

**Justificativa**: Random Forest foi escolhido por:
- Capacidade de capturar relações não-lineares
- Robustez a outliers
- Importância de features interpretável
- Não requer escalonamento de variáveis

---

## 3. Resultados Detalhados

### 3.1 Comparação de Modelos

| Modelo | ROC-AUC | Accuracy | Precision | Recall | F1 |
|--------|---------|----------|-----------|--------|-----|
| Logistic Regression | 0.587 | 58.0% | 20.3% | 54.0% | 0.295 |
| **Random Forest** | **0.663** | **60.2%** | **23.5%** | **64.1%** | **0.344** |

**Melhoria**: +12.9% em ROC-AUC, +18.7% em Recall

### 3.2 Otimização de Threshold

**Threshold Padrão (0.50)**:
- Precision: 27.2%
- Recall: 41.7%
- F1: 0.329

**Threshold Otimizado (0.421)**:
- Precision: 23.5% (-13.6%)
- Recall: 64.1% (+53.7%) ✅
- F1: 0.344 (+4.6%) ✅

**Trade-off**: Sacrificamos precision para aumentar recall (objetivo do projeto).

### 3.3 Análise de Erros

**Matriz de Confusão**:
- True Negatives: 1,134,451 (59.4%)
- False Positives: 775,326 (40.6%)
- False Negatives: 133,393 (35.9%)
- True Positives: 238,457 (64.1%) ✅

**Custos de Erro**:
- **Falso Negativo (FN)**: Alto impacto - Passageiro não é alertado e voo atrasa
- **Falso Positivo (FP)**: Baixo impacto - Passageiro é alertado, mas voo sai no horário

**Estratégia**: Minimizar FN em detrimento de FP (aceitável no contexto).

---

## 4. Feature Importance

| Rank | Feature | Importância | Interpretação |
|------|---------|-------------|---------------|
| 1 | dep_hour | 29.1% | Voos noturnos/madrugada têm maior risco |
| 2 | carrier_delay_rate | 15.3% | Histórico da companhia é preditor forte |
| 3 | time_of_day | 14.4% | Efeito cascata ao longo do dia |
| 4 | origin_delay_rate | 14.3% | Aeroportos problemáticos (JFK, ORD) |
| 5 | origin_traffic | 4.6% | Congestionamento afeta operações |

**Insight**: Fatores temporais (hora + período) representam **43.5%** da importância total.

---

## 5. Validação Cruzada

Não foi realizada validação cruzada tradicional devido ao caráter temporal dos dados. Split temporal foi usado para simular deployment real.

**Alternativa Futura**: Time Series Cross-Validation com expanding window.

---

## 6. Limitações

### 6.1 Técnicas
- **Sem dados climáticos em tempo real**: Limitado a padrões históricos
- **Threshold fixo**: Não adapta por contexto (feriados, eventos especiais)
- **Precision baixa**: 76% dos alertas são falsos positivos

### 6.2 Dados
- **Apenas voos domésticos**: Não generaliza para internacional
- **53 aeroportos**: Cobertura parcial (>300 aeroportos nos EUA)
- **Período limitado**: 2 anos pode não capturar sazonalidade de longo prazo

---

## 7. Recomendações

### 7.1 Curto Prazo (1-2 meses)
1. **Deploy em API REST** (FastAPI + Docker)
2. **Dashboard de monitoramento** (Streamlit)
3. **A/B testing** com diferentes thresholds por contexto

### 7.2 Médio Prazo (3-6 meses)
1. **Adicionar features climáticas** (Open-Meteo API)
2. **Ensemble de modelos** (RF + XGBoost + LightGBM)
3. **Explicabilidade por predição** (SHAP values)

### 7.3 Longo Prazo (6-12 meses)
1. **Deep Learning** (LSTM para sequências temporais)
2. **Multi-task learning** (prever duração do atraso)
3. **Integração com sistemas de companhias aéreas**

---

## 8. Conclusão

O modelo FlightOnTime atingiu **ROC-AUC de 0.663** e **Recall de 64.1%**, superando a meta de 60%. Com threshold otimizado, detecta **2 em cada 3 voos atrasados**, fornecendo valor significativo para passageiros e companhias aéreas.

**Impacto Esperado**:
- ✈️ Redução de conexões perdidas
- 💰 Economia em custos operacionais
- 😊 Melhoria na satisfação de passageiros

---

**Data**: Dezembro 2024  
**Versão**: 1.0  
**Status**: Produção (MVP)
