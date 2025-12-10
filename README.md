# ✈️ FlightOnTime - Predição de Atrasos de Voos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Sistema de Machine Learning para prever atrasos de voos domésticos nos EUA usando dados históricos de 2023-2024.

---

## 📊 **Visão Geral do Projeto**

**FlightOnTime** é um modelo preditivo que analisa **11.4 milhões de voos** e prevê atrasos com **66.3% de ROC-AUC** e **64.1% de Recall**, permitindo que passageiros e companhias aéreas se preparem para possíveis atrasos.

### **Principais Resultados**
- ✅ **ROC-AUC: 0.663** - Boa capacidade de discriminação
- ✅ **Recall: 64.1%** - Detecta 2 em cada 3 voos atrasados
- ✅ **F1-Score: 0.344** - Equilíbrio entre precisão e cobertura
- ✅ **Dataset: 11.4M voos** - 2 anos de dados (2023-2024)
- ✅ **16 features** - Engenharia de features sem data leakage

---

## 🎯 **Problema de Negócio**

Atrasos de voos custam **bilhões de dólares** anualmente para companhias aéreas e passageiros:
- 💰 **US$ 33 bilhões/ano** em custos para a indústria (FAA, 2023)
- ⏱️ **~20% dos voos** atrasam mais de 15 minutos
- 😤 **Frustração de passageiros** e perda de conexões

**Solução**: Prever atrasos com antecedência para:
- ✈️ Companhias: Realocar recursos e otimizar operações
- 👥 Passageiros: Planejar melhor e evitar conexões arriscadas

---

## 🗂️ **Estrutura do Repositório**

\`\`\`
FlightOnTime/
├── README.md                          # Este arquivo
├── FlightOnTime_MVP.ipynb             # Notebook principal (Colab)
├── requirements.txt                   # Dependências Python
├── models/
│   ├── random_forest_full_model.pkl   # Modelo treinado (196 MB)
│   ├── label_encoders.pkl             # Encoders categóricos
│   └── optimal_threshold.txt          # Threshold otimizado (0.421)
├── visualizations/
│   ├── roc_curve.png                  # Curva ROC
│   ├── precision_recall_curve.png     # Curva Precision-Recall
│   ├── feature_importance.png         # Top 10 features
│   ├── confusion_matrix.png           # Matriz de confusão
│   └── models_comparison.png          # Comparação de modelos
├── data/
│   └── flight_data_with_features.parquet  # Dataset processado (186 MB)
└── docs/
    └── technical_report.md            # Relatório técnico detalhado
\`\`\`

---

## 🚀 **Como Usar**

### **1. Instalação**

\`\`\`bash
# Clonar repositório
git clone https://github.com/FlightOnTime/flight-delay-ds.git
cd FlightOnTime

# Instalar dependências
pip install -r requirements.txt
\`\`\`

### **2. Fazer Predições**

\`\`\`python
import joblib
import pandas as pd

# Carregar modelo e encoders
model = joblib.load('models/random_forest_full_model.pkl')
encoders = joblib.load('models/label_encoders.pkl')

# Threshold otimizado
OPTIMAL_THRESHOLD = 0.421

# Dados de exemplo (novo voo)
new_flight = pd.DataFrame({
    'carrier': ['AA'],
    'Origin': ['JFK'],
    'Dest': ['LAX'],
    'dep_hour': [18],
    'DAY_OF_WEEK': [5],
    'is_weekend': [0],
    'MONTH': [12],
    'quarter': [4],
    'time_of_day': ['Evening'],
    'DISTANCE': [2475],
    'distance_category': ['Long'],
    'route_frequency': [1500],
    'origin_delay_rate': [0.25],
    'origin_traffic': [50000],
    'carrier_delay_rate': [0.22]
})

# Aplicar encoding
for col in ['carrier', 'Origin', 'Dest', 'time_of_day', 'distance_category']:
    new_flight[col] = encoders[col].transform(new_flight[col])

# Predição
proba = model.predict_proba(new_flight)[0, 1]
prediction = 'ATRASADO' if proba >= OPTIMAL_THRESHOLD else 'PONTUAL'

print(f"Probabilidade de atraso: {proba*100:.1f}%")
print(f"Predição: {prediction}")
\`\`\`

**Saída esperada:**
\`\`\`
Probabilidade de atraso: 45.2%
Predição: ATRASADO
\`\`\`

---

## 📊 **Metodologia**

### **1. Coleta de Dados**
- **Fonte**: [Bureau of Transportation Statistics (BTS)](https://www.transtats.bts.gov/)
- **Período**: Janeiro 2023 - Dezembro 2024
- **Volume**: 11.4 milhões de voos
- **Cobertura**: 53 aeroportos e 10 companhias principais

### **2. Feature Engineering**
Criamos **16 features preditivas** sem data leakage:

| Categoria | Features |
|-----------|----------|
| **Temporais** | dep_hour, DAY_OF_WEEK, is_weekend, MONTH, quarter, time_of_day |
| **Rotas** | route, route_frequency, DISTANCE, distance_category |
| **Aeroportos** | Origin, Dest, origin_delay_rate, origin_traffic |
| **Companhias** | carrier, carrier_delay_rate |

**Destaques**:
- ✅ **Rolling window de 7 dias** para taxas históricas
- ✅ **Split temporal** (80% treino, 20% teste)
- ✅ **Nenhuma informação futura** usada

### **3. Modelagem**
- **Algoritmo**: Random Forest (100 árvores)
- **Tratamento de desbalanceamento**: `class_weight='balanced'`
- **Otimização de threshold**: 0.421 (vs 0.50 padrão)
- **Dados de treino**: 9.1M voos
- **Tempo de treino**: 35 minutos

---

## 📈 **Resultados**

### **Métricas Finais**

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **ROC-AUC** | 0.663 | Boa discriminação entre classes |
| **Accuracy** | 60.2% | 6 em 10 predições corretas |
| **Precision** | 23.5% | 1 em 4 alertas é verdadeiro |
| **Recall** | 64.1% | Detecta 64% dos atrasos reais |
| **F1-Score** | 0.344 | Equilíbrio precision-recall |

### **Matriz de Confusão**

|  | Predito Pontual | Predito Atrasado |
|---|-----------------|------------------|
| **Real Pontual** | 1,134,451 (59.4%) | 775,326 (40.6%) |
| **Real Atrasado** | 133,393 (35.9%) | 238,457 (64.1%) |

**Interpretação**:
- ✅ **238k atrasos detectados** (True Positives)
- ⚠️ **133k atrasos perdidos** (False Negatives)
- ⚠️ **775k falsos alarmes** (False Positives)

---

## 🔍 **Features Mais Importantes**

As **3 features mais impactantes** no modelo:

1. **dep_hour (29.1%)** - Hora da partida
   - Voos noturnos/madrugada têm maior risco
   
2. **carrier_delay_rate (15.3%)** - Histórico da companhia
   - Companhias com histórico ruim tendem a atrasar mais
   
3. **time_of_day (14.4%)** - Período do dia
   - Tarde/noite têm efeito cascata de atrasos

![Feature Importance](visualizations/feature_importance.png)

---

## 🎯 **Limitações e Trabalhos Futuros**

### **Limitações**
- ⚠️ **Precision baixa (23.5%)**: Muitos falsos alarmes
- ⚠️ **Sem dados climáticos em tempo real**: Limitado a dados históricos
- ⚠️ **Threshold fixo**: Não adapta por contexto (feriados, eventos)

### **Próximos Passos**
- 🔧 **Adicionar features climáticas** (API em tempo real)
- 🔧 **Ensemble de modelos** (XGBoost, LightGBM)
- 🔧 **Deploy em API REST** (FastAPI + Docker)
- 🔧 **Dashboard interativo** (Streamlit)
- 🔧 **Explicabilidade** (SHAP values para cada predição)

---

## 🛠️ **Tecnologias Utilizadas**

- **Python 3.8+** - Linguagem principal
- **Pandas** - Manipulação de dados
- **Scikit-learn** - Machine Learning
- **Matplotlib/Seaborn** - Visualizações
- **Joblib** - Serialização do modelo
- **Google Colab** - Ambiente de desenvolvimento

---

## 📚 **Referências**

- [Bureau of Transportation Statistics](https://www.transtats.bts.gov/)
- [FAA Flight Delay Data](https://www.faa.gov/data_research/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

---

## 👨‍💻 **Autor**

Desenvolvido por **[H12-25-B-Equipo 15-Data Science]**

---

## 📄 **Licença**

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 🙏 **Agradecimentos**

- Bureau of Transportation Statistics pelo dataset público
- Comunidade Kaggle por inspiração em projetos similares

---

**⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!**
