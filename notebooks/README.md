# ⚖️ Comparação entre o Modelo Clássico e o Modelo Quântico


Este documento apresenta uma análise comparativa entre duas abordagens aplicadas ao problema de predição de atrasos de voos: um modelo clássico consolidado, voltado à produção, e um modelo quântico variacional, de caráter experimental.

---

## 🎯 Contexto do Problema

O objetivo é prever se um voo sofrerá atraso maior ou igual a 15 minutos, um problema
caracterizado por:

- Grande volume de dados
- Desbalanceamento de classes
- Forte impacto operacional
- Necessidade de robustez temporal

---

## 🧠Visão Geral dos Modelos

| Critério | Modelo Clássico | Modelo Quântico (VQC) |
|--------|----------------|-----------------------|
| Abordagem | Machine Learning Tradicional | Machine Learning Quântico |
| Ambiente de execução | Computação clássica | Simulação quântica |
| Volume de dados | Alto | Reduzido (subamostra) |
| Tempo de treinamento | Baixo | Elevado |
| Estabilidade | Alta | Sensível a inicializações |
| Escalabilidade | Alta | Limitada |
| Interpretabilidade | Alta | Limitada |
| Maturidade | Consolidada | Experimental |
---

## 📊 Comparação de Métricas

### 🔵 Modelo Clássico 

| Métrica | Valor |
|-------|-------|
| ROC-AUC | **0.6252** |
| Recall | **94.28%** |
| Precision | 17.76% |
| F1-Score | 0.2989 |
| Validação | TimeSeriesSplit |
| Escala | Milhões de voos |

> O modelo clássico prioriza **Recall**, reduzindo atrasos não detectados, com validação temporal
robusta e foco em custo operacional.

---

###  ⚛️ Modelo Quântico 

| Métrica | Valor |
|-------|-------|
| ROC-AUC | **0.6410** |
| Accuracy | 0.6220 |
| Precision | 0.3217 |
| Recall | 0.6916 |
| Ambiente | Simulação (`default.qubit`) |
| Escala | Subamostra |

> Os resultados indicam potencial teórico, porém o modelo é limitado por simulação clássica, alto custo computacional e baixa escalabilidade.

---

## ⚙️ Comparação Técnica



| Critério | Modelo Clássico | Modelo Quântico |
|--------|----------------|----------------|
| Tempo de Treinamento | ✅ Viável | ❌ Elevado |
| Uso em Produção | ✅ Sim | ❌ Não |
| Integração via API | ✅ FastAPI | ❌ Não |
| Reprodutibilidade | ✅ Alta | ⚠️ Experimental |
| Maturidade Tecnológica | ✅ Consolidada | ❌ Emergente |

---
## 📌 Escolha do Modelo

> **Modelo escolhido:** 🔵 **Modelo Clássico (Machine Learning Tradicional)**

A escolha do modelo final não foi baseada exclusivamente em métricas isoladas, mas em
um conjunto de **critérios técnicos, operacionais e práticos**, alinhados aos objetivos do projeto.

Os principais critérios considerados foram:

- **Viabilidade operacional**: capacidade de ser utilizado em um cenário real.
- **Escalabilidade**: possibilidade de lidar com grandes volumes de dados.
- **Custo computacional**: tempo de treinamento e consumo de recursos.
- **Estabilidade e robustez**: comportamento consistente entre execuções.
- **Integração com sistemas**: facilidade de deploy e consumo via API.
- **Maturidade tecnológica**: disponibilidade de ferramentas estáveis e bem documentadas.

Embora o modelo quântico apresente resultados promissores em métricas específicas, ele ainda enfrenta limitações significativas relacionadas à simulação clássica, restrições de escala e alto custo computacional. Dessa forma, a decisão priorizou **robustez, confiabilidade e aplicabilidade prática**.

---

## 🤔 Qual é o Papel do Modelo Quântico?
O modelo quântico foi mantido como uma **prova de conceito e ferramenta exploratória**, permitindo:

- Exploração prática de Machine Learning Quântico.
- Comparação direta com modelos clássicos.
- Discussão realista sobre limitações atuais da tecnologia.

> ⚠️ Importante: o modelo quântico é executado em **simulação clássica**, não em hardware quântico real.

---

## 🏁 Conclusão

- 🔵 **Modelo Clássico**: escolhido para produção por ser robusto, escalável e aplicável  
- ⚛️ **Modelo Quântico**: mantido como abordagem experimental e exploratória  

Embora o modelo quântico represente uma abordagem inovadora e promissora, **o modelo clássico foi escolhido por apresentar melhor desempenho, maior estabilidade e viabilidade prática**. O uso do VQC reforça o caráter experimental do estudo e contribui para uma análise mais completa e crítica sobre o uso de computação quântica em problemas reais.

---
## 👥 Time

**H12-25-B-Equipo 15-Data Science**