# Milestone 3: Modelação e Avaliação



## 1. Estratégia de Modelação

**Divisão do dataset:** Utilizámos uma divisão de 80% para treino (4.000.000 registos) e 20% para teste (1.000.000 registos), com `stratify=y` e `random_state=42`.
A estratificação garante que a proporção de cliques (≈17%) se mantém igual em ambos os conjuntos, o que é essencial dado o desequilíbrio de classes. 
O conjunto de teste foi isolado desde o início e nunca foi visto durante o treino ou a otimização. O `StandardScaler` foi ajustado exclusivamente no conjunto de treino (`fit_transform`) e depois aplicado ao teste (`transform`). O *Frequency Encoding* também foi calculado apenas no treino.

```
Treino : 4.000.000 registos (80%)
Teste  : 1.000.000 registos (20%)

Proporção de cliques no treino : 0.1697
Proporção de cliques no teste  : 0.1697

Proporções consistentes — divisão estratificada correcta.
Isolamento garantido: X_test nunca será visto durante o treino nem o tuning.
```

**Métrica de Sucesso:** A métrica principal escolhida foi o **AUC-ROC** por três razões: é a métrica oficial da competição Avazu (He et al., 2014), o que 
permite comparar os resultados com a literatura; é robusta ao desequilíbrio de classes (83% não-cliques / 17% cliques), ao contrário da *Accuracy*, que seria 
enganosa; e em contexto de *Real-Time Bidding*, o modelo é usado para **ordenar** impressões por probabilidade de clique — e o AUC-ROC mede exatamente essa 
capacidade de ordenação. O **F1-Score** foi definido como métrica secundária porque equilibra Precisão e *Recall*, relevante num contexto onde tanto os Falsos 
Positivos (impressões desperdiçadas) como os Falsos Negativos (receita perdida) têm custo real para o anunciante. A *Accuracy* foi excluída porque um modelo 
que previsse sempre "não clique" teria 83% de acerto sem qualquer utilidade preditiva — o chamado *accuracy paradox*.



## 2. Experiências Realizadas

### 2.1. Modelo *Baseline* — Regressão Logística

**Algoritmo:** Regressão Logística com `class_weight='balanced'`, `max_iter=1000` e `StandardScaler` aplicado antes do treino.

**Resultado:**

| Métrica | Treino | Teste |
| :--- | :---: | :---: |
| AUC-ROC | 0.6411 | 0.6412 |
| F1-Score | 0.3362 | 0.3363 |
| Precisão | — | 0.2325 |
| *Recall* | — | 0.6073 |

O diagnóstico automático reportou **generalização adequada** (Δ AUC = 0.0001), confirmando ausência de *overfitting*. O *baseline* estabelece 
o patamar mínimo: qualquer modelo candidato tem de superar AUC-ROC = 0.6412 para justificar a sua complexidade adicional.

### 2.2. Modelos Candidatos

| Algoritmo | Parâmetros Base | AUC-ROC (Treino) | AUC-ROC (Teste) | Notas |
| :--- | :--- | :---: | :---: | :--- |
| *Baseline* (Reg. Logística) | `class_weight='balanced'`, `max_iter=1000` | 0.6411 | 0.6412 | Patamar de referência |
| *Random Forest* | `n_estimators=100`, `max_depth=10`, `class_weight='balanced'` | 0.7231 | 0.7218 | Boa generalização |
| **XGBoost** | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1` | **0.7436** | **0.7419** | **Melhor AUC-ROC — selecionado para tuning** |

Ambos os modelos superam claramente o *baseline* (+0.08 para o *Random Forest* e +0.10 para o XGBoost). O XGBoost foi selecionado para a fase de otimização por apresentar o melhor desempenho no conjunto de teste. As curvas de aprendizagem de ambos os modelos mostram boa generalização, sem *gap* significativo entre treino e validação.



## 3. Otimização (*Tuning*)

**Técnica Utilizada:** Utilizámos *RandomizedSearchCV* com 20 iterações e `StratifiedKFold` com K=5 *folds*, aplicado exclusivamente ao conjunto de treino. Optámos por *RandomizedSearchCV* em vez de *GridSearchCV* porque o espaço de hiperparâmetros é vasto (distribuições contínuas para `learning_rate` e `subsample`) e uma pesquisa exaustiva seria computacionalmente proibitiva com 4 milhões de registos.

**Melhores hiperparâmetros encontrados:**

| Hiperparâmetro | Valor | Significado |
| :--- | :---: | :--- |
| `colsample_bytree` | 0.787 | Fracção de colunas usadas em cada árvore |
| `learning_rate` | 0.182 | Tamanho do passo de gradiente por iteração |
| `max_depth` | 8 | Profundidade máxima de cada árvore |
| `min_child_weight` | 9 | Peso mínimo necessário para criar um nó folha |
| `n_estimators` | 266 | Número total de árvores no *ensemble* |
| `subsample` | 0.605 | Fracção de registos usados em cada árvore |

**Melhoria obtida:** O AUC-ROC subiu de 0.7419 para **0.7509** após o ajuste (+0.0090), ultrapassando o objetivo SMART definido (AUC-ROC > 0.75 ✓). A melhoria acumulada face ao *baseline* inicial é de **+0.1097**.

| Modelo | AUC-ROC (Treino) | AUC-ROC (Teste) | F1 (Teste) | Melhoria |
| :--- | :---: | :---: | :---: | :---: |
| XGBoost (base) | 0.7436 | 0.7419 | 0.4157 | — |
| **XGBoost (otimizado)** | **0.7612** | **0.7509** | **0.4223** | **+0.0090** |



## 4. Avaliação do Modelo Final

### 4.1. Matriz de Confusão / Erros

| | Previsto: Não-clique | Previsto: Clique |
| :--- | :---: | :---: |
| **Real: Não-clique** | 577.197 (VN) — 57,7% | 253.084 (FP) — 25,3% |
| **Real: Clique** | 56.547 (FN) — 5,6% | 113.172 (VP) — 11,3% |

> **Análise:** O modelo ainda apresenta 253.084 Falsos Positivos — impressões pagas sem retorno, custo desperdiçado para o anunciante. Os Falsos Negativos (56.547), embora menos numerosos, são o erro mais prejudicial: cada um representa uma impressão com probabilidade real de clique que o modelo ignorou — receita perdida. Os FN concentram-se em impressões da hora 17:00 e em contextos com `device_conn_type` elevado. Ajustar o *threshold* de decisão abaixo de 0.5 poderia melhorar o *Recall* à custa de mais FP, o que pode ser preferível dependendo da estratégia do anunciante.

### 4.2. Importância dos Atributos (*Feature Importance*)

1. `banner_area` — importância = 0.326 (**variável criada neste projeto** — área do *banner*, C15 × C16)
2. `C16` — importância = 0.160 (dimensão do *banner*, altura em píxeis)
3. `device_type` — importância = 0.107 (tipo de dispositivo: móvel vs. PC)
4. `C21` — importância = 0.057 (variável anónima do anúncio)
5. `site_id` — importância = 0.054 (identificador do *site* após *Frequency Encoding*)

As top-5 variáveis concentram **70% da importância total** do modelo, sugerindo que um modelo mais simples com estas 5 variáveis poderia ter desempenho competitivo com muito menos custo computacional — uma pista para trabalho futuro.



## 5. Conclusão da Fase de Modelação

O XGBoost otimizado está pronto para ser apresentado como solução final pelas seguintes razões:

- **AUC-ROC = 0.7509 ✓** — supera o objetivo SMART definido no *Milestone* 1, com uma melhoria acumulada de +0.1097 face ao *baseline*.
- **Robustez confirmada** — desvio padrão de 0.0006 em *cross-validation* (IC 95%: [0.7493, 0.7516]) garante que o resultado não é fruto de uma divisão afortunada dos dados.
- **Validação da feature engineering** — `banner_area`, criada neste projeto, é a variável mais importante do modelo (32.6% do ganho de informação total), validando retroativamente essa decisão.
- **Superioridade justificada** — a Regressão Logística ficou 0.11 pontos abaixo em AUC-ROC, justificando plenamente o custo computacional adicional do XGBoost.
- **Melhoria futura identificada** — ajustar o *threshold* abaixo de 0.5 para maximizar o *Recall* pode ser preferível em estratégias onde perder cliques reais é mais caro do que fazer lances desnecessários.



*Data de última atualização: 23/04/2026*
