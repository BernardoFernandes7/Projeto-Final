# Previsão de Cliques em Anúncios Online — CTR com Avazu

> Projeto desenvolvido no âmbito da UC de Projeto em Ciência de Dados, cujo objetivo é prever se um utilizador vai ou não clicar num anúncio, com base nas características da impressão publicitária.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)

---

## Equipa

| Campo | Detalhe |
|---|---|
| **Grupo** | Nº 6 |
| **Instituição** | Coimbra Business School — ISCAC |
| **Curso** | Licenciatura em Ciência de Dados para a Gestão |
| **UC** | Projeto em Ciência de Dados |
| **Docente** | Dora Melo — dmelo@iscac.pt |
| **Ano Letivo** | 2025/2026 |

| Nome | Nº de Aluno |
|---|---|
| Bernardo Fernandes | 2023132747 |
| Hugo Grou | 2023137127 |

---

## Estrutura do Repositório

```
Projeto-Final/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # dataset original (demasiado grande — ver Kaggle)
│   └── processed/        # amostra exportada após pré-processamento
├── docs/
│   ├── M1_iniciacao.md
│   ├── M2_exploracao.md
│   └── M3_modelacao.md
├── notebooks/
│   ├── 1.0_eda_limpeza.ipynb
│   └── 2.0_modelacao_treino.ipynb
├── src/
│   ├── data_loader.py
│   └── evaluation_utils.py
└── reports/
    └── figures/
```

---

## 1. Iniciação (Milestone 1)

### Contexto e Problema

A publicidade digital funciona, em grande parte, através de leilões em tempo real — o chamado *Real-Time Bidding* (RTB). Cada vez que uma página carrega, realiza-se um leilão para decidir qual anúncio aparece. A pergunta central é simples: este utilizador, neste contexto, vai clicar?

Prever isso com antecedência tem valor direto: permite que os anunciantes apostem mais nos casos com maior probabilidade de clique e evitem gastar dinheiro em impressões que ninguém vai clicar. É exatamente este o problema que trabalhámos — usando o *dataset* Avazu, que contém cerca de 40 milhões de impressões reais recolhidas em dispositivos móveis.

O *dataset* tem 24 variáveis: algumas dizem respeito ao dispositivo (tipo, modelo, ligação), outras ao contexto do anúncio (posição do *banner*, categoria do site ou *app*) e uma variável temporal. A variável alvo é `click` — 1 se houve clique, 0 se não houve.

### Perguntas de Investigação

1. Que variáveis contextuais têm mais influência na probabilidade de clique — o dispositivo, a hora, ou as características do anúncio?
2. Consegue-se manter um desempenho competitivo com um subconjunto reduzido de variáveis?
3. Como é que o forte desequilíbrio entre cliques (~17%) e não-cliques (~83%) afeta os modelos e como se pode lidar com isso?

### Objetivo SMART

Desenvolver um modelo de classificação binária capaz de prever a probabilidade de um utilizador clicar num anúncio *mobile*, usando o *dataset* Avazu (amostra de 5 milhões de registos), atingindo um AUC-ROC superior a 0,75 no conjunto de teste isolado, até à data de entrega do *Milestone* 3 (23 de abril de 2026).

### Principais descobertas — ponto de partida para o Milestone 2

A inspeção inicial ao *dataset* revelou quatro aspetos que condicionaram diretamente o trabalho da fase seguinte:

- **Sem valores nulos explícitos**, mas com 2.344.248 registos com valor `-1` na coluna `C20` (46,88% da amostra) — valores em falta codificados de forma não *standard* que precisaram de imputação.
- **Desequilíbrio acentuado na variável alvo**: 83,03% de não-cliques vs. 16,97% de cliques — o que torna a *Accuracy* uma métrica inútil e impõe o uso de AUC-ROC.
- **1.130.966 duplicados lógicos** (22,6% da amostra) — mesmo utilizador, mesmo anúncio, mesmo momento. Decidimos mantê-los por serem impressões legítimas em contexto de *RTB*.
- **Alta cardinalidade** em variáveis como `site_id`, `app_id` e `device_model` (milhares de valores únicos) — o que excluiu o uso direto de *One-Hot Encoding* e orientou a escolha para *Frequency Encoding*.

- ***Dataset*:** Avazu CTR Prediction — [Kaggle](https://www.kaggle.com/datasets/madhu41289/avazu-ctr-prediction-exp)
- **Dimensão:** ~40 milhões de registos × 24 colunas
- **Variável alvo:** `click` (0 = não clicou; 1 = clicou)
- **Nota:** Trabalhámos com uma amostra aleatória de 5 milhões de registos (`random_state=42`), por sugestão da docente, dado que o *dataset* completo esgota a RAM disponível no Kaggle.

### Ambiente de Trabalho (Kaggle)

- **Repositório GitHub:** [github.com/BernardoFernandes7/Projeto-Final](https://github.com/BernardoFernandes7/Projeto-Final)
- **Ambiente:** Kaggle *Notebooks* com GPU T4 × 2, 30 GB de RAM
- **Sincronização:** *File → Link to GitHub* → pasta `notebooks/`

---

## 2. Exploração (Milestone 2)

### Limpeza e Preparação

- Nenhum valor nulo explícito encontrado — os valores em falta aparecem como `-1` nas colunas anónimas C14–C21.
- Imputámos pela moda as colunas com mais de 1% de valores `-1`.
- Verificámos a existência de observações repetidas — os duplicados lógicos foram mantidos porque em *RTB* é normal o mesmo anúncio ser mostrado várias vezes ao mesmo utilizador.
- Removemos `id`, `device_id` e `device_ip` por não terem valor preditivo.
- Aplicámos *Frequency Encoding* nas variáveis categóricas de alta cardinalidade, depois do *split* treino/teste, para evitar *data leakage*.
- Criámos três novas variáveis: `hora_do_dia` (extraída de `hour`), `banner_area` (C15 × C16) e `visibilidade_anuncio`.

> Detalhes completos em [`docs/M2_exploracao.md`](docs/M2_exploracao.md).

### Principais conclusões da EDA

- A variável alvo está muito desequilibrada: ~83% não-cliques e ~17% cliques. Por isso, usar *Accuracy* como métrica seria enganoso — um modelo que previsse sempre 0 teria 83% de acerto sem qualquer utilidade.
- O CTR varia ao longo do dia, com os valores mais altos nas primeiras horas da madrugada, o que não era esperado à partida.
- As variáveis anónimas C14, C15 e C16 são as que têm correlação mais alta com `click`, o que sugere que representam características do anúncio.

---

## 3. Modelação (Milestone 3)

### Abordagem

Testámos três modelos: *Regressão Logística* como *baseline*, *Random Forest* e *XGBoost*. A métrica principal foi o AUC-ROC, que é a métrica oficial da competição Avazu e é robusta ao desequilíbrio de classes.

Dividimos os dados 80/20 com estratificação. O *StandardScaler* foi ajustado apenas no treino. A otimização de hiperparâmetros foi feita com *RandomizedSearchCV* e validação cruzada estratificada com 5 *folds*, tudo dentro do conjunto de treino.

### Resultados

| Modelo | AUC-ROC (Treino) | AUC-ROC (Teste) | F1 (Teste) |
|--------|-----------------|-----------------|------------|
| Regressão Logística *(baseline)* | 0.6411 | 0.6412 | 0.3363 |
| *Random Forest* | 0.7231 | 0.7218 | 0.3938 |
| XGBoost | 0.7436 | 0.7419 | 0.4157 |
| **XGBoost (Otimizado)** | **0.7612** | **0.7509** | **0.4223** |

**Modelo final:** XGBoost otimizado — AUC-ROC de 0.7509 no conjunto de teste, estável em *cross-validation* (0.7505 ± 0.0006). Objetivo SMART atingido (AUC-ROC > 0.75). Detalhes em [`docs/M3_modelacao.md`](docs/M3_modelacao.md).

---

## 4. Finalização (Milestone 4)

*A preencher após conclusão do Milestone 4.*

---

## Referências

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

He, X., Pan, J., Jin, O., Xu, T., Liu, B., Xu, T., Shi, Y., Atallah, A., Herbrich, R., Bowers, S., & Candela, J. Q. (2014). Practical lessons from predicting clicks on ads at Facebook. *Proceedings of the 8th International Workshop on Data Mining for Online Advertising*, 1–9. https://doi.org/10.1145/2648584.2648589

Japkowicz, N., & Stephen, S. (2002). The class imbalance problem: A systematic study. *Intelligent Data Analysis*, *6*(5), 429–449. https://doi.org/10.3233/IDA-2002-6504

Kaggle. (s.d.). *Avazu CTR prediction* [Dataset]. https://www.kaggle.com/datasets/madhu41289/avazu-ctr-prediction-exp

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesneau, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

---

*Grupo 6 — ISCAC | 2025/2026*
