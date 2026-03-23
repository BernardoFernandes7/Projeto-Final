# Previsão de Clicks em Anuncios On-line
## Identificação da Equipa
**Grupo nº:** 6
* **Membros:** 2
 * Bernardo Fernandes - 2023132747
 * Hugo Grou - 2023137127

## Organização do Repositório
A estrutura deste projeto segue as boas práticas de Ciência de Dados e Engenharia de Software:
* **`data/`**: Armazenamento de dados (dados brutos em `raw/` e processados em `processed/`).
* **`docs/`**: Documentação técnica detalhada dividida por Milestones (M1, M2 e M3).
* **`notebooks/`**: Jupyter Notebooks para experimentação, limpeza e modelação.
* **`src/`**: Código-fonte modular (scripts `.py`) para funções reutilizáveis.
* **`reports/`**: Relatórios finais, apresentações e exportação de figuras (`figures/`).
* **`requirements.txt`**: Ficheiro de configuração com as bibliotecas necessárias.
## 1. Iniciação (Milestone 1)
### Contexto e Problema de Negócio
O objetivo principal é desenvolver um modelo de classificação (XGBoost/Random Forest) capaz de prever a probabilidade de clique em anúncios online (CTR) no dataset Avazu com 40 milhões de registos, analisando de que forma as variáveis contextuais (dispositivo, ambiente de navegação e tempo) influenciam essa decisão, garantindo um AUC-ROC superior a 0.75 e um Log Loss inferior a 0.40, de forma a permitir às empresas otimizar o seu investimento publicitário até à entrega do Milestone 3


### Objetivos do Projeto

* **Objetivo 1:** Desenvolver um modelo preditivo de classificação binária capaz de estimar a probabilidade de clique em anúncios online (CTR) no dataset Avazu com 40 milhões de registos, atingindo um AUC-ROC superior a 0.75, até à entrega do Milestone 3.
* **Objetivo 2:** Identificar as 5 variáveis de maior impacto na decisão de clique, utilizando métricas de importância de features dos modelos de árvore (Random Forest/XGBoost), até à entrega do Milestone 3.
* **Objetivo 3:** Comparar o desempenho de pelo menos 3 algoritmos de aprendizagem automática (Regressão Logística, Random Forest e Gradient Boosting), selecionando o modelo com melhor desempenho preditivo, até à entrega do Milestone 3.
* **Objetivo 4:** Avaliar todos os modelos treinados através de métricas adequadas a problemas de classificação binária desbalanceada, nomeadamente AUC-ROC, Log Loss inferior a 0.40, Precision, Recall e F1-score, até à entrega do Milestone 3.

### Fonte de Dados
* **Dataset:**  https://www.kaggle.com/datasets/madhu41289/avazu-ctr-prediction-exp
* **Dimensão:** O dataset Avazu tem aproximadamente **40 milhões de linhas e 24 colunas**.

## 2. Exploração (Milestone 2)
### Limpeza e Preparação
* Confirmada ausência de valores em falta em todas as 24 colunas do dataset
* Detetados e imputados valores `-1` mascarados na coluna `C20` (valores em falta não óbvios), substituídos pela moda global
* Removidas colunas não preditivas: `id`, `device_id`, `device_ip`
* Aplicada Codificação por Rótulo nas 7 colunas categóricas de alta cardinalidade (`site_id`, `site_domain`, `site_category`, `app_id`, `app_domain`, `app_category`, `device_model`)
* Aplicada Normalização por Padronização nas colunas numéricas (`C1`, `C14`–`C21`)
* Criadas 3 novas variáveis: `hora_do_dia` (extraída de `hour`), `banner_area` (C15 × C16) e `visibilidade_anuncio` (banner_pos / device_type + 1)
* Detalhes completos em [`docs/M2_exploracao.md`](docs/M2_exploracao.md)
### Principais Conclusões (EDA)

> *Inserir aqui o gráfico de CTR por hora do dia*

* **Sazonalidade Temporal:** O CTR varia significativamente ao longo do dia, com períodos de maior propensão de clique identificados nas primeiras horas da madrugada (0h–3h), o que contraria a intuição inicial.
* **Desequilíbrio de Classes:** A variável alvo `click` está fortemente desequilibrada, com aproximadamente 83% de não-cliques (0) e 17% de cliques (1), justificando o uso de métricas como AUC-ROC e F1-score em vez de simples Accuracy.
* **Variáveis mais correlacionadas com `click`:** As variáveis anónimas `C14`, `C15` e `C16` apresentam as correlações mais relevantes com a variável alvo, sugerindo que representam características do anúncio com forte impacto no CTR.

## 3. Modelação (Milestone 3)
### Abordagem Técnica
* **Modelos:** [Ex: Random Forest e XGBoost]
* **Métrica Principal:** [Ex: F1-Score ou RMSE]
## 4. Finalização (Milestone 4)
### Resposta ao Problema
[Resumo da solução e como ela gera valor para o negócio.]
### Recomendações de Inovação
1. [Sugestão prática baseada nos resultados]
## Como Reproduzir este Projeto
1. Clone o repositório: git clone (https://github.com/BernardoFernandes7/Projeto-Final.git)
2. Instale as dependências: pip install -r requirements.txt
3. Aceda ao Kaggle e faça upload do notebook
4. Adicione o dataset Avazu CTR Prediction via Add Data no Kaggle
5. Execute os notebooks seguindo a ordem numérica

**Instituição:** Coimbra Business School | ISCAC  
**Curso:** Licenciatura em Ciência de Dados para a Gestão  
**Unidade Curricular:** Projeto em Ciência de Dados  
**Professor Responsável:** Dora Melo (dmelo@iscac.pt)   
