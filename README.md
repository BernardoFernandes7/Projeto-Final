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
O objetivo principal é desenvolver um modelo preditivo capaz de estimar a **probabilidade de clique em anúncios online** (CTR), usando o dataset Avazu.
### Objetivos do Projeto
* **Objetivo 1:** Desenvolver um modelo preditivo para estimar a probabilidade de clique
* **Objetivo 2:** Identificar as 5 variáveis de maior impacto na decisão de clique
* **Objetivo 3:** Comparar diferentes algoritmos (Regressão Logística, Random Forest, Gradient Boosting)
* **Objetivo 4:** Avaliar os modelos com métricas adequadas a classes desbalanceadas (AUC-ROC, Log-Loss, F1-score)

### Fonte de Dados
* **Dataset:**  https://www.kaggle.com/datasets/madhu41289/avazu-ctr-prediction-exp
* **Dimensão:** O dataset Avazu tem aproximadamente **40 milhões de linhas e 24 colunas**.
## 2. Exploração (Milestone 2)
### Limpeza e Preparação
* [Breve resumo das ações de limpeza tomadas. Detalhes em `docs/M2_exploracao.md`]
### Principais Conclusões (EDA)
> *Dica: Insere aqui o gráfico mais importante do projeto.*
* **Ponto-chave:** [Ex: Identificámos que o fator X influencia em 40% o resultado Y, por aplicação
do método ganho de informação]
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
