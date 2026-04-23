# Previsão de Cliques em Publicidade Digital (Avazu CTR Prediction)

## Identificação da Equipa

* **Grupo nº:** 6
* **Membros:**
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
O setor da publicidade digital movimenta milhares de milhões de euros anualmente, assentando em grande parte em leilões em tempo real — o chamado *Real-Time Bidding* (RTB). Neste ecossistema, a taxa de clique (CTR — *Click-Through Rate*) é o principal indicador de eficácia: prever com precisão se um utilizador irá clicar num anúncio permite aos anunciantes otimizar os seus lances e às plataformas e maximizar a relevância dos espaços publicitários exibidos.

Este projeto utiliza o dataset Avazu CTR Prediction, que contém registos reais de impressões de anúncios em dispositivos móveis recolhidos ao longo de 10 dias. O desafio central consiste em construir um modelo de classificação binária capaz de prever, com base em 24 variáveis contextuais (como a hora do dia, tipo de dispositivo, posição do banner e categoria da aplicação), se a variável alvo click assumirá o valor 1 (clique) ou 0 (não-clique). Esta capacidade preditiva tem valor direto no mercado, permitindo reduzir o desperdício de orçamento em impressões irrelevantes e aumentar a rentabilidade das campanhas digitais.

### Objetivos do Projeto

* **Objetivo 1:** Desenvolver um modelo de classificação binária capaz de prever se um utilizador irá clicar num anúncio, atingindo um AUC-ROC mínimo de 0.75 no conjunto de teste, utilizando o dataset Avazu CTR Prediction até ao final do Milestone 3.
* **Objetivo 2:** Identificar as 5 variáveis mais determinantes para a previsão do clique (através de Feature Importance) e diagnosticar os perfis de erro do modelo, fornecendo recomendações acionáveis sobre os contextos de anúncios com maior probabilidade de conversão, até à conclusão da fase de modelação.

### Fonte de Dados
* **Dataset:** [Avazu CTR Prediction — Kaggle](https://www.kaggle.com/competitions/avazu-ctr-prediction)
* **Dimensão:** 40.428.967 registos × 24 colunas (dataset original); amostra de 5.000.000 registos utilizada no projeto, por sugestão da professora, com `random_state=42` para garantir reprodutibilidade.

## 2. Exploração (Milestone 2)
### Limpeza e Preparação

* A variável `C20` continha valores `-1` como marcador de dados omissos — foi aplicada imputação pela **moda**, estratégia justificada pela natureza categórica da variável e pela robustez da moda face a distribuições assimétricas. Detalhes em `docs/M2_exploracao.md`.
* Colunas sem valor preditivo (`id`, `device_id`, `device_ip`) foram removidas para reduzir ruído e dimensionalidade.
* Os tipos de dados foram auditados e corrigidos para garantir conformidade com o perfil esperado.
* Foram criadas duas novas variáveis por *feature engineering*: `hora_do_dia` (extraída de `hour`) e `banner_area` (produto de `C15 × C16`).
* Variáveis categóricas de alta cardinalidade foram codificadas com **Target Encoding** (Leave-One-Out); as restantes com **Label Encoding**.

### Principais Conclusões (EDA)

> *Ver o gráfico de distribuição da variável `click` — `reports/figures/`*

* **Ponto-chave 1:** A variável alvo `click` está fortemente desequilibrada — aproximadamente **83% de não-cliques (0)** e **17% de cliques (1)**, rácio de 1:5. Este desequilíbrio justifica a priorização do **AUC-ROC** como métrica principal.
* **Ponto-chave 2:** O CTR varia significativamente ao longo do dia, com picos nas primeiras horas da madrugada, tornando `hora_do_dia` uma variável preditiva relevante.
* **Ponto-chave 3:** As variáveis relacionadas com as dimensões do banner (`C16`, `banner_area`) apresentam correlação com a probabilidade de clique, confirmada posteriormente pelo modelo final.
* **Ponto-chave 4:** A matriz de correlação de Pearson não revelou pares com correlação superior a 0,85, confirmando a ausência de multicolinearidade crítica.

## 3. Modelação (Milestone 3
### Abordagem Técnica

* **Modelos:** Regressão Logística (Baseline), Random Forest, XGBoost, XGBoost Otimizado (modelo final)
* **Métrica Principal:** AUC-ROC — escolhida pela capacidade de avaliar modelos em datasets desequilibrados, independentemente do limiar de decisão
* **Resultado Final:** O **XGBoost Otimizado** alcançou um **AUC-ROC de 0,7528** no conjunto de teste, superando o objetivo SMART de AUC-ROC > 0,75, com uma melhoria total de **+0,0398** face ao Baseline
* **Validação Cruzada (5-Fold):** Média de **0,7517 ± 0,0007**, confirmando estabilidade excecional do modelo

> *Ver o gráfico das Curvas ROC comparativas — `reports/figures/curvas_roc_comparacao.png`*

## 4. Finalização (Milestone 4)

### Resposta ao Problema
[A preencher na Milestone 4 — resumo da solução final e como ela gera valor para o negócio de publicidade digital.]

### Recomendações de Inovação
1. [A preencher na Milestone 4 — sugestão prática baseada nos resultados obtidos.]

## Como Reproduzir este Projeto

1. Clone o repositório: `git clone "https://github.com/BernardoFernandes7/Projeto-Final"`
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute os notebooks na pasta `notebooks/` seguindo a ordem numérica.

> **Nota:** O dataset original (40M registos) deve ser descarregado diretamente do Kaggle. O notebook carrega automaticamente uma amostra de 5.000.000 registos com `random_state=42`.

**Instituição:** Coimbra Business School | ISCAC  
**Curso:** Licenciatura em Ciência de Dados para a Gestão  
**Unidade Curricular:** Projeto em Ciência de Dados  
**Professor Responsável:** Dora Melo (dmelo@iscac.pt)
