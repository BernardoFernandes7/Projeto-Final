# [Previsão de Clicks em Anuncios On-line]
## Identificação da Equipa
* **Grupo nº:6 **
* **Membros:2 **
 * [Bernardo Fernandes] - [2023132747]
 * [Hugo Grou] - [2023137127]
   
## Organização do Repositório
A estrutura deste projeto segue as boas práticas de Ciência de Dados e Engenharia de Software:
** data/: Armazenamento de dados (dados brutos em raw/ e processados em processed/).
** docs/: Documentação técnica detalhada dividida por Milestones (M1, M2 e M3).
** notebooks/: Jupyter Notebooks para experimentação, limpeza e modelação.
** src/: Código-fonte modular (scripts .py) para funções reutilizáveis.
** reports/: Relatórios finais, apresentações e exportação de figuras (figures/).
** requirements.txt: Ficheiro de configuração com as bibliotecas necessárias.

## Iniciação (Milestone 1)
Contexto e Problema de Negócio
O desafio central deste projeto insere-se no setor de Publicidade Digital (AdTech). O problema consiste em prever a probabilidade de um utilizador clicar num anúncio online, uma métrica conhecida como Click-Through Rate (CTR).

No ecossistema de Real-Time Bidding (RTB), as plataformas têm milissegundos para decidir se compram um espaço publicitário. Prever o clique com precisão é vital para a otimização de inventário (garantindo que o anúncio certo chega ao utilizador certo), para a redução de custos operacionais e para a maximização do retorno sobre o investimento (ROI) tanto para anunciantes como para editores.

## Objetivos do Projeto
Os objetivos deste projeto focam-se no desenvolvimento de um sistema preditivo robusto, capaz de classificar com elevada eficácia a probabilidade de interação do utilizador com anúncios digitais. Para além da componente técnica de modelação, o trabalho visa extrair conhecimento estratégico através da identificação e hierarquização dos principais fatores determinantes para o clique, permitindo compreender como variáveis de contexto — como a temporalidade e o perfil do dispositivo — impactam a performance das campanhas no ecossistema da Avazu.

## Fonte de Dados
Dataset: Avazu Click-Through Rate Prediction (disponível via Kaggle). Os dados incluem 10 dias de cliques históricos para estratégia de treino.

Dimensão: O dataset é composto por 40.428.967 linhas e 24 colunas, incluindo variáveis temporais, identificadores de dispositivos, categorias de sites/apps e diversas variáveis anonimizadas (C1, C14-C21).


## Links Uteis
Conjunto de Dados: https://www.kaggle.com/datasets/madhu41289/avazu-ctr-prediction-exp 

Kaggle Notebook: https://www.kaggle.com/code/hugogrou/projeto-cdg
