# Previsão de Cliques em Publicidade Digital (Avazu CTR Prediction)
Um projeto de pesquisa que compara modelos de sequência modernos para prever click em anúncio on-line, usando o conjunto de dados Avazu CTR Prediction.
Nota: Link do vídeo:https://drive.google.com/file/d/1JhS-cE3PIqNkZGqZ6Ck42rTA5uDHOveo/view?usp=sharing
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

O setor da publicidade digital movimenta milhares de milhões de euros anualmente, assentando em grande parte em leilões em tempo real (*Real-Time Bidding* (RTB)). 
Neste ecossistema, a taxa de clique (CTR — *Click-Through Rate*) é o principal indicador de eficácia: prever com precisão se um utilizador irá clicar num anúncio
permite aos anunciantes otimizar os seus lances e às plataformas maximizar a relevância dos espaços publicitários exibidos.

Este projeto utiliza o dataset Avazu CTR Prediction, que contém registos reais de impressões de anúncios em dispositivos móveis recolhidos ao longo de 10 dias.
O desafio central consiste em construir um modelo de classificação binária capaz de prever, com base em variáveis contextuais como a hora do dia, tipo de dispositivo, 
posição do banner e categoria da aplicação, se a variável alvo `click` assumirá o valor 1, correspondente a clique, ou 0, correspondente a não clique. Esta capacidade 
preditiva tem valor direto no mercado, permitindo reduzir o desperdício de orçamento em impressões irrelevantes e aumentar a rentabilidade das campanhas digitais.

Do ponto de vista da Aprendizagem Automática, este projeto enquadra-se num problema preditivo supervisionado de classificação binária. É um problema preditivo porque 
o objetivo é estimar o valor futuro ou desconhecido da variável alvo `click`; é supervisionado porque o dataset contém exemplos históricos já rotulados com o resultado 
real; e é de classificação binária porque a variável alvo possui apenas duas classes possíveis: `0` para não clique e `1` para clique. Assim, o problema não é descritivo 
nem não supervisionado, pois não se pretende apenas descobrir grupos, padrões ou associações nos dados. O objetivo principal é treinar modelos capazes de aprender a partir 
de exemplos previamente rotulados e prever corretamente a classe de novos registos. 

Para resolver este problema, serão testados diferentes modelos de classificação supervisionada, 
incluindo Regressão Logística como modelo de referência e modelos mais complexos baseados em árvores, como Random Forest,
XGBoost e LightGBM. A comparação entre estes modelos permitirá avaliar qual apresenta melhor capacidade preditiva para a variável alvo `click`, 
tendo em conta métricas adequadas para classificação binária, como AUC-ROC, F1-Score, Precisão e Recall.

### Objetivos do Projeto

* **Objetivo 1:** Desenvolver um modelo de classificação binária capaz de prever se um utilizador irá clicar num anúncio, atingindo um AUC-ROC mínimo de 0.75
* no conjunto de teste, utilizando o dataset Avazu CTR Prediction até ao final do Milestone 3.
* **Objetivo 2:** Identificar as 5 variáveis mais determinantes para a previsão do clique (através de Feature Importance) e diagnosticar os perfis de erro do modelo,
*  fornecendo recomendações acionáveis sobre os contextos de anúncios com maior probabilidade de conversão, até à conclusão da fase de modelação.

### Dicionário de Variáveis 

| Variável | Tipo de Variável | Subtipo | Descrição |
| :--- | :--- | :--- | :--- |
| `id` | Categórica | Nominal / alfanumérica | Identificador único do registo, anonimizado. Não representa uma característica preditiva direta. |
| `click` | Binária | Numérica discreta, valores 0/1 | Variável alvo. Indica se o anúncio foi clicado (`1`) ou não clicado (`0`). |
| `hour` | Numérica | Discreta temporal | Data e hora da impressão do anúncio, no formato `YYMMDDhh`. Pode ser transformada em variáveis temporais como hora, dia ou dia da semana. |
| `C1` | Categórica | Nominal codificada numericamente | Variável anonimizada da base Avazu. Apesar de ser numérica, representa categorias. |
| `banner_pos` | Categórica | Nominal codificada numericamente | Posição do banner na página. Embora esteja codificada com números, representa categorias de posição. |
| `site_id` | Categórica | Nominal / alfanumérica | Identificador do site onde o anúncio foi exibido. |
| `site_domain` | Categórica | Nominal / alfanumérica | Domínio do site onde ocorreu a impressão do anúncio. |
| `site_category` | Categórica | Nominal / alfanumérica | Categoria temática do site. |
| `app_id` | Categórica | Nominal / alfanumérica | Identificador da aplicação mobile onde o anúncio foi exibido. |
| `app_domain` | Categórica | Nominal / alfanumérica | Domínio associado à aplicação mobile. |
| `app_category` | Categórica | Nominal / alfanumérica | Categoria da aplicação mobile. |
| `device_id` | Categórica | Nominal / alfanumérica | Identificador anonimizado do dispositivo. |
| `device_ip` | Categórica | Nominal / alfanumérica | Endereço IP anonimizado do dispositivo. |
| `device_model` | Categórica | Nominal / alfanumérica | Modelo do dispositivo utilizado pelo utilizador. |
| `device_type` | Categórica | Nominal codificada numericamente | Tipo de dispositivo, por exemplo smartphone, tablet ou desktop. |
| `device_conn_type` | Categórica | Nominal codificada numericamente | Tipo de ligação à internet utilizada pelo dispositivo, como WiFi ou rede móvel. |
| `C14` | Categórica | Nominal codificada numericamente | Variável anonimizada relacionada com o contexto do anúncio. |
| `C15` | Numérica | Discreta finita | Dimensão associada ao anúncio, geralmente relacionada com largura ou configuração visual. |
| `C16` | Numérica | Discreta finita | Dimensão associada ao anúncio, geralmente relacionada com altura ou configuração visual. |
| `C17` | Categórica | Nominal codificada numericamente | Variável anonimizada relacionada com o contexto do anúncio. |
| `C18` | Categórica | Nominal codificada numericamente | Variável anonimizada relacionada com o contexto do anúncio. |
| `C19` | Categórica | Nominal codificada numericamente | Variável anonimizada relacionada com o contexto do anúncio. |
| `C20` | Categórica | Nominal codificada numericamente | Variável anonimizada relacionada com o contexto do anúncio. |
| `C21` | Categórica | Nominal codificada numericamente | Variável anonimizada relacionada com o contexto do anúncio. |

### Observações no Conjunto de Dados
O conjunto de dados utilizado neste projeto é composto por registos de impressões de anúncios digitais em dispositivos móveis. Cada observação representa 
uma impressão de anúncio exibida a um utilizador, podendo ou não resultar num clique.

A variável alvo do problema é `click`, que assume dois valores possíveis:

- `0`: o anúncio foi exibido, mas não recebeu clique;
- `1`: o anúncio foi exibido e recebeu clique.

Desta forma, cada linha do dataset corresponde a uma observação individual do comportamento do utilizador perante um anúncio. O objetivo do projeto é utilizar 
as características associadas a cada impressão, como a hora, o tipo de dispositivo, a posição do banner, o site, a aplicação e outras variáveis contextuais, 
para prever a probabilidade de ocorrência de clique.

Uma característica importante deste conjunto de dados é o forte desbalanceamento da variável alvo. 
A maioria das observações pertence à classe `0`, ou seja, anúncios que não receberam clique, enquanto a classe `1`, 
correspondente aos cliques, representa uma proporção menor dos dados. Este aspeto é comum em problemas de publicidade digital, 
uma vez que a taxa de clique tende naturalmente a ser baixa.

Além disso, o dataset contém várias variáveis categóricas de elevada cardinalidade, como `site_id`, `app_id`, `device_id`, `device_ip` e `device_model`. 
Estas variáveis possuem muitos valores distintos, o que exige cuidados adicionais no pré-processamento, nomeadamente na codificação das categorias e na gestão
de categorias raras.

Assim, as observações do conjunto de dados refletem eventos reais de exposição a anúncios,
sendo adequadas para a construção 
de modelos preditivos supervisionados de classificação binária.


### Fonte de Dados
* **Dataset:** [Avazu CTR Prediction — Kaggle](https://www.kaggle.com/competitions/avazu-ctr-prediction)
* **Dimensão:** 40.428.967 registos × 24 colunas (dataset original); amostra de 5.000.000 registos utilizada no projeto, por sugestão da professora, com `random_state=42` para garantir reprodutibilidade.

## 2. Exploração (Milestone 2)
### Limpeza e Preparação

* A variável `C20` continha valores `-1` como marcador de dados omissos — foi aplicada imputação pela **moda**, estratégia justificada pela natureza categórica
da variável e pela robustez da moda face a distribuições assimétricas. Detalhes em `docs/M2_exploracao.md`.
* Colunas sem valor preditivo (`id`, `device_id`, `device_ip`) foram removidas para reduzir ruído e dimensionalidade.
* Os tipos de dados foram auditados e corrigidos para garantir conformidade com o perfil esperado.
* Foram criadas duas novas variáveis por *feature engineering*: `hora_do_dia` (extraída de `hour`) e `banner_area` (produto de `C15 × C16`).
* Variáveis categóricas de alta cardinalidade foram codificadas com **Target Encoding** (Leave-One-Out); as restantes com **Label Encoding**.


### Principais Conclusões (EDA)

> *Ver o gráfico de distribuição da variável `click` — `reports/figures/`*

* **Ponto-chave 1:** A variável alvo `click` está fortemente desequilibrada — aproximadamente **83% de não-cliques (0)** e **17% de cliques (1)**,
rácio de 1:5. Este desequilíbrio justifica a priorização do **AUC-ROC** como métrica principal.
* **Ponto-chave 2:** O CTR varia significativamente ao longo do dia, com picos nas primeiras horas da madrugada, tornando `hora_do_dia` uma variável preditiva relevante.
* **Ponto-chave 3:** As variáveis relacionadas com as dimensões do banner (`C16`, `banner_area`) apresentam correlação com a probabilidade de clique, confirmada posteriormente pelo modelo final.
* **Ponto-chave 4:** A matriz de correlação de Pearson não revelou pares com correlação superior a 0,85, confirmando a ausência de multicolinearidade crítica.

## 3. Modelação (Milestone 3)
### Abordagem Técnica

* **Modelos:** Regressão Logística (Baseline), Random Forest, XGBoost, XGBoost Otimizado (modelo final)
* **Métrica Principal:** AUC-ROC — escolhida pela capacidade de avaliar modelos em datasets desequilibrados, independentemente do limiar de decisão
* **Resultado Final:** O **XGBoost Otimizado** alcançou um **AUC-ROC de 0,7509** no conjunto de teste, superando o objetivo SMART de AUC-ROC > 0,75, 
com uma melhoria total de **+0,1097** face ao Baseline
* **Validação Cruzada (5-Fold):** Média de **0,7504 ± 0,0006**, confirmando estabilidade excecional do modelo

> *Ver o gráfico das Curvas ROC comparativas*
![Curva ROC de comparacao](reports/figures/curvas_roc_comparacao.png)  

## 4. Finalização (Milestone 4)

### Resumo de Resultados

O modelo **prevê se um utilizador vai clicar num anúncio com 75,3% de capacidade discriminativa (AUC-ROC)** — o que significa que, 
em 3 de cada 4 comparações entre uma impressão que será clicada e outra que não será, o modelo atribui a pontuação mais alta à impressão certa.
Face ao modelo de referência inicial (Regressão Logística, AUC-ROC = 0,6412), a melhoria acumulada foi de **+17,1% na capacidade preditiva**, 
atingindo o objetivo SMART definido no início do projeto.

A variável mais importante descoberta pelo modelo foi `banner_area` — a área visual do anúncio em píxeis²,
**criada durante este projeto** — com **32,6% do poder preditivo total**. Isto traduz-se numa recomendação direta:
A importância de `banner_area` indica que, nos dados analisados, a área visual do anúncio ajudou o modelo a distinguir impressões com maior 
e menor probabilidade de clique. Isto não prova que aumentar sempre a dimensão do anúncio cause mais cliques, mas mostra que esta variável tem valor preditivo relevante.


### Resposta ao Problema

O objetivo central do projeto era desenvolver um modelo capaz de prever cliques em anúncios digitais com AUC-ROC ≥ 0,75. 
Esse objetivo foi **alcançado**: o XGBoost Otimizado atingiu AUC-ROC = 0,7509 no conjunto de teste, confirmado por validação 
cruzada com IC 95%: [0,7493 – 0,7516].

Com esta solução, os anunciantes em plataformas de *Real-Time Bidding* podem concentrar os seus lances nas impressões com maior 
probabilidade de clique, reduzindo o desperdício de orçamento em impressões irrelevantes e aumentando o retorno sobre o investimento (ROI)
das campanhas digitais. As recomendações acionáveis identificadas — privilegiar banners de maior área visual, concentrar lances 
nas primeiras horas da madrugada (0h–6h) e focar em dispositivos móveis (`device_type = 0`) — podem ser implementadas diretamente 
nas estratégias de *bidding* sem necessidade de alterações técnicas adicionais.

### Recomendações de Inovação
1. **Implementar SMOTE** (*Synthetic Minority Over-sampling Technique*) para lidar melhor com o desequilíbrio de classes (1:5), 
reduzindo os 56.547 Falsos Negativos identificados na matriz de confusão e melhorando o *Recall* do modelo.
2. **Integrar dados sazonais e de calendário** (dia da semana, feriados, eventos desportivos) para refinar as previsões de CTR ao 
longo do tempo — dimensão que o dataset de apenas 10 dias não permite captar.
3. **Desenvolver uma interface de programação de aplicações** (FastAPI ou Flask) que disponibilize o
4.  modelo para classificação em tempo real, integrável em sistemas de *Real-Time Bidding*, e uma interface web em Streamlit para 
utilização por gestores de marketing sem necessidade de código.

### Apresentação Final (Pitch)

O vídeo de apresentação final foi estruturado como uma síntese narrativa do projeto, com o objetivo de comunicar o problema, a solução
desenvolvida, a prova técnica dos resultados e o valor prático da solução para publicidade digital. Esta apresentação complementa os relatórios
técnicos, traduzindo os resultados para uma linguagem acessível a uma audiência não técnica.

O *pitch* segue a estrutura definida para a Milestone 4, organizada em quatro momentos principais:

1. **O Problema:** apresentação do contexto da publicidade digital e da necessidade de prever a variável `click`, que indica se uma impressão 
publicitária gerou clique (`1`) ou não gerou clique (`0`). Nesta fase, é explicado o desafio do desequilíbrio da variável alvo, com cerca de 83% de não-cliques e 17% de cliques reais.

2. **A Solução:** descrição resumida do percurso técnico desenvolvido ao longo do projeto. Primeiro, os dados foram 
preparados na Milestone 2, através do tratamento de valores em falta, remoção de identificadores com baixo poder preditivo, 
codificação das variáveis categóricas e criação de novas variáveis como `hora_do_dia` e `banner_area`. Depois, na Milestone 3, 
foram testados diferentes modelos de classificação, começando pela Regressão Logística como *baseline* e evoluindo para Random Forest e XGBoost.

3. **A Prova:** apresentação dos resultados principais do modelo final. O XGBoost otimizado atingiu **AUC-ROC = 0.7509** no conjunto de teste, 
cumprindo o objetivo SMART definido na Milestone 1. O modelo também obteve **F1-Score = 0.4223** e apresentou uma melhoria acumulada de **+0.1097** 
em AUC-ROC face à Regressão Logística. No vídeo, esta parte corresponde ao momento de maior impacto, onde são mostradas as métricas principais, 
a matriz de confusão e a importância das variáveis.

4. **O Valor:** explicação da utilidade prática do modelo. A solução permite ordenar impressões por probabilidade de clique, apoiando decisões em campanhas digitais. 
Em vez de tratar todas as impressões como igualmente relevantes, o modelo ajuda a identificar contextos com maior potencial de interação. A variável `banner_area`,
criada durante o projeto, surgiu como a mais importante no modelo final, mostrando que a engenharia de atributos contribuiu para gerar conhecimento útil.

O vídeo inclui também uma componente de demonstração prática, com referência aos resultados produzidos no notebook, nomeadamente a matriz de confusão, 
as curvas ROC comparativas e o gráfico de importância das variáveis. Esta demonstração permite mostrar, de forma visual, onde o modelo acerta, onde ainda 
falha e que variáveis tiveram maior peso na previsão.

A apresentação foi preparada para ter uma duração máxima de 5 minutos, respeitando o formato exigido para a defesa final. A narrativa foi dividida entre os dois 
elementos do grupo, garantindo participação equilibrada e uma explicação progressiva: contexto e problema, preparação dos dados, comparação dos modelos, prova dos
resultados, valor de negócio, limitações e trabalhos futuros.

> **Vídeo de Apresentação:** [*inserir link após publicação*]

Para apoiar o momento de avaliação e antecipar dúvidas técnicas, foi também preparado um ficheiro de perguntas e respostas assíncronas. Esse ficheiro resume as 
decisões mais importantes do projeto, nomeadamente a escolha do AUC-ROC, a seleção do XGBoost como modelo final e as principais limitações da solução.

Para dúvidas técnicas sobre o modelo, consultar o ficheiro [`Q&A.md`](Q&A.md).

## Como Reproduzir este Projeto

1. Clone o repositório: `git clone "https://github.com/BernardoFernandes7/Projeto-Final"`
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute os notebooks na pasta `notebooks/` seguindo a ordem numérica.

> **Nota:** O dataset original (40M registos) deve ser descarregado diretamente do Kaggle. O notebook carrega automaticamente uma amostra de 5.000.000 registos com `random_state=42`.

**Instituição:** Coimbra Business School | ISCAC  
**Curso:** Licenciatura em Ciência de Dados para a Gestão  
**Unidade Curricular:** Projeto em Ciência de Dados  
**Professor Responsável:** Dora Melo (dmelo@iscac.pt)

