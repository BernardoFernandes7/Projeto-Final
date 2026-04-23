# *Milestone* 1: Iniciação e Definição do Projeto

> *Data de última atualização: Abril 2026*

---

## 1. Descrição Detalhada do Problema

A publicidade digital é hoje uma das principais formas de monetização da internet. Grande parte dos anúncios que vemos em aplicações e sites são decididos em tempo real — quando uma página carrega, existe um leilão instantâneo (*Real-Time Bidding*) que determina qual anúncio é exibido e a que utilizador. Neste contexto, os anunciantes precisam de decidir quanto estão dispostos a pagar por cada impressão, e essa decisão deveria ser baseada numa estimativa da probabilidade de o utilizador clicar no anúncio.

É exatamente este o problema que o nosso projeto aborda. Trabalhamos com o *dataset* Avazu CTR Prediction, disponível no Kaggle, que contém registos reais de impressões publicitárias recolhidas ao longo de 10 dias em dispositivos móveis. Com cerca de 40 milhões de registos e 24 variáveis contextuais, o *dataset* documenta o comportamento de utilizadores perante anúncios em aplicações e *sites* móveis, incluindo informação sobre o tipo de dispositivo, o ambiente de navegação, a posição do anúncio na página e a hora de exibição.

Do ponto de vista da ciência de dados, trata-se de um problema de **classificação binária supervisionada**. A variável objetivo é `click`, que assume o valor 1 quando o utilizador clicou no anúncio e 0 quando não clicou. O modelo deve aprender, a partir das variáveis contextuais, a distinguir impressões com elevada probabilidade de clique das restantes, permitindo que os anunciantes otimizem os seus lances e reduzam o desperdício de orçamento em impressões irrelevantes.

O principal desafio deste problema é o forte desequilíbrio da variável alvo: aproximadamente 83% dos registos correspondem a não-cliques e apenas 17% a cliques. Isto significa que um modelo que simplesmente previsse sempre "não clique" teria uma precisão aparente de 83%, sem qualquer utilidade real. Por isso, a escolha das métricas de avaliação e o tratamento deste desequilíbrio são decisões centrais no nosso trabalho.

### Dicionário de Dados

| Variável | Tipo | Descrição |
| :--- | :--- | :--- |
| `id` | Numérico | Identificador único da impressão |
| `click` | Binário | **Variável alvo** — 1: clique; 0: não clique |
| `hour` | Numérico | Marcação temporal (formato YYMMDDHH) |
| `C1` | Numérico | Variável anónima de contexto |
| `banner_pos` | Numérico | Posição do anúncio na página |
| `site_id` | Alfanumérico | Identificador do *site* |
| `site_domain` | Alfanumérico | Domínio do *site* |
| `site_category` | Alfanumérico | Categoria temática do *site* |
| `app_id` | Alfanumérico | Identificador da aplicação |
| `app_domain` | Alfanumérico | Domínio da aplicação |
| `app_category` | Alfanumérico | Categoria temática da aplicação |
| `device_id` | Alfanumérico | Identificador do dispositivo |
| `device_ip` | Alfanumérico | Endereço IP do utilizador |
| `device_model` | Alfanumérico | Modelo do dispositivo móvel |
| `device_type` | Numérico | Tipo de dispositivo (ex.: *smartphone*, *tablet*) |
| `device_conn_type` | Numérico | Tipo de ligação (ex.: 3G, *Wi-Fi*) |
| `C14` a `C21` | Numérico | Variáveis anónimas relativas ao anúncio |

---

## 2. Objetivo SMART

**Objetivo:** Desenvolver um modelo de classificação binária capaz de prever a probabilidade de um utilizador clicar num anúncio *mobile*, usando o *dataset* Avazu (amostra de 5 milhões de registos), atingindo um AUC-ROC superior a 0,75 no conjunto de teste isolado, até à data de entrega do *Milestone* 3 (23 de abril de 2026).

**Fundamentação SMART:**

- **Específico (*Specific*):** o objetivo centra-se na previsão da variável binária `click`, num contexto bem delimitado — publicidade *mobile* em ambiente de *Real-Time Bidding*.
- **Mensurável (*Measurable*):** o critério de sucesso é quantificável: AUC-ROC > 0,75 no conjunto de teste. Esta métrica é a oficial da competição Avazu e é robusta ao desequilíbrio de classes, ao contrário da *Accuracy* simples.
- **Atingível (*Achievable*):** o *dataset* é público, os dados estão disponíveis no Kaggle, e algoritmos como *Random Forest* e *XGBoost* são reconhecidos na literatura como adequados para este tipo de problema (He et al., 2014).
- **Relevante (*Relevant*):** um modelo capaz de ordenar impressões por probabilidade de clique tem valor direto para anunciantes e plataformas de *RTB*, reduzindo custo por aquisição e aumentando a eficiência das campanhas.
- **Temporal (*Time-bound*):** delimitado pela data de entrega do *Milestone* 3, a 23 de abril de 2026.

---

## 3. Perguntas de Investigação

Estas questões guiarão a análise de dados para entender os padrões de comportamento dos utilizadores e a eficácia dos anúncios:

Sazonalidade Temporal: Existe algum período específico do dia ou da semana em que a probabilidade de clique (click) aumenta significativamente? (Análise baseada na variável hour).

Influência do Posicionamento: Até que ponto a posição do anúncio na página (banner_pos) é um fator determinante para o CTR? Estará o topo da página sempre correlacionado com mais cliques?

Ambiente de Navegação (App vs. Site): Existe uma diferença estatisticamente relevante na taxa de clique entre utilizadores que navegam via aplicações (app_id) comparativamente a sites (site_id)?

Perfil de Dispositivo e Conectividade: De que forma o tipo de dispositivo (device_type) e a qualidade da ligação à internet (device_conn_type) influenciam a propensão do utilizador para interagir com o anúncio?

Segmentação de Conteúdo: Quais são as categorias de sites (site_category) ou aplicações (app_category) que apresentam o melhor desempenho em termos de conversão, e quais devem ser evitadas para otimizar o investimento?

## 4. Metodologia de Gestão (PBL)
Divisão de Tarefas

Hugo: Responsável pela preparação e tratamento dos dados (EDA, limpeza, encoding, engenharia de atributos).

Bernardo: Responsável pela modelação estatística e avaliação dos algoritmos.

Colaboração conjunta nas decisões metodológicas e validação dos resultados.

Ferramentas

GitHub (controlo de versões)

WhatsApp (comunicação)

Bibliotecas previstas

pandas, numpy – manipulação de dados

matplotlib, seaborn – visualização

scikit-learn – modelação e métricas

Possível integração futura de xgboost ou lightgbm, dada a natureza do problema

## 5. Análise de Viabilidade dos Dados
Disponibilidade

Os dados estão totalmente disponíveis e prontos para processamento, localizados no diretório /kaggle/input/. A estrutura de ficheiros inclui os conjuntos de treino (train.csv) e teste (test.csv), além do modelo de submissão. O acesso via Python foi validado com sucesso através da biblioteca Pandas, utilizando o método de leitura por blocos (chunksize) para garantir a estabilidade do sistema perante o volume de dados.

Qualidade Inicial

A integridade técnica dos dados é excelente, apresentando zero valores nulos em todas as 24 colunas analisadas ao longo dos 40.428.967 registos. No entanto, foram identificados pontos que exigem tratamento na Milestone 2: a variável hour está formatada como um número inteiro, necessitando de conversão para datetime para extração de tendências temporais. Além disso, a elevada cardinalidade de variáveis categóricas, como o device_ip (6.729.486 valores únicos) e o device_id (2.686.408 valores únicos), exigirá a aplicação de técnicas de Feature Hashing para viabilizar a modelação sem exceder a memória disponível.

Ética

O dataset cumpre as normas do RGPD e os princípios de privacidade por design. Todas as variáveis de identificação, como o ID do dispositivo, o IP e os IDs de aplicações ou sites, foram devidamente anonimizadas através de representações hexadecimais. Esta técnica de ofuscação impede a reidentificação de utilizadores individuais ou a exposição de comportamentos de navegação privados, permitindo que a análise se foque exclusivamente em padrões estatísticos agregados e anónimos.

## 6. Descrição Técnica da Inspeção Inicial

Processamento de Big Data- Devido ao tamanho do dataset, será utilizada a leitura em blocos (chunking) e a conversão de tipos de dados para reduzir o consumo de memória RAM.

Feature Engineering- Conversão da coluna hour para o formato datetime para extrair padrões temporais (hora do dia, dia da semana). Aplicação de Feature Hashing (ou Hashing Trick) para lidar com as categorias de alta cardinalidade sem explodir a dimensão da matriz.

Algoritmos Candidatos- Serão testados modelos de Regressão Logística (como baseline), seguidos de algoritmos de Gradient Boosting (como LightGBM ou XGBoost), conhecidos pela sua eficiência com dados tabulares volumosos.

Ambiente- Execução em ambiente Cloud (Kaggle/Colab) aproveitando a capacidade de CPU/GPU disponível.

## 7. Cronograma Interno
Fase	Data Limite	Entregável
| **Fase**            | **Data Limite** | **Entregável Esperado**                                    |
| ------------------- | --------------- | ---------------------------------------------------------- |
| **M1: Iniciação**   | 24/02/2026      | Repositório estruturado e Plano de Projeto                 |
| **M2: Exploração**  | [Data]          | Notebook de Análise Exploratória (EDA) e dados processados |
| **M3: Modelação**   | [Data]          | Comparação de algoritmos e avaliação de métricas           |
| **M4: Finalização** | [Data]          | Relatório Final e apresentação (Pitch)                     |


Data da última atualização: 24/02/2026
