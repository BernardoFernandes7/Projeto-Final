# Milestone 1: Iniciação e Definição do Projeto
## 1. Descrição Detalhada do Problema

Contexto do Setor: O projeto insere-se no setor de Publicidade Digital (AdTech). No ecossistema de publicidade online, as empresas utilizam sistemas de Real-Time Bidding (RTB) para decidir, em milissegundos, se devem apresentar um anúncio a um utilizador e quanto devem pagar por essa impressão.

Relevância: Prever a Taxa de Clique (Click-Through Rate - CTR) é o "santo graal" deste setor. Se uma empresa conseguir prever com precisão se um utilizador vai clicar num anúncio, ela pode:

Otimizar o Orçamento: Gastar apenas em impressões com alta probabilidade de conversão.

Melhorar a Experiência do Utilizador: Mostrar anúncios que sejam realmente relevantes, reduzindo o "ruído" digital.

Aumentar a Receita: Para os publishers, um CTR mais alto traduz-se em maior rentabilidade do inventário publicitário.

## 2. Objetivos SMART
Objetivo 1

Desenvolver e avaliar um modelo preditivo capaz de estimar a probabilidade de clique em anúncios online, utilizando o dataset Avazu CTR Prediction.

Objetivo 2

Identificar as 5 variáveis de maior impacto (ex: site_category, device_type, hour) na decisão de clique, permitindo criar um perfil de segmentação de audiência mais eficaz.

Objetivo 3

Comparar diferentes algoritmos de aprendizagem automática (ex.: Regressão Logística, Random Forest, Gradient Boosting), identificando o modelo com melhor desempenho preditivo.

Objetivo 4

Avaliar os modelos através de métricas adequadas a problemas de classificação binária desbalanceada, como:

AUC-ROC

Log-Loss

Precision, Recall e F1-score

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
