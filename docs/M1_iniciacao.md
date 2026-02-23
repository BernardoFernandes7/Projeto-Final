# Milestone 1: Iniciação e Definição do Projeto
## 1. Descrição Detalhada do Problema

A previsão de cliques em anúncios online (Click-Through Rate Prediction – CTR Prediction) consiste na estimação da probabilidade de um utilizador clicar num determinado anúncio digital, com base em informação histórica, comportamental e contextual. Este problema enquadra-se na área da ciência de dados aplicada ao marketing digital e à publicidade programática, sendo normalmente formulado como uma tarefa de classificação binária, onde a variável alvo assume valor 1 (clique) ou 0 (não clique).

No contexto atual da publicidade digital, decisões automatizadas são tomadas em milissegundos para determinar qual anúncio deve ser apresentado a cada utilizador. A eficácia destas decisões depende fortemente da precisão dos modelos preditivos utilizados. Assim, a previsão da taxa de cliques assume relevância económica significativa, pois influencia diretamente o retorno sobre investimento (ROI) das campanhas publicitárias e a eficiência na alocação de recursos.

Do ponto de vista técnico, este problema apresenta desafios importantes:

Elevado volume de dados (milhões de registos);

Grande número de variáveis categóricas com alta cardinalidade;

Forte desbalanceamento entre classes;

Necessidade de modelos capazes de capturar interações complexas.

O dataset utilizado neste projeto corresponde ao conjunto de treino do desafio Avazu CTR Prediction, amplamente utilizado na literatura para estudo de problemas de previsão de cliques em grande escala.

## 2. Objetivos SMART
Objetivo 1

Desenvolver e avaliar um modelo preditivo capaz de estimar a probabilidade de clique em anúncios online, utilizando o dataset Avazu CTR Prediction.

Objetivo 2

Implementar um pipeline completo de ciência de dados, incluindo:

Análise exploratória dos dados (EDA);

Tratamento e preparação do dataset;

Engenharia de atributos;

Treino e validação de modelos de classificação.

Objetivo 3

Comparar diferentes algoritmos de aprendizagem automática (ex.: Regressão Logística, Random Forest, Gradient Boosting), identificando o modelo com melhor desempenho preditivo.

Objetivo 4

Avaliar os modelos através de métricas adequadas a problemas de classificação binária desbalanceada, como:

AUC-ROC

Log-Loss

Precision, Recall e F1-score

## 3. Perguntas de Investigação

Quais variáveis apresentam maior influência na probabilidade de clique?

Qual modelo apresenta melhor capacidade de generalização?

Como o contexto de exibição (dispositivo, horário, tipo de site/app) afeta a probabilidade de clique?

O modelo mantém desempenho consistente em subconjuntos distintos do dataset?

Como melhorias na capacidade preditiva podem otimizar decisões de segmentação publicitária?

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

O dataset encontra-se disponível na plataforma Kaggle e foi carregado com sucesso no ambiente de desenvolvimento. Devido à sua dimensão (aproximadamente 40 milhões de registos), a análise exploratória inicial foi realizada com uma amostra representativa de 1 milhão de observações, garantindo viabilidade computacional.

Qualidade Inicial

A inspeção preliminar do ficheiro train.csv permitiu identificar:

Elevado número de variáveis categóricas;

Presença de identificadores codificados numericamente;

Variável alvo binária (click);

Forte desbalanceamento entre classes (maior predominância de não cliques).

Foi realizada verificação de:

Valores nulos;

Registos duplicados;

Tipos de dados;

Valores infinitos;

Estatísticas descritivas numéricas e categóricas.

Não foram identificados valores nulos na amostra analisada. A variável alvo apresenta distribuição desbalanceada, característica típica de problemas de CTR.

Ética

Os dados encontram-se anonimizados, não permitindo identificação direta dos utilizadores. O projeto será conduzido exclusivamente para fins académicos, respeitando princípios de ética em ciência de dados e conformidade com o RGPD.

## 6. Descrição Técnica da Inspeção Inicial

A análise exploratória foi realizada utilizando a biblioteca pandas, recorrendo aos métodos:

df.head() – visualização inicial dos registos;

df.info() – estrutura e tipos de dados;

df.describe() – estatísticas descritivas;

df.isnull().sum() – verificação de valores nulos;

df.duplicated().sum() – verificação de duplicados.

O dataset train.csv contém dezenas de variáveis que descrevem:

Informações temporais (ex.: hour);

Identificadores de site e aplicação (site_id, app_id);

Informações do dispositivo (device_type, device_ip);

Categorias de anúncio (ex.: C1, C14, C15, etc.);

Variável alvo binária click.

A inspeção revelou:

Ausência de valores nulos na amostra analisada;

Ausência de valores infinitos;

Predominância de variáveis categóricas codificadas;

Forte desbalanceamento da variável click;

Alta cardinalidade em múltiplas variáveis (especialmente identificadores).

Dada a dimensão do dataset completo (~40 milhões de registos), optou-se por trabalhar inicialmente com amostragem para viabilizar a análise exploratória e o desenvolvimento preliminar dos modelos.

## 7. Cronograma Interno
Fase	Data Limite	Entregável
-M1: Iniciação	24/02/2026	Plano de Projeto
-M2: Exploração	[Data]	Notebook EDA e Dados Tratados
-M3: Modelação	[Data]	Comparação de Modelos
-M4: Finalização	[Data]	Relatório Final e Pitch

Data da última atualização: 23/02/2026
