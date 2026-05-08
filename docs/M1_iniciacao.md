# Milestone 1: Iniciação e Entendimento do Negócio

## 1. Descrição do Problema

A publicidade digital representa hoje um dos maiores mercados globais, onde o sucesso depende da precisão com que se prevê o comportamento do utilizador. Grande parte dos anúncios exibidos em aplicações e sites são decididos através de leilões instantâneos (*Real-Time Bidding*), nos quais a estimativa da Taxa de Clique (CTR — Click-Through Rate) é a métrica central. Para os anunciantes, prever esta probabilidade é essencial para otimizar o investimento e reduzir o desperdício em impressões irrelevantes; para as plataformas, é a chave para maximizar a receita.

Este projeto aborda precisamente esse desafio utilizando o dataset Avazu CTR Prediction. Composto por dados reais de publicidade móvel recolhidos ao longo de 10 dias, o conjunto de dados inclui cerca de 40 milhões de registos e 24 variáveis contextuais — como o tipo de dispositivo, posição do anúncio e ambiente de navegação. Do ponto de vista da ciência de dados, enfrentamos um problema de classificação binária supervisionada, onde o objetivo é prever a variável click.

O maior desafio identificado reside no forte desequilíbrio das classes: apenas cerca de 17% dos dados correspondem a cliques reais. Este cenário torna a previsão complexa, uma vez que modelos simplistas poderiam apresentar uma precisão elevada sem possuírem qualquer utilidade prática. Por isso, o tratamento deste desbalanceamento e a seleção rigorosa de métricas como o AUC-ROC foram decisões fundamentais para garantir a eficácia e o valor de negócio da solução desenvolvida.

## 2. Objetivos SMART

1. **Objetivo 1:** Desenvolver um modelo de classificação binária capaz de prever se um utilizador irá clicar num anúncio, atingindo um **AUC-ROC mínimo de 0.75** no conjunto de teste, utilizando o *dataset* Avazu CTR Prediction até ao final do Milestone 3.

2. **Objetivo 2:** Identificar as **5 variáveis mais determinantes** para a previsão do clique (através de *Feature Importance*) e diagnosticar os perfis de erro do modelo, fornecendo recomendações acionáveis sobre os contextos de anúncios com maior probabilidade de conversão, até à conclusão da fase de modelação.

## 3. Perguntas de Investigação
Que variáveis contextuais — características do dispositivo, hora do dia, posição do anúncio ou categoria do site — têm maior peso na decisão de clique, e como esse peso se traduz em valor para a otimização de lances em RTB?

De que forma o forte desequilíbrio entre cliques (~17%) e não-cliques (~83%) afeta a capacidade preditiva dos modelos testados, e qual a estratégia mais eficaz para o mitigar sem introduzir data leakage?

O modelo consegue identificar perfis de utilizador ou contextos de exibição com probabilidade de clique sistematicamente acima ou abaixo da média global — e que implicação prática teria esse conhecimento para um anunciante?

## 4. Metodologia de Gestão (PBL)

* **Divisão de Tarefas:**
  * **Bernardo:** Responsável pela Engenharia de dados, pré-processamento e feature engineering.
  * **Hugo:** Responsável pela Modelação, otimização de hiperparâmetros e avaliação de modelos.
* **Ferramentas de Colaboração:** GitHub Projects, reuniões semanais via Discord.

### 4. Análise de Viabilidade dos Dados

* **Disponibilidade:** Os dados provêm do dataset público da competição Avazu CTR Prediction no Kaggle. Devido à sua dimensão original (40,4 milhões de registos), os dados foram descarregados e processados através de uma amostra aleatória representativa de 5.000.000 de registos, garantindo a viabilidade técnica no ambiente de trabalho (Kaggle/GitHub) e a reprodutibilidade dos resultados.

* **Qualidade Inicial:** O dataset apresenta uma qualidade elevada, sem valores nulos convencionais. Contudo, identificou-se que os dados em falta estão mascarados com o valor `-1` em colunas anónimas (C14–C21), com maior incidência na variável `C20`. Estas omissões, juntamente com a necessidade de transformar a coluna `hour` para extrair componentes temporais, serão o foco de tratamento na Milestone 2.

* **Ética:** Os dados cumprem as normas de privacidade, uma vez que o dataset é totalmente anonimizado. Identificadores sensíveis como `device_id` e `device_ip` foram substituídos por *hashes*, impossibilitando a identificação de utilizadores reais. Sendo um dataset público para fins académicos e de competição, não apresenta conflitos com o RGPD.

### Dicionário de Variáveis

| Variável | Tipo de variável | Intervalo dos dados | Descrição |
| :--- | :--- | :--- | :--- |
| `id` | Categórica nominal | `521159400000` a `18446740000000000000` | Identificador único do registo. Não deve ser usado como variável preditiva direta. |
| `click` | Numérica discreta binária | `0` a `1` | Variável alvo. Indica se o anúncio foi clicado (`1`) ou não clicado (`0`). |
| `hour` | Numérica discreta temporal | `14102100` a `14103023` | Data e hora da impressão do anúncio, no formato `YYMMDDHH`. |
| `C1` | Numérica discreta | `1001` a `1012` | Variável anonimizada da base Avazu. Os valores representam códigos associados ao contexto do anúncio. |
| `banner_pos` | Numérica discreta | `0` a `7` | Representa a posição do banner na página. |
| `site_id` | Categórica nominal | `4737` categorias distintas | Identificador do site onde o anúncio foi exibido. |
| `site_domain` | Categórica nominal | `7745` categorias distintas | Domínio do site onde ocorreu a impressão do anúncio. |
| `site_category` | Categórica nominal | `26` categorias distintas | Categoria temática do site. |
| `app_id` | Categórica nominal | `8552` categorias distintas | Identificador da aplicação mobile onde o anúncio foi exibido. |
| `app_domain` | Categórica nominal | `559` categorias distintas | Domínio associado à aplicação mobile. |
| `app_category` | Categórica nominal | `36` categorias distintas | Categoria da aplicação mobile. |
| `device_id` | Categórica nominal | `2686408` categorias distintas | Identificador anonimizado do dispositivo. |
| `device_ip` | Categórica nominal | `6729486` categorias distintas | Endereço IP anonimizado do dispositivo. |
| `device_model` | Categórica nominal | `8251` categorias distintas | Modelo do dispositivo utilizado pelo utilizador. |
| `device_type` | Numérica discreta | `0` a `5` | Representa o tipo de dispositivo utilizado. |
| `device_conn_type` | Numérica discreta | `0` a `5` | Representa o tipo de ligação à internet utilizada pelo dispositivo. |
| `C14` | Numérica discreta | `375` a `24052` | Variável anonimizada relacionada com o contexto do anúncio. |
| `C15` | Numérica discreta | `120` a `1024` | Variável associada à dimensão ou configuração visual do anúncio. |
| `C16` | Numérica discreta | `20` a `1024` | Variável associada à dimensão ou configuração visual do anúncio. |
| `C17` | Numérica discreta | `112` a `2758` | Variável anonimizada relacionada com o contexto do anúncio. |
| `C18` | Numérica discreta | `0` a `3` | Variável anonimizada relacionada com o contexto do anúncio. |
| `C19` | Numérica discreta | `33` a `1959` | Variável anonimizada relacionada com o contexto do anúncio. |
| `C20` | Numérica discreta | `-1` a `100248` | Variável anonimizada relacionada com o contexto do anúncio. O valor `-1` pode representar valor em falta mascarado. |
| `C21` | Numérica discreta | `1` a `255` | Variável anonimizada relacionada com o contexto do anúncio. |

## 5. Cronograma Interno
| Fase | Data Limite | Entregável Esperado |
| :--- | :--- | :--- |
| M1: Iniciação | 24/02/2026 | Repositório estruturado e Plano de Projeto. |
| M2: Exploração | 24/03/2026 | Notebook de EDA e Dados Processados. |
| M3: Modelação | 23/04/2026 | Comparação de algoritmos e métricas. |
| M4: Finalização| [Data] | Pitch e Relatório Final. | 

---

*Data de última atualização: 23/04/2026*
