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

1. Que variáveis contextuais — características do dispositivo, hora do dia, posição do anúncio ou categoria do *site* — têm maior peso na decisão de clique, e como esse peso se traduz em valor para a otimização de lances em *RTB*?

2. De que forma o forte desequilíbrio entre cliques (~17%) e não-cliques (~83%) afeta a capacidade preditiva dos modelos testados, e qual a estratégia mais eficaz para o mitigar sem introduzir *data leakage*?

3. O modelo consegue identificar perfis de utilizador ou contextos de exibição com probabilidade de clique sistematicamente acima ou abaixo da média global — e que implicação prática teria esse conhecimento para um anunciante?

---

## 4. Metodologia de Gestão

### Divisão de Tarefas

| Membro | Responsabilidade principal |
| :--- | :--- |
| Bernardo Fernandes | Engenharia de dados, pré-processamento e *feature engineering* |
| Hugo Grou | Modelação, otimização de hiperparâmetros e avaliação de modelos |

As tarefas de documentação, revisão do código e atualização do GitHub são partilhadas por ambos os elementos.

### Ferramentas de Colaboração

- **GitHub Projects** — quadro Kanban com *issues* e tarefas por *milestone*
- **Kaggle Notebooks** — ambiente de desenvolvimento e execução do código
- **GitHub** — versionamento, documentação e entrega de cada *milestone*

---

## 5. Análise de Viabilidade dos Dados e Descobertas Iniciais

**Disponibilidade:** o *dataset* está publicamente disponível no Kaggle e foi adicionado diretamente ao ambiente de execução dos *notebooks*. Não requer acesso especial nem transferência manual de ficheiros.

**Qualidade inicial — valores reais da inspeção:** a primeira exploração dos dados revelou os seguintes factos que definem o ponto de partida para o *Milestone* 2:

Não existem valores nulos explícitos em nenhuma das 24 colunas. No entanto, a coluna `C20` tem 2.344.248 registos com o valor `-1` (46,88% da amostra), que correspondem a dados em falta codificados de forma não *standard* — uma prática comum em *datasets* de publicidade. As restantes colunas anónimas (C14–C21) apresentam percentagens residuais de `-1`, abaixo de 1%.

Em relação a observações repetidas, não existem duplicados totais (linhas completamente iguais), mas existem 1.130.966 duplicados lógicos (22,6% da amostra), entendidos como o mesmo utilizador a ver o mesmo anúncio no mesmo momento. Esta situação é esperada em contexto de *RTB* e os registos foram mantidos.

A variável alvo `click` está fortemente desequilibrada: 83,03% dos registos são não-cliques (0) e apenas 16,97% são cliques (1). Este desequilíbrio é determinante para a escolha das métricas de avaliação — a *Accuracy* simples seria enganosa, pelo que o AUC-ROC foi adotado como métrica principal.

As variáveis categóricas de identificação (`site_id`, `app_id`, `device_model`, entre outras) têm alta cardinalidade (milhares de valores únicos), o que impede o uso direto de *One-Hot Encoding* e orienta a fase de preparação para técnicas como *Frequency Encoding*.

O volume total do *dataset* (~40 milhões de registos) inviabiliza o carregamento direto em memória no Kaggle, pelo que o *Milestone* 2 arrancou com uma amostra aleatória de 5 milhões de registos (`random_state=42`), por sugestão da docente.

**Ética e privacidade:** os dados estão anonimizados — identificadores como `device_id` e `device_ip` foram sujeitos a *hashing* antes da publicação. Não contêm dados pessoais identificáveis, pelo que não levantam questões de conformidade com o RGPD no contexto deste projeto académico.

---

## 6. Cronograma Interno

| *Milestone* | Data Limite | Entregável |
| :--- | :--- | :--- |
| M1 — Iniciação | 24 fev. 2026 | Repositório estruturado, README e M1_iniciacao.md |
| M2 — Exploração | 24 mar. 2026 | *Notebook* de EDA, dados processados e M2_exploracao.md |
| M3 — Modelação | 23 abr. 2026 | *Notebook* de modelação, resultados e M3_modelacao.md |
| M4 — Finalização | A definir | Apresentação final e relatório |

---

## Referências

He, X., Pan, J., Jin, O., Xu, T., Liu, B., Xu, T., Shi, Y., Atallah, A., Herbrich, R., Bowers, S., & Candela, J. Q. (2014). Practical lessons from predicting clicks on ads at Facebook. *Proceedings of the 8th International Workshop on Data Mining for Online Advertising*, 1–9. https://doi.org/10.1145/2648584.2648589

Kaggle. (s.d.). *Avazu CTR prediction* [*Dataset*]. https://www.kaggle.com/datasets/madhu41289/avazu-ctr-prediction-exp

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesneau, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.
