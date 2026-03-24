# Milestone 2: Análise Exploratória e Engenharia de Atributos

> **Nota de Revisão:** Este documento pressupõe que o dataset já foi identificado e descrito no ficheiro `docs/M1_iniciacao.md`. Caso precise de consultar o significado original das variáveis, deve consultar essa Milestone.

---

## 1. Análise Exploratória de Dados (EDA)

A Análise Exploratória de Dados (EDA) constitui uma fase fundamental no ciclo de vida de qualquer projeto de Ciência de Dados. Segundo Tukey (1977), a EDA representa um conjunto de técnicas estatísticas e visuais que permite compreender a estrutura interna de um conjunto de dados antes de qualquer modelação preditiva. No contexto do presente projeto, a EDA foi conduzida de forma incremental sobre os 40 milhões de registos do dataset Avazu, utilizando processamento em *chunks* de modo a superar as limitações de memória RAM do ambiente Kaggle.

---

### 1.1. Distribuição da Variável Alvo

A variável alvo `click` é binária (0 = não clicou, 1 = clicou) e apresenta um forte desequilíbrio de classes.

> **Factos importantes:** A variável alvo `click` está fortemente desequilibrada, com aproximadamente **83% de não-cliques (0)** e **17% de cliques (1)**. Este desequilíbrio é esperado em problemas de CTR e justifica o uso de métricas como AUC-ROC, Log Loss e F1-score em detrimento da simples Accuracy, que seria enganosa neste contexto — um modelo que previsse sempre "0" teria 83% de *accuracy* sem qualquer utilidade preditiva real, fenómeno denominado *accuracy paradox* (Japkowicz & Stephen, 2002).


Ver figura Distribuição da variável alvo 'click' 

---

### 1.2. Correlações Relevantes

A matriz de correlação de Pearson foi calculada de forma incremental sobre os 40 milhões de registos para identificar as variáveis com maior relação linear com a variável alvo. O coeficiente de Pearson mede a força e a direção da associação linear entre duas variáveis, assumindo valores entre −1 e 1, sendo 0 indicativo de ausência de relação linear (Cohen, Cohen, West & Aiken, 2003). Foi escolhido em detrimento de outras métricas de associação (como o coeficiente de Spearman) por ser computacionalmente eficiente em datasets de grande dimensão.


Ver figura Matriz de Correlação de Pearson


* **`C14` vs. `click`:** A variável anónima `C14` apresenta a correlação mais elevada com `click`, sugerindo que representa uma característica do anúncio ou do utilizador com forte impacto na decisão de clique. O *scatter plot* de CTR por C14 confirma que determinados valores de C14 têm CTR significativamente acima da média global de 17%.

<!-- INSERIR FIGURA 3 -->
<!-- Barplot do CTR médio por valor de C14 (top 20 valores mais frequentes) -->
<!-- reports/figures/fig3_ctr_por_c14.png -->

* **`hora_do_dia` vs. `click`:** A análise bivariada do CTR por hora do dia revela variações significativas ao longo das 24 horas, com as primeiras horas da madrugada (0h–3h) a apresentarem CTR mais elevado, contrariando a intuição inicial de que o período diurno seria mais eficaz.

<!-- INSERIR FIGURA 4 -->
<!-- Gráfico de linha do CTR médio por hora do dia (0–23h), com destaque nas horas 0h–3h -->
<!-- reports/figures/fig4_ctr_por_hora.png -->

* **`banner_pos` vs. `click`:** A posição do banner na página influencia o CTR, mas a relação não é linear — o topo da página nem sempre corresponde ao maior CTR, dependendo do tipo de dispositivo utilizado.

* **`device_type` vs. `click`:** Dispositivos móveis apresentam padrões de clique distintos dos PCs, com diferenças relevantes no CTR médio por tipo de dispositivo.

---

## 2. Qualidade dos Dados e Limpeza

---

### 2.1. Tratamento de Dados em Falta

#### Valores em falta padrão

Foi realizada uma auditoria completa de valores em falta (`NaN`, `None`, células vazias) sobre todas as 24 colunas do dataset. O resultado confirmou a **ausência total de valores em falta padrão** — nenhuma célula estava vazia ou nula. Esta característica é típica de datasets de log de comportamento em tempo real, onde os sistemas de registo garantem a completude dos campos.

#### Valores em falta mascarados (*missings* codificados)

O dataset Avazu utiliza o valor `-1` para representar informação desconhecida nas colunas anónimas — uma prática comum em datasets de publicidade computacional que codifica a ausência de informação como um inteiro, contornando problemas de serialização de dados nulos (He et al., 2014).

<!-- INSERIR FIGURA 5 -->
<!-- Barplot horizontal da percentagem de valores -1 por coluna (destaca C20) -->
<!-- reports/figures/fig5_missings_codificados.png -->

* **Colunas afetadas:** `C20`
* **Estratégia adotada:** Substituição dos valores `-1` pela **moda global** da coluna (calculada sobre os 40 milhões de registos, excluindo os próprios `-1`). Optou-se pela moda em vez da média ou mediana porque `C20` é uma variável categórica codificada numericamente — a média de categorias não tem significado semântico, enquanto a moda representa o valor mais frequente e, portanto, mais representativo da distribuição real da população.

---

### 2.2. Outliers e Inconsistências

#### Outliers numéricos

A deteção de *outliers* foi realizada através do método **IQR (Intervalo Interquartil)**, um critério robusto e não paramétrico proposto por Tukey (1977). Segundo este critério, valores inferiores a Q1 − 1,5 × IQR ou superiores a Q3 + 1,5 × IQR são classificados como atípicos. Foram identificados *outliers* nas colunas `C1`, `banner_pos` e `C14`.

<!-- INSERIR FIGURA 6 -->
<!-- Boxplots das variáveis C1, C14, banner_pos e C15/C16 para visualizar os outliers identificados pelo IQR -->
<!-- reports/figures/fig6_boxplots_outliers.png -->

Dado que o dataset Avazu é um dataset de comportamento real de utilizadores, os valores extremos não representam erros de medição, mas sim **comportamentos legítimos e raros**. Optou-se por **não remover** estes *outliers* para não perder informação potencialmente relevante para o modelo.

#### Erros de tipo de dados

As colunas categóricas (`site_id`, `site_domain`, `site_category`, `app_id`, `app_domain`, `app_category`, `device_model`) foram confirmadas e convertidas explicitamente para tipo `string` antes do *encoding*, garantindo consistência no processamento e prevenindo erros silenciosos quando inteiros são interpretados como categorias numéricas por bibliotecas como o scikit-learn (Pedregosa et al., 2011).

---

## 3. Engenharia de Atributos (Feature Engineering)

A Engenharia de Atributos é o processo de transformar dados brutos em representações mais informativas para os algoritmos de aprendizagem automática. Segundo Domingos (2012), *"more data beats a clever algorithm, but good features beat more data"* — destacando a centralidade desta fase no desempenho preditivo dos modelos. Foram realizadas três categorias de transformações: codificação, normalização e criação de novas variáveis.

---

### 3.1. Transformações Realizadas

#### Encoding — Codificação por Rótulo (*Label Encoding*)

Aplicado nas 7 colunas categóricas de alta cardinalidade: `site_id`, `site_domain`, `site_category`, `app_id`, `app_domain`, `app_category` e `device_model`. Esta técnica atribui um inteiro único a cada categoria, preservando a estrutura do dataset sem expansão dimensional.

A opção pelo **Label Encoding** em detrimento do *One-Hot Encoding* justifica-se porque estas colunas têm **milhares de categorias únicas** (*high cardinality*). O *One-Hot Encoding* geraria dezenas de milhares de novas colunas, tornando o dataset inviável em termos de memória para 40 milhões de registos. Modelos baseados em árvore como XGBoost e Random Forest, que serão utilizados no Milestone 3, lidam nativamente com Label Encoding sem pressupor uma ordenação cardinal (He et al., 2014).

#### Escalonamento — Padronização (*StandardScaler*)

Aplicado nas colunas numéricas `C1`, `C14`, `C15`, `C16`, `C17`, `C18`, `C19`, `C20` e `C21`. Esta transformação centra cada variável em torno da média zero e escala pelo desvio padrão: **z = (x − μ) / σ**.

A padronização garante que todas as variáveis contribuem equitativamente para os algoritmos sensíveis à escala (como Regressão Logística), evitando que variáveis com valores absolutos maiores dominem o modelo. O StandardScaler foi calculado de forma **incremental** sobre o dataset completo para garantir que os parâmetros μ e σ refletem a distribuição real dos 40 milhões de registos, assegurando a ausência de *data leakage* no *pipeline*.

#### Remoção de variáveis redundantes

Foram removidas as colunas `id`, `device_id` e `device_ip` por serem identificadores únicos sem poder preditivo — identificadores únicos não generalizam para dados não vistos. A coluna `hour` foi igualmente removida após a extração da variável `hora_do_dia`.

---

### 3.2. Criação de Novos Atributos

Foram criadas **3 novas variáveis** que introduzem informação adicional não capturada diretamente pelo dataset original. A correlação de cada nova variável com `click` foi verificada incrementalmente sobre os 40 milhões de registos antes de ser incluída no *pipeline* final.

<!-- INSERIR FIGURA 7 -->
<!-- Barplot da correlação de Pearson de cada variável (incluindo as 3 novas) com 'click', ordenado descendentemente -->
<!-- reports/figures/fig7_correlacao_novas_variaveis.png -->

* **`hora_do_dia`** — Extraída da variável `hour` (formato `YYMMDDHH`) através da operação `% 100`, que retém apenas os 2 últimos dígitos correspondentes à hora (0–23). A análise bivariada da secção 1.2 confirmou a correlação entre a hora do dia e o CTR, tornando esta variável temporalmente informativa para o modelo.

* **`banner_area`** — Produto entre `C15` (largura do banner em píxeis) e `C16` (altura do banner em píxeis): `C15 × C16`. Um banner com maior área ocupa mais espaço visual na página, potencialmente aumentando a probabilidade de ser visto e clicado.

* **`visibilidade_anuncio`** — Rácio entre `banner_pos` e `device_type + 1`: `banner_pos / (device_type + 1)`. O mesmo banner no topo da página tem visibilidade distinta num telemóvel versus num PC, dado que os tamanhos de ecrã e os padrões de *scroll* diferem entre dispositivos. Esta variável captura a interação (*interaction feature*) entre posição e dispositivo, duas das variáveis mais relevantes para o CTR identificadas na EDA (He et al., 2014).

<!-- INSERIR FIGURA 8 -->
<!-- Histogramas das 3 novas variáveis criadas: hora_do_dia, banner_area e visibilidade_anuncio -->
<!-- reports/figures/fig8_histogramas_novas_variaveis.png -->

---

## 4. Dicionário de Dados Final (Pós-Processamento)

Listagem final das variáveis que serão entregues ao modelo na Fase 3.

| Atributo               | Tipo                   | Descrição                                                      |
| :--------------------- | :--------------------- | :------------------------------------------------------------- |
| `click`                | Inteiro (binário)      | Variável alvo: 1 = clicou, 0 = não clicou                      |
| `C1`                   | Float (normalizado)    | Variável anónima normalizada (StandardScaler)                  |
| `banner_pos`           | Inteiro                | Posição do banner na página                                    |
| `site_id`              | Inteiro (label enc.)   | ID do site (codificado por rótulo)                             |
| `site_domain`          | Inteiro (label enc.)   | Domínio do site (codificado por rótulo)                        |
| `site_category`        | Inteiro (label enc.)   | Categoria do site (codificado por rótulo)                      |
| `app_id`               | Inteiro (label enc.)   | ID da aplicação (codificado por rótulo)                        |
| `app_domain`           | Inteiro (label enc.)   | Domínio da aplicação (codificado por rótulo)                   |
| `app_category`         | Inteiro (label enc.)   | Categoria da aplicação (codificado por rótulo)                 |
| `device_type`          | Inteiro                | Tipo de dispositivo (0=móvel, 1=PC, 4=tablet)                  |
| `device_conn_type`     | Inteiro                | Tipo de ligação à internet                                     |
| `device_model`         | Inteiro (label enc.)   | Modelo do dispositivo (codificado por rótulo)                  |
| `C14` – `C21`          | Float (normalizado)    | Variáveis anónimas normalizadas (StandardScaler)               |
| `hora_do_dia`          | Inteiro                | **NOVA** — Hora do dia extraída de `hour` (0–23)               |
| `banner_area`          | Inteiro                | **NOVA** — Área do banner em píxeis (C15 × C16)                |
| `visibilidade_anuncio` | Float                  | **NOVA** — Índice de visibilidade (banner_pos / device_type+1) |
| `id`                   | —                      | **Removido** — Identificador único, sem poder preditivo        |
| `device_id`            | —                      | **Removido** — Identificador único, sem poder preditivo        |
| `device_ip`            | —                      | **Removido** — Identificador único, sem poder preditivo        |
| `hour`                 | —                      | **Removido** — Substituído por `hora_do_dia`                   |

---

## 5. Conclusões da Fase de Exploração

A fase de exploração revelou vários padrões importantes que não eram visíveis no final do Milestone 1:

1. **Desequilíbrio de classes severo:** O dataset apresenta uma distribuição 83%/17% que inviabiliza o uso de *Accuracy* como métrica de avaliação. O Milestone 3 adotará AUC-ROC, Log Loss e F1-score como métricas principais, em linha com os Objetivos SMART do projeto.

2. **Padrão temporal contra-intuitivo:** As horas de madrugada (0h–3h) apresentam CTR acima da média global, contrariando o pressuposto inicial de que o período diurno seria o mais eficaz. Este *insight* é relevante para a otimização temporal de campanhas publicitárias.

3. **Variáveis anónimas com alto poder preditivo:** As variáveis `C14`, `C15` e `C16` são as mais correlacionadas com a variável alvo, sugerindo que representam características intrínsecas do anúncio com forte poder discriminante.

4. **Dataset pronto para modelação:** Todos os valores em falta foram tratados, as variáveis categóricas foram codificadas, o escalonamento foi aplicado e 3 novas variáveis com correlação verificada com o objetivo foram adicionadas ao *pipeline*. O dataset processado foi exportado para `data/processed/` e está pronto para o Milestone 3.

---

## Referências Bibliográficas

Domingos, P. (2012). A few useful things to know about machine learning. *Communications of the ACM, 55*(10), 78–87.

He, X., Pan, J., Jin, O., Xu, T., Liu, B., Xu, T., Shi, Y., Atallah, A., Herbrich, R., Bowers, S., & Candela, J. Q. (2014). Practical lessons from predicting clicks on ads at Facebook. In *Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1–9). ACM.

Japkowicz, N., & Stephen, S. (2002). The class imbalance problem: A systematic study. *Intelligent Data Analysis, 6*(5), 429–449.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.

Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences (3rd ed.). Lawrence Erlbaum.

---

*Data de última atualização: 23/03/2026*
---

*Data de última atualização: 23/03/2026*
