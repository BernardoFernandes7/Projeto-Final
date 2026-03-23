# Milestone 2: Análise Exploratória e Engenharia de Atributos

> **Nota de Revisão:** Este documento pressupõe que o dataset já foi identificado e descrito no ficheiro `docs/M1_iniciacao.md`. Caso precise de consultar o significado original das variáveis, deve consultar essa Milestone.

---

## 1. Análise Exploratória de Dados (EDA)

### 1.1. Distribuição da Variável Alvo

A variável alvo `click` é binária (0 = não clicou, 1 = clicou) e apresenta um forte desequilíbrio de classes:

> **Factos importantes:** A variável alvo `click` está fortemente desequilibrada, com aproximadamente **83% de não-cliques (0)** e **17% de cliques (1)**. Este desequilíbrio é esperado em problemas de CTR e justifica o uso de métricas como AUC-ROC, Log Loss e F1-score em detrimento da simples Accuracy, que seria enganosa neste contexto — um modelo que preveja sempre "0" teria 83% de accuracy sem qualquer utilidade preditiva real.

### 1.2. Correlações Relevantes

A matriz de correlação de Pearson foi calculada de forma incremental sobre os 40 milhões de registos para identificar as variáveis com maior relação com a variável alvo.

* **C14 vs. click:** A variável anónima `C14` apresenta a correlação mais elevada com `click`, sugerindo que representa uma característica do anúncio ou do utilizador com forte impacto na decisão de clique. O scatter plot de CTR por C14 confirma que determinados valores de C14 têm CTR significativamente acima da média global.
* **hora_do_dia vs. click:** A análise bivariada do CTR por hora do dia revela variações significativas ao longo das 24 horas, com as primeiras horas da madrugada (0h–3h) a apresentarem CTR mais elevado, contrariando a intuição inicial de que o período diurno seria mais eficaz.
* **banner_pos vs. click:** A posição do banner na página influencia o CTR, mas a relação não é linear — o topo da página nem sempre corresponde ao maior CTR, dependendo do tipo de dispositivo utilizado.
* **device_type vs. click:** Dispositivos móveis apresentam padrões de clique distintos dos PCs, com diferenças relevantes no CTR médio por tipo de dispositivo.

---

## 2. Qualidade dos Dados e Limpeza

### 2.1. Tratamento de Dados em Falta

* **Valores em falta padrão:** Confirmada a ausência total de valores em falta padrão em todas as 24 colunas do dataset — nenhuma célula estava vazia ou nula.
* **Valores em falta mascarados:** O dataset Avazu utiliza o valor `-1` para representar informação desconhecida nas colunas anónimas. Foi detetada uma percentagem relevante de `-1` na coluna `C20`.
* **Colunas afetadas:** `C20`
* **Estratégia adotada:** Substituição dos valores `-1` pela **moda global** da coluna (calculada sobre os 40 milhões de registos, excluindo os próprios `-1`). Optou-se pela moda em vez da média ou mediana porque `C20` é uma variável categórica codificada numericamente — a média de categorias não tem significado semântico, enquanto a moda representa o valor mais frequente e portanto mais representativo da população.

### 2.2. Outliers e Inconsistências

* **Outliers numéricos:** Foram identificados outliers nas colunas `C1`, `banner_pos` e `C14` através do método IQR (Intervalo Interquartil). Dado que o dataset Avazu é um dataset de comportamento real de utilizadores, os valores extremos não representam erros de medição mas sim comportamentos legítimos e raros — optou-se por **não remover** estes outliers para não perder informação potencialmente relevante para o modelo.
* **Erros de tipo de dados:** As colunas categóricas (`site_id`, `site_domain`, `site_category`, `app_id`, `app_domain`, `app_category`, `device_model`) foram confirmadas e convertidas explicitamente para tipo `string` antes do encoding, garantindo consistência no processamento.

---

## 3. Engenharia de Atributos (Feature Engineering)

### 3.1. Transformações Realizadas

* **Codificação por Rótulo (Label Encoding):** Aplicada nas 7 colunas categóricas de alta cardinalidade (`site_id`, `site_domain`, `site_category`, `app_id`, `app_domain`, `app_category`, `device_model`). Optou-se pelo Label Encoding em vez do One-Hot Encoding porque estas colunas têm milhares de categorias únicas — o One-Hot Encoding geraria milhares de novas colunas, tornando o dataset inviável em termos de memória e tempo de processamento para 40 milhões de registos.
* **Normalização por Padronização (StandardScaler):** Aplicada nas colunas numéricas `C1`, `C14`, `C15`, `C16`, `C17`, `C18`, `C19`, `C20` e `C21`. Esta técnica coloca todas as variáveis na mesma escala (média 0, desvio padrão 1), evitando que variáveis com valores absolutos maiores dominem o modelo. O StandardScaler foi calculado de forma incremental sobre o dataset completo para garantir que os parâmetros de escalonamento refletem a distribuição real dos 40 milhões de registos.
* **Remoção de variáveis redundantes:** Foram removidas as colunas `id`, `device_id` e `device_ip` por serem identificadores únicos sem poder preditivo, e a coluna `hour` após a extração da `hora_do_dia`.

### 3.2. Criação de Novos Atributos

Foram criadas 3 novas variáveis que introduzem informação nova não presente diretamente no dataset original:

* **`hora_do_dia`** — Extraída da variável `hour` (formato `YYMMDDHH`) através da operação `% 100`, que retém apenas os 2 últimos dígitos correspondentes à hora (0–23). Justificação: a análise bivariada confirmou correlação entre a hora do dia e o CTR, e esta transformação torna a variável temporal utilizável pelo modelo.
* **`banner_area`** — Produto entre `C15` (largura do banner) e `C16` (altura do banner), representando a área total do anúncio em píxeis (`C15 × C16`). Justificação: um banner maior ocupa mais espaço visual e pode influenciar a probabilidade de clique — esta variável captura uma dimensão física do anúncio que não existia diretamente no dataset.
* **`visibilidade_anuncio`** — Rácio entre `banner_pos` e `device_type + 1`, representando um índice de visibilidade que combina a posição do anúncio com o tipo de ecrã onde é visualizado. Justificação: o mesmo banner no topo da página tem visibilidade diferente num telemóvel versus num PC — esta variável captura a interação entre posição e dispositivo, duas das variáveis mais relevantes para o CTR identificadas na EDA.

A correlação de cada nova variável com `click` foi verificada incrementalmente sobre os 40 milhões de registos antes de ser incluída no pipeline final.

---

## 4. Dicionário de Dados Final (Pós-Processamento)

| Atributo               | Tipo                   | Descrição                                           |
| :--------------------- | :--------------------- | :-------------------------------------------------- |
| `click`                | Inteiro (binário)      | Variável alvo: 1 = clicou, 0 = não clicou           |
| `C1`                   | Número decimal (float) | Variável anónima normalizada                        |
| `banner_pos`           | Número inteiro         | Posição do banner na página                         |
| `site_id`              | Número inteiro         | ID do site (codificado por rótulo)                  |
| `site_domain`          | Número inteiro         | Domínio do site (codificado por rótulo)             |
| `site_category`        | Número inteiro         | Categoria do site (codificado por rótulo)           |
| `app_id`               | Número inteiro         | ID da aplicação (codificado por rótulo)             |
| `app_domain`           | Número inteiro         | Domínio da aplicação (codificado por rótulo)        |
| `app_category`         | Número inteiro         | Categoria da aplicação (codificado por rótulo)      |
| `device_type`          | Número inteiro         | Tipo de dispositivo (0=móvel, 1=PC, 4=tablet)       |
| `device_conn_type`     | Número inteiro         | Tipo de ligação à internet                          |
| `device_model`         | Número inteiro         | Modelo do dispositivo (codificado por rótulo)       |
| `C14`–`C21`            | Número decimal (float) | Variáveis anónimas normalizadas                     |
| `hora_do_dia`          | Número inteiro         | Hora do dia extraída de `hour` (0–23)               |
| `banner_area`          | Número inteiro         | Área do banner em píxeis (C15 × C16)                |
| `visibilidade_anuncio` | Número decimal (float) | Índice de visibilidade (banner_pos / device_type+1) |
| `id`                   | —                      | Removido (identificador único, não preditivo)       |
| `device_id`            | —                      | Removido (identificador único, não preditivo)       |
| `device_ip`            | —                      | Removido (identificador único, não preditivo)       |
| `hour`                 | —                      | Removido após extração de `hora_do_dia`             |


---

## 5. Conclusões da Fase de Exploração

A fase de exploração revelou vários padrões importantes que não eram visíveis no Milestone 1:

* O dataset está fortemente desequilibrado (83%/17%), o que condiciona diretamente a escolha das métricas de avaliação na fase de modelação.
* A hora do dia tem um impacto inesperado no CTR, com as primeiras horas da madrugada a superarem o período diurno — um insight relevante para a segmentação de campanhas publicitárias.
* As variáveis anónimas `C14`, `C15` e `C16` são as mais correlacionadas com a variável alvo, sugerindo que representam características do anúncio com forte poder preditivo.
* O dataset está pronto para a fase de modelação: valores em falta tratados, variáveis codificadas, escalonamento aplicado e 3 novas variáveis criadas com correlação verificada com o objetivo.

---

*Data de última atualização: 23/03/2026*
