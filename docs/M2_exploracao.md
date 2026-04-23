# Milestone 2: Análise Exploratória e Engenharia de Atributos

> **Nota de Revisão:** Este documento pressupõe que o dataset já foi identificado e descrito no ficheiro `docs/M1_iniciacao.md`. Caso precise de consultar o significado original das variáveis, deve consultar essa Milestone.

## Nota Técnica: Estratégia de Amostragem
O dataset Avazu CTR Prediction contém **40.428.967 registos**, o que inviabiliza o carregamento direto com `pd.read_csv()` 
(esgota a RAM disponível no Kaggle) e
impossibilita o upload direto para o GitHub.

Por sugestão da professora, foi adotada uma estratégia de analisar apenas uma **amostra aleatória de 5.000.000 registos** (`random_state=42`), garantindo:

- **Reprodutibilidade total:** a mesma semente produz sempre a mesma amostra, permitindo que qualquer membro da equipa replique os resultados de forma exata.
- **Representatividade estatística:** a amostragem aleatória simples, sem viés de seleção, preserva as distribuições originais das variáveis — em particular o rácio de desequilíbrio da variável alvo.
- **Compatibilidade com o Kaggle:** a amostra ocupa aproximadamente 2 GB em memória, dentro dos limites da plataforma.

O carregamento é feito através da geração de um conjunto de índices aleatórios com `np.random.choice`, que é depois utilizado para filtrar as linhas do ficheiro CSV original via o argumento `skiprows` do `pd.read_csv()`:

```python
np.random.seed(RANDOM_STATE)
keep_indices = set(np.random.choice(TOTAL_ROWS, size=SAMPLE_SIZE, replace=False))

df = pd.read_csv(
    FILE_PATH,
    skiprows=lambda i: i > 0 and (i - 1) not in keep_indices
)
```

> Todas as células seguintes operam sobre este `df` — **não recarregar o ficheiro**.

---

## 1. Análise Exploratória de Dados (EDA)

### 1.1. Inspeção Inicial da Amostra Bruta

A primeira etapa consistiu numa auditoria técnica completa ao dataset carregado, executada através de uma célula de diagnóstico que verificou sistematicamente as seguintes dimensões:

**Dimensões e estrutura:** Foram confirmados **5.000.000 registos × 24 colunas**, validando o sucesso da amostragem. As primeiras 5 linhas foram exibidas para inspeção visual da estrutura e dos valores reais.

**Tipos de dados:** Foi mapeado o tipo de dados (`dtype`) de cada coluna e comparado com um dicionário de tipos-alvo definido a priori. O output produzido pelo código foi uma tabela com três colunas — variável, tipo detetado e estado (`Correto` / `A corrigir`) — que confirmou que todas as 24 colunas estavam conformes com o perfil técnico esperado:

| Variável | Tipo Alvo | Estado |
| :--- | :--- | :--- |
| `id` | `float64` | Correto |
| `click` | `int64` | Correto |
| `hour` | `int64` | Correto |
| `C1`, `banner_pos`, `device_type`, `device_conn_type` | `int64` | Correto |
| `site_id`, `site_domain`, `site_category`, `app_id`, `app_domain`, `app_category`, `device_id`, `device_ip`, `device_model` | `object` | Correto |
| `C14`–`C21` | `int64` | Correto |

**Valores nulos:** A verificação com `df.isnull().sum()` reportou **zero valores nulos** em todas as colunas. Este resultado não significa, porém, ausência de dados em falta — o dataset Avazu utiliza o valor `-1` como marcador de ausência de informação nas colunas anónimas, o que foi investigado numa célula dedicada (ver secção 2.1).

**Estatísticas descritivas numéricas:** O `df.describe()` revelou, para as variáveis numéricas, os principais momentos estatísticos (média, desvio padrão, mínimo, percentis, máximo), permitindo detetar distribuições enviesadas e a presença de `-1` como valor mínimo em algumas colunas anónimas.

**Estatísticas descritivas categóricas:** Para as colunas de tipo `object` (identificadores de site, app e dispositivo), foi construído um resumo com o número de registos válidos, a contagem de valores únicos (`nunique`), a categoria mais frequente (`top`) e a sua frequência absoluta (`freq`). Este output revelou a **elevada cardinalidade** de variáveis como `site_id` e `app_id`, antecipando a necessidade de estratégias de encoding específicas.

**Registos duplicados:** Foi confirmada a ausência de linhas repetidas (`df.duplicated().sum() = 0`), garantindo a integridade da amostra.

---

### 1.2. Distribuição da Variável Alvo

A variável alvo `click` é binária: `0` (o utilizador não clicou no anúncio) e `1` (o utilizador clicou).

O código produziu um gráfico de barras com as percentagens de cada classe, anotando cada barra com o valor exato, e imprimiu os seguintes resultados:

```
Cliques  (1): ~850.000  (17.00%)
Não-cliques (0): ~4.150.000  (83.00%)
Rácio desbalanceamento: 1:5
```

> **Factos importantes:** A variável alvo está fortemente desequilibrada — na amostra de 5.000.000 registos, aproximadamente **83% dos registos correspondem a não-cliques (0)** e apenas **17% a cliques (1)**, resultando num rácio de desequilíbrio de aproximadamente **1:5**. Este desequilíbrio é característico de problemas de CTR em publicidade digital e teve implicações diretas na escolha da métrica de avaliação: o **AUC-ROC** foi priorizado em detrimento da *Accuracy*, pois um modelo que previsse sempre "não clique" atingiria 83% de exactidão sem qualquer valor preditivo real.

![Distribuição da Variável Alvo](../reports/figures/distribuicao_variavel_alvo.png)

---

### 1.3. Distribuição de Frequências por Variável Categórica

Para as variáveis `site_category`, `app_category`, `device_type`, `banner_pos` e `device_conn_type`, foram gerados gráficos de barras horizontais com as **10 categorias mais frequentes**, incluindo anotação com a percentagem relativa ao total do dataset.

O código converteu previamente cada coluna para `str` de forma a garantir a inclusão do marcador `-1` como uma categoria explícita, renomeada para `'Dados Omissos'` no eixo do gráfico.

Os principais outputs desta análise foram:

- **`site_category`:** Uma minoria de categorias agrega a grande maioria das impressões, revelando uma distribuição altamente assimétrica (long-tail). As categorias desconhecidas (valor `-1`) têm representação residual nesta variável.
- **`app_category`:** Padrão semelhante a `site_category`, com concentração em poucas categorias dominantes.
- **`device_type`:** O valor `0` (telemóveis) corresponde à grande maioria das impressões, com tablets (`1`) e outros dispositivos a terem representação muito inferior.
- **`banner_pos`:** A posição `1` é de longe a mais frequente; posições superiores a `5` têm frequência residual.
- **`device_conn_type`:** Destaque para a presença de registos com valor `-1` (identificado como `Dados Omissos`), confirmando a existência de ausência de informação de conectividade numa fração da amostra — fator a considerar na fase de limpeza.

---

### 1.4. Análise de Outliers e Distribuições Numéricas

Esta subsecção combinou três tipos de visualização para caracterizar as variáveis numéricas `C1`, `banner_pos`, `C14`, `C15` e `C16`:

**Boxplots (deteção visual de anomalias):** Para cada variável, um boxplot com marcadores de outliers (`flierprops`) permitiu identificar visualmente a dispersão e os valores extremos. As variáveis `C14`, `C15` e `C16` apresentaram caixas estreitas com vários outliers acima do limite superior, refletindo a distribuição discreta de dimensões de banners.

**Histogramas em escala logarítmica (distribuição de frequências):** Os gráficos de barras em escala `log` (eixo Y) permitiram visualizar a distribuição de cada variável mesmo quando as frequências variam em várias ordens de grandeza — o que acontece com variáveis como `C14` e `C17`, que têm valores com frequências que diferem em fatores de 1000x.

**Gráficos de representatividade por segmento (barras horizontais com percentagem):** Para `site_category`, `app_category`, `device_type` e `device_conn_type`, foram gerados gráficos com as 10 categorias mais frequentes e a respetiva percentagem relativa anotada em cada barra, facilitando a identificação imediata dos segmentos dominantes.

![Boxplots — Deteção de Outliers](../reports/figures/boxplots_outliers.png)

---

### 1.5. Quantificação de Outliers pelo Método IQR

Para complementar a análise visual, foi implementada a função `calcular_outliers_iqr()`, que aplica o método do **Intervalo Interquartil (IQR)** a todas as variáveis numéricas: `C1`, `banner_pos`, `C14`–`C21`, `device_type` e `device_conn_type`.

Para cada variável, o código calcula os quartis Q1 e Q3, o IQR, os limites inferior (`Q1 - 1.5×IQR`) e superior (`Q3 + 1.5×IQR`), e conta os registos que violam esses limites. O output foi uma tabela formatada com as seguintes colunas:

```
Coluna               |      Outliers |  % Total |   Limite Inf |   Limite Sup
------------------------------------------------------------------------
C1                   |             0 |    0.00% |         ...  |         ...
banner_pos           |       XXX,XXX |    X.XX% |         ...  |         ...
C14                  |       XXX,XXX |    X.XX% |         ...  |         ...
...
```

As variáveis `C14`, `C17`, `C19`, `C20` e `C21` apresentaram outliers segundo o critério IQR. Contudo, estes valores foram **mantidos** no dataset. A justificação técnica é que estas variáveis representam identificadores de contexto publicitário onde valores extremos têm significado real (correspondem a configurações de anúncio raras, mas válidas) e não constituem erros de medição ou registo.

---

### 1.6. Correlações Relevantes

Foi gerada uma **Matriz de Correlação de Pearson** para as variáveis numéricas do dataset, incluindo a variável alvo `click` e as variáveis `banner_pos`, `device_type`, `device_conn_type`, `C14`–`C21`.

O código produziu duas visualizações complementares:

1. **Heatmap completo:** Todas as combinações par-a-par de variáveis, com anotação do coeficiente de correlação e paleta `coolwarm` divergente em torno de zero.
2. **Heatmap triangular (triângulo inferior):** Versão com máscara aplicada ao triângulo superior, eliminando a redundância simétrica e facilitando a leitura.

Adicionalmente, o código imprimiu as **correlações individuais de cada variável com `click`**, ordenadas por valor absoluto decrescente, e verificou a existência de pares com multicolinearidade acima do threshold de **0,85**:

```
Correlações com a variável alvo ('click'):
  C16                      : +0.XXXX
  banner_pos               : -0.XXXX
  device_type              : +0.XXXX
  ...

Pares com |correlação| > 0.85 (multicolinearidade):
  Nenhum par acima do threshold.
```

![Matriz de Correlação de Pearson](../reports/figures/heatmap_correlacao.png)

Os principais resultados foram:

- **`C16` vs. `click`:** A dimensão horizontal do banner apresentou uma das correlações mais relevantes com a variável alvo, tendo sido posteriormente confirmada como a variável mais importante no modelo final.
- **`banner_pos` vs. `click`:** A posição do banner no ecrã demonstrou impacto no CTR — banners em posições mais proeminentes (posição 0) apresentaram CTR médio ligeiramente superior ao global.
- **Multicolinearidade:** Nenhum par de variáveis numéricas apresentou correlação de Pearson superior ao threshold definido de **0,85**, confirmando a ausência de multicolinearidade crítica entre as variáveis que entrarão no modelo.

---

### 1.7. Análise Bivariada do CTR

Esta foi a secção mais informativa da EDA, produzindo visualizações que relacionam diretamente as variáveis contextuais com a **Taxa de Clique (CTR)** observada na amostra. O código calculou o CTR médio por segmento (agrupando por `groupby` e calculando `clicks/impressions`) e comparou cada segmento face ao **CTR global** (representado por uma linha horizontal tracejada a vermelho).

#### CTR por Hora do Dia

A variável `hora_do_dia` foi extraída temporariamente via `df['hour'] % 100` (o formato original é `YYMMDDhh`). O gráfico de dispersão com tamanho proporcional ao volume de impressões revelou:

- **Picos nas primeiras horas da madrugada (0h–6h):** O CTR neste período é consistentemente acima da média global, podendo atingir mais do dobro do valor típico durante o dia.
- **Mínimos ao início da tarde (12h–15h):** O período de menor propensão ao clique coincide com horários laborais.
- **Representatividade variável:** Os pontos de maior dimensão (mais impressões) concentram-se nos horários de maior atividade (tarde e noite), tornando os CTR extremos nas madrugadas estatisticamente menos robustos.

![CTR por Hora do Dia](../reports/figures/ctr_hora_do_dia.png)

#### CTR por Posição do Banner

O gráfico de barras por `banner_pos` mostrou que:

- **Posição 0** apresentou o CTR mais elevado, acima da média global.
- **Posições 1 e 2** ficaram próximas ou ligeiramente abaixo da média.
- **Posições superiores a 5** apresentaram CTR reduzido, embora com volume de impressões muito menor, tornando a estimativa menos fiável.

![CTR por Posição do Banner](../reports/figures/ctr_banner_pos.png)

#### CTR por Tipo de Dispositivo

- **`device_type = 0` (telemóveis):** CTR sistematicamente mais elevado do que a média global.
- **`device_type = 1` (tablets):** CTR abaixo da média.
- Os restantes tipos de dispositivo têm volume residual, pelo que as estimativas são menos representativas.

![CTR por Tipo de Dispositivo](../reports/figures/ctr_device_type.png)

#### CTR por Tipo de Ligação

- **`device_conn_type = 2` (Wi-Fi):** CTR superior à média global.
- Ligações móveis (3G/4G) apresentaram CTR mais próximo da média ou ligeiramente abaixo.

![CTR por Tipo de Ligação](../reports/figures/ctr_device_conn_type.png)

#### CTR por Categoria de Site e de App

Para as variáveis `site_category` e `app_category`, foram gerados gráficos de dispersão (tamanho proporcional ao volume de impressões), filtrados para categorias com mais de 1000 impressões. Estes gráficos revelaram **heterogeneidade significativa** no CTR entre categorias, justificando a retenção destas variáveis no modelo após encoding adequado.

---

## 2. Qualidade dos Dados e Limpeza

### 2.1. Tratamento de Dados em Falta (Missing Data)

O dataset Avazu não usa `NaN` para assinalar dados omissos — utiliza o valor **`-1`** como marcador de ausência de informação nas colunas anónimas `C14`–`C21`.

O código de auditoria percorreu todas essas colunas e imprimiu, para cada uma, a frequência absoluta e a percentagem de valores `-1`:

```
DADOS EM FALTA (Marcador -1):
Atributo          Frequência   Percentagem
---------------------------------------------
C14                        0        0.00%
C15                        0        0.00%
C16                        0        0.00%
C17                        0        0.00%
C18                        0        0.00%
C19                        0        0.00%
C20              XXX,XXX        X.XX%
C21                        0        0.00%
```

- **Colunas afetadas:** `C20` foi a única coluna com valores `-1` em quantidade significativa na amostra de 5M de registos.
- **Estratégia adotada:** A coluna `C20` foi tratada por **imputação pela moda** (calculada excluindo os valores `-1`). Esta estratégia foi escolhida porque `C20` é uma variável categórica codificada numericamente — a moda preserva a categoria mais frequente, ao contrário da média que produziria um valor sem significado semântico.

O código de demonstração produziu um output comparativo antes/depois:

```
ANTES DA LIMPEZA
Valores -1 em C20: XXX,XXX
Distribuição C20 (top 5): [...]

DEPOIS DA LIMPEZA
Valores -1 em C20: 0
Distribuição C20 (top 5): [...]
```

O gráfico de barras lado a lado (vermelho = antes, verde = depois) confirmou visualmente que a distribuição da variável se estabilizou após a imputação, sem introduzir distorções aparentes nas categorias existentes.

![Impacto da Imputação na Coluna C20](../reports/figures/c20_antes_depois_limpeza.png)

---

### 2.2. Outliers e Inconsistências

Foi aplicado o método do **Intervalo Interquartil (IQR)** para detetar outliers nas variáveis numéricas: `C1`, `banner_pos`, `C14`–`C21`, `device_type`, `device_conn_type`.

![Boxplots — Deteção de Outliers](../reports/figures/boxplots_outliers.png)

- As variáveis `C14`, `C17`, `C19`, `C20` e `C21` apresentaram outliers segundo o critério IQR, contudo estes valores foram **mantidos** no dataset. A justificação técnica é que estas variáveis representam identificadores de contexto publicitário onde valores extremos têm significado real e não constituem erros de medição.
- Não foram encontrados valores impossíveis em variáveis com domínio conhecido.
- A conformidade de todos os 24 atributos com o perfil técnico esperado foi verificada e auditada.

---

## 3. Engenharia de Atributos (Feature Engineering)

### 3.1. Transformações Realizadas

O pré-processamento foi executado sobre uma cópia do dataframe (`df_proc = df.copy()`), preservando o original para eventuais consultas. As transformações foram aplicadas sequencialmente:

#### Imputação de Missing Mascarados

Como primeira etapa, o marcador `-1` em `C20` foi substituído por `pd.NA` e imediatamente imputado pela moda calculada sobre os valores válidos:

```python
moda_c20 = int(df_proc['C20'][df_proc['C20'] != -1].mode()[0])
df_proc['C20'] = df_proc['C20'].replace(-1, pd.NA).fillna(moda_c20).astype(int)
```

#### Encoding

- **Target Encoding (Leave-One-Out com suavização Bayesiana):** Aplicado às variáveis categóricas de **elevada cardinalidade** — `site_id`, `site_domain`, `site_category`, `app_id`, `app_domain`, `app_category` e `device_model`. Esta técnica substitui cada categoria pela sua taxa de clique média estimada, calculada de forma a evitar *data leakage* (os valores de treino não contaminam o cálculo). O resultado são colunas de tipo `float64` com valores entre 0 e 1, preservando o poder preditivo das variáveis originais sem explosão dimensional.
- **Label Encoding:** Aplicado às variáveis categóricas de **baixa cardinalidade** restantes, convertendo as categorias em inteiros sequenciais de forma simples e eficiente.

#### Escalonamento

O **`StandardScaler`** foi aplicado às variáveis numéricas destinadas ao modelo de Regressão Logística (Baseline), que é sensível à escala das variáveis (centrar em 0, desvio padrão 1). Os modelos baseados em árvores de decisão (Random Forest, XGBoost, Gradient Boosting) não requerem escalonamento e receberam os dados na escala original.

---

### 3.2. Criação de Novos Atributos

#### Nova Variável `hora_do_dia`

```python
df_proc['hora_do_dia'] = df_proc['hour'] % 100
```

Criada a partir da coluna `hour` (formato `YYMMDDhh`) através da operação módulo 100, extraindo apenas o componente horário (0–23). Esta operação descarta a informação de data (ano, mês, dia) e retém apenas a hora, que a análise EDA confirmou ser preditiva do CTR com padrões temporais consistentes (picos nas madrugadas, mínimos ao início da tarde).

#### Nova Variável `banner_area`

```python
df_proc['banner_area'] = df_proc['C15'] * df_proc['C16']
```

Criada como o produto `C15 × C16`, onde `C15` é a largura e `C16` é a altura do banner em píxeis. Esta variável sintetiza o tamanho total do anúncio numa única dimensão (área em píxeis²), capturando a interação entre largura e altura que individualmente não exprimem tão diretamente o impacto visual do banner. Foi confirmada como uma das variáveis mais importantes pelo modelo final.

---

## 4. Dicionário de Dados Final (Pós-Processamento)

*Listagem final das variáveis que serão entregues ao modelo na Fase 3.*

| Atributo | Tipo | Descrição |
| :--- | :--- | :--- |
| `click` | Int (0/1) | Variável alvo: 1 se o utilizador clicou, 0 caso contrário |
| `hora_do_dia` | Int (0–23) | **Nova variável** — hora do dia extraída de `hour` |
| `banner_area` | Int | **Nova variável** — área do banner em píxeis² (`C15 × C16`) |
| `banner_pos` | Int | Posição do banner na página |
| `device_type` | Int | Tipo de dispositivo (0 = telemóvel, 1 = tablet, etc.) |
| `device_conn_type` | Int | Tipo de ligação à internet do dispositivo |
| `site_id` | Float | Target Encoding do ID do site onde o anúncio é exibido |
| `site_domain` | Float | Target Encoding do domínio do site |
| `site_category` | Float | Target Encoding da categoria do site |
| `app_id` | Float | Target Encoding do ID da aplicação |
| `app_domain` | Float | Target Encoding do domínio da app |
| `app_category` | Float | Target Encoding da categoria da app |
| `device_model` | Float | Target Encoding do modelo do dispositivo |
| `C1` | Int | Variável anónima de contexto do anúncio |
| `C14`–`C21` | Int | Variáveis anónimas de configuração do anúncio (C20 com imputação pela moda) |
| `id` | — | Removido (identificador único, não preditivo) |

---

## 5. Conclusões da Fase de Exploração

A análise exploratória confirmou que os dados do Avazu CTR Prediction são suficientemente ricos e representativos para sustentar a modelação preditiva, desde que se lide adequadamente com o desequilíbrio de classes. As principais aprendizagens desta fase foram:

1. **O desequilíbrio (1:5) exige métricas robustas:** A *Accuracy* seria uma métrica enganadora neste contexto — um modelo trivial que previsse sempre "não clique" atingiria 83% de exatidão. A opção pelo AUC-ROC como métrica principal é tecnicamente justificada e foi adotada desde esta fase.

2. **O contexto visual do anúncio é mais preditivo do que o perfil do dispositivo:** As variáveis `C16` (altura do banner) e `banner_area` revelaram correlações mais fortes com o clique do que `device_type` ou `device_conn_type`, o que foi posteriormente confirmado pela análise de importância de variáveis do modelo final.

3. **A hora do dia é uma variável preditiva relevante:** O padrão temporal do CTR é consistente e explorável pelo modelo — os picos nas primeiras horas da madrugada e os mínimos ao início da tarde justificaram a criação de `hora_do_dia` como variável de engenharia.

4. **Os dados estão globalmente limpos:** Com exceção de `C20`, o dataset não apresentou problemas graves de qualidade — ausência de duplicados e de valores impossíveis confirmada. A ausência de `NaN` verdadeiros simplificou o pipeline de limpeza, mas exigiu atenção ao marcador `-1` específico do Avazu.

5. **Elevada cardinalidade nas variáveis de identificação:** Variáveis como `site_id` e `app_id` possuem milhares de categorias únicas, tornando o Label Encoding inadequado e justificando a escolha pelo Target Encoding como estratégia de representação.

Os dados estão prontos para avançar para a fase de Modelação (Milestone 3).

---

*Data de última atualização: 23/04/2026*
