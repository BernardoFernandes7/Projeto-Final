# Milestone 2: Análise Exploratória e Engenharia de Atributos

> **Nota de Revisão:** Este documento pressupõe que o dataset já foi identificado e descrito no ficheiro `docs/M1_iniciacao.md`. Caso precise de consultar o significado original das variáveis, deve consultar essa Milestone.

---

## 1. Análise Exploratória de Dados (EDA)

A Análise Exploratória de Dados (EDA) constitui uma fase fundamental no ciclo de vida de qualquer projeto de Ciência de Dados. Segundo Tukey (1977), a EDA representa um conjunto de técnicas estatísticas e visuais que permite compreender a estrutura interna de um conjunto de dados antes de qualquer modelação preditiva. No contexto do presente projeto, a EDA foi conduzida de forma incremental sobre os 40 milhões de registos do dataset Avazu, utilizando processamento em *chunks* de modo a superar as limitações de memória RAM do ambiente Kaggle.

---

### 1.1. Distribuição da Variável Alvo

A variável alvo `click` é binária (0 = não clicou, 1 = clicou) e apresenta um forte desequilíbrio de classes.

> **Factos importantes:** A variável alvo `click` está fortemente desequilibrada, com aproximadamente **83% de não-cliques (0)** e **17% de cliques (1)**. A distribuição não é normal. O desequilíbrio exige que a performance do modelo não seja medida por Accuracy, mas sim por Log-Loss (métrica oficial do desafio Avazu) e AUC-ROC, para garantir que o modelo penaliza corretamente as falsas previsões de cliques — um modelo que previsse sempre "0" teria 83% de *accuracy* sem qualquer utilidade preditiva real, fenómeno denominado *accuracy paradox* (Japkowicz & Stephen, 2002).


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

* **Posição do Banner (`banner_pos`)** : A análise bivariada demonstra que as posições 0 e 1 detêm a vasta maioria das impressões, mas o CTR (Click-Through Rate) varia significativamente entre elas, validando a importância desta variável para o modelo.

* **`device_type` vs. `click`:** Dispositivos móveis apresentam padrões de clique distintos dos PCs, com diferenças relevantes no CTR médio por tipo de dispositivo.
* **Variáveis de Dispositivo**: `device_type` e `device_conn_type` mostram correlação direta com o comportamento de clique. Notou-se que certos tipos de ligação (ex: tipo 0) concentram a maioria dos dados.
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

#### Colunas afetadas: 

`C14` a `C21` e `device_id`/`device_ip`.

#### Deteção e Diagnóstico de "Hidden NaNs":

Uma inspeção inicial revelou que o dataset não contém valores nulos do tipo NaN padrão. No entanto, através de uma análise de frequências, identificámos a presença de Missing Data mascarado sob os valores -1 e 0, que funcionam como placeholders operacionais. Esta prática é comum em sistemas de logging de publicidade programática para representar campos não preenchidos ou erros de sensor.

#### Estratégia de Imputação para a Coluna `C20`:

A variável `C20` revelou-se a mais crítica, com aproximadamente 46,8% de valores nulos mascarados como -1. Para tratar esta lacuna sem comprometer o volume de dados (visto que a remoção de quase metade do dataset introduziria um viés de seleção inaceitável), implementámos a seguinte estratégia:

-   Optámos por calcular a moda (valor mais frequente) de forma incremental através de chunks, garantindo que o valor imputado fosse estatisticamente representativo do conjunto total de 40 milhões de linhas.
-   Rejeitámos categoricamente a utilização da média ou mediana para a coluna `C20`, dado que a variável é uma variável categórica nominal anonimizada (representando um ID de sistema), o cálculo da média não possui significado físico ou lógico. Operações aritméticas sobre identificadores categóricos resultariam em valores inexistentes no domínio original, enquanto a moda preserva a natureza discreta e categórica do dado, mantendo a consistência da distribuição.

---

### 2.2. Outliers e Inconsistências

#### Outliers numéricos
 
Na deteção de *outliers* utilizámos o método do **Intervalo Interquartil (IQR)**  para quantificar a presença de outliers nas variáveis numéricas de dimensões e categorias anonimizadas. O cálculo baseou-se na amplitude entre o primeiro quartil ($Q1$) e o terceiro quartil ($Q3$), definindo como outliers todos os valores fora do intervalo $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$.

- Resultados Criticos: Identificámos que variáveis como C15 (Largura) apresentam 10.65% de outliers e C16 (Altura) apresentam 13.04%.

<!-- INSERIR FIGURA 6 -->
<!-- Boxplots das variáveis C1, C14, banner_pos e C15/C16 para visualizar os outliers identificados pelo IQR -->
<!-- reports/figures/fig6_boxplots_outliers.png -->

Dado que o dataset Avazu é um dataset de comportamento real de utilizadores, os valores extremos não representam erros de medição, mas sim **comportamentos legítimos e raros**. Optou-se por **não remover** estes *outliers* para não perder informação potencialmente relevante para o modelo.

#### Erros de tipo de dados

As colunas categóricas (`site_id`, `site_domain`, `site_category`, `app_id`, `app_domain`, `app_category`, `device_model`) foram confirmadas e convertidas explicitamente para tipo `string` antes do *encoding*, garantindo consistência no processamento e prevenindo erros silenciosos quando inteiros são interpretados como categorias numéricas por bibliotecas como o scikit-learn (Pedregosa et al., 2011).

#### Decisão Estratégica( Normalização via `StandardScaler`) : Em vez da técnica de Capping ou Trimming (que alteraria a distribuição original), optámos pela aplicação do StandardScaler.

- **Mitigação do Impacto no Gradiente**: Algoritmos baseados em descida de gradiente ou redes neuronais são sensíveis à magnitude dos dados. Ao transformar as variáveis para terem média $\mu=0$ e desvio padrão $\sigma=1$, garantimos que os outliers (valores com elevado $Z-score$) não dominem a função de custo durante o treino.
-   **Convergência do Algoritmo**: Esta abordagem "suaviza" a influência dos valores extremos, permitindo que o otimizador encontre o mínimo global de forma mais estável, mantendo a informação de que aquele anúncio possui uma dimensão fora do comum, sem que isso cause instabilidade numérica no modelo.
---

## 3. Engenharia de Atributos (Feature Engineering)

A Engenharia de Atributos é o processo de transformar dados brutos em representações mais informativas para os algoritmos de aprendizagem automática. Segundo Domingos (2012), *"more data beats a clever algorithm, but good features beat more data"* — destacando a centralidade desta fase no desempenho preditivo dos modelos. Foram realizadas três categorias de transformações: codificação, normalização e criação de novas variáveis.

---

### 3.1. Transformações Realizadas

Nesta fase, o objetivo foi converter o espaço de estados original (misto entre categorias textuais e valores discretos de grande amplitude) num formato tensorial otimizado para a convergência de modelos baseados em gradiente.

#### Encoding — Codificação por Rótulo (*Label Encoding*)

Para as variáveis de alta cardinalidade, como `site_id`, `app_id` e `device_model`, implementámos o **Label Encoding**.

**Fundamentação Técnica**:Ao contrário do One-Hot Encoding, que criaria milhares de colunas esparsas (aumentando drasticamente o consumo de memória e o risco de overfitting devido à maldição da dimensionalidade), o Label Encoding mapeia cada categoria única para um inteiro distinto.

**Gestão de Memória**: Esta abordagem permitiu manter o dataset compacto, essencial para o processamento em chunks no ambiente Kaggle, preservando a identidade de cada domínio ou aplicação sem explodir a matriz de características.

**Tratamento de Unseen Labels**: O pipeline foi desenhado para mapear valores desconhecidos ou residuais para um índice comum, garantindo a robustez do modelo durante a fase de teste.

#### Escalonamento — Padronização (*StandardScaler*)

Aplicado nas colunas numéricas `C1`, `C14`, `C15`, `C16`, `C17`, `C18`, `C19`, `C20` e `C21`. Esta transformação centra cada variável em torno da média zero e escala pelo desvio padrão: **z = (x − μ) / σ**.

**Simetria e Centragem**: Esta técnica centraliza os dados na média ($\mu=0$) e escala-os para um desvio padrão unitário ($\sigma=1$).

**Prevenção de Bias de Magnitude**: Sem o escalonamento, variáveis com amplitudes maiores (como IDs de categorias representados por números elevados) exerceriam uma influência desproporcional na atualização dos pesos do modelo, "abafando" variáveis binárias ou de menor escala que podem ser altamente preditivas.

**Estabilidade Numérica**: Ao colocar todas as features na mesma ordem de magnitude, facilitamos o trabalho do otimizador (ex: Adam ou SGD), resultando numa descida de gradiente mais suave e rápida, reduzindo o risco de oscilações em torno do mínimo global.

#### Remoção de variáveis redundantes

Foram removidas as colunas `id`, `device_id` e `device_ip` por serem identificadores únicos sem poder preditivo — identificadores únicos não generalizam para dados não vistos. A coluna `hour` foi igualmente removida após a extração da variável `hora_do_dia`.

- **Racional**: Estas variáveis possuem uma cardinalidade próxima do número total de observações. Manter identificadores únicos de utilizadores levaria o modelo a "memorizar" instâncias específicas (overfitting) em vez de aprender padrões generalizáveis de comportamento, o que comprometeria seriamente a performance em dados não vistos.

---

### 3.2. Criação de Novos Atributos

Foram criadas **3 novas variáveis** que introduzem informação adicional não capturada diretamente pelo dataset original. A correlação de cada nova variável com `click` foi verificada incrementalmente sobre os 40 milhões de registos antes de ser incluída no *pipeline* final.

<!-- INSERIR FIGURA 7 -->
<!-- Barplot da correlação de Pearson de cada variável (incluindo as 3 novas) com 'click', ordenado descendentemente -->
<!-- reports/figures/fig7_correlacao_novas_variaveis.png -->

* **`hora_do_dia`(Sazonalidade Temporal)** — A extração a variável `hour` horária a partir do timestamp original (`hour % 100`) permitiu a modelação do ciclo circadiano dos utilizadores. Esta variável é matematicamente essencial para isolar padrões de comportamento que variam drasticamente entre períodos de lazer, horário laboral e períodos noturnos. Dado que o custo de licitação e a propensão ao clique (CTR) flutuam consoante o horário nobre, este atributo oferece ao modelo um referencial temporal cíclico indispensável para a precisão das previsões.
  
* **`banner_area`(Métrica de Impacto Visual** Através do produto escalar das dimensões `C15` (largura) e `C16` (altura), condensámos duas variáveis espaciais numa única métrica de magnitude visual. Do ponto de vista da eficácia publicitária, a área total ocupada por um anúncio é um preditor de visibilidade mais robusto do que as suas medidas isoladas. Esta transformação permite que o modelo identifique, de forma linear e direta, a correlação entre a mancha gráfica do banner e a probabilidade de interação, capturando a premissa de que anúncios com maior impacto visual tendem a gerar comportamentos de clique distintos.
 
* **`visibilidade_anuncio`(Interação Contexto-Dispositivo)** — Este atributo foi concebido como uma variável de interação rácio-dependente, calculada através da relação entre a posição do anúncio (`banner_pos`) e o tipo de dispositivo (`device_type`). A lógica subjacente assume que o valor de uma posição no topo da página não é absoluto; a sua relevância é relativa ao hardware de visualização. Por exemplo, a visibilidade de um anúncio num ecrã reduzido de um smartphone (Mobile) difere substancialmente da mesma posição num monitor de alta resolução (Desktop). Ao normalizar a posição pelo tipo de dispositivo (`banner_pos` / (`device_type` + 1)), introduzimos um índice de visibilidade que ajuda o modelo a discernir em que contextos uma determinada posição de banner é efetivamente mais estratégica para converter impressões em cliques.

 Com estas transformações, o dataset deixa de ser uma mera coleção de registos isolados e passa a incorporar uma camada de inteligência contextual, fundamental para a robustez preditiva na fase de modelação.

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

A fase de Análise Exploratória de Dados (EDA) e Engenharia de Atributos permitiu uma compreensão profunda da dinâmica do dataset Avazu, indo muito além da simples descrição estatística inicial. As principais conclusões que fundamentam a transição para a fase de modelação são::

1. **Validação da Qualidade e Integridade:** A estratégia de processamento em chunks e o uso de bibliotecas como Dask permitiram superar as limitações de memória, garantindo que a análise refletisse a realidade total dos 40 milhões de registos. A identificação e correção de "nulos mascarados" na coluna `C20` (46.8% de valores -1) foi um passo crítico para eliminar o ruído que, de outra forma, enviesaria a aprendizagem do modelo.

2. **Poder Preditivo das Variáveis Sintéticas:** A análise bivariada e de correlação confirmou que as novas variáveis criadas — `hora_do_dia`, `banner_area` e `visibilidade_anuncio` — capturam sinais latentes de comportamento de clique que as variáveis originais, de forma isolada, não conseguiam expressar. Estas transformações converteram dados estáticos em indicadores dinâmicos de contexto e impacto visual.

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
