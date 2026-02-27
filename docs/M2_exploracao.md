# Milestone 2: Análise Exploratória e Engenharia de Atributos
> **Nota de Revisão:** Este documento pressupõe que o dataset já foi identificado e descrito no
ficheiro `docs/M1_iniciacao.md`. Caso precise de consultar o significado original das variáveis,
deve consultar essa Milestone.
## 1. Análise Exploratória de Dados (EDA)
### 1.1. Distribuição da Variável Alvo
A nossa variável alvo é o click (binária: 0 para não clique, 1 para clique). Através da análise do dataset completo (40.428.967 linhas), observámos o seguinte:

Equilíbrio de Classes: A variável está altamente desequilibrada.

Factos importantes:

Classe 0 (Não Clicou): Aproximadamente 83.0% das ocorrências.

Classe 1 (Clicou): Aproximadamente 17.0% das ocorrências.

Implicação: Para a aprendizagem supervisionada, o modelo terá uma tendência natural para prever a classe majoritária. Será necessário utilizar métricas de avaliação como F1-Score ou AUC-ROC em vez da acurácia.

### 1.2. Correlações Relevantes

Banner Position vs. Alvo (Análise de Localização)
Foi realizada uma contagem total de todas as posições onde os anúncios foram exibidos. O código identificou que os dados estão concentrados em apenas dois valores principais. Os valores que aparecem raramente foram classificados como outliers de frequência. A análise consistiu em observar se esses casos raros mantêm a mesma proporção de cliques que as posições comuns, validando se a localização é um fator determinante para a variável-alvo.

C14, C15, C16 (Variáveis Críticas): Estas variáveis apresentam uma alta variância e uma grande quantidade de outliers (identificados com "x" nos nossos boxplots).

<img width="600" height="200" alt="image" src="https://github.com/user-attachments/assets/07909e97-d5ff-472c-af24-0e0757f89ed7" />

Relevância para o Negócio: Variáveis como site_category e app_category mostram uma concentração extrema. O "Top 3" das categorias domina mais de 50% do tráfego, indicando que o contexto do site é um preditor fundamental para o sucesso do clique.

## 2. Qualidade dos Dados e Limpeza
### 2.1. Tratamento de Dados em Falta (Missing Data)
* **Colunas afetadas:** [Lista de colunas]
* **Estratégia adotada:** (Ex: "Substituímos os nulos da coluna 'Salário' pela mediana para
evitar o impacto de outliers.")
### 2.2. Outliers e Inconsistências
*Descrevam se encontraram valores impossíveis (ex: idade = 200) e como os resolveram.*
## 3. Engenharia de Atributos (Feature Engineering)
### 3.1. Transformações Realizadas
* **Encoding:** (Ex: "Convertemos a variável 'Género' em numérica usando One-Hot Encoding.")
* **Escalonamento:** (Ex: "Aplicámos o StandardScaler nas variáveis numéricas para que todas
fiquem na mesma escala.")
### 3.2. Criação de Novos Atributos
*Descrevam as variáveis que criaram para ajudar o modelo.*
* **Nova Variável [Nome]:** (Ex: "Criámos a 'Tenure_Per_Year' que divide o tempo de contrato
pela idade do cliente.")
## 4. Dicionário de Dados Final (Pós-Processamento)
*Listagem final das variáveis que serão entregues ao modelo na Fase 3.*
| Atributo | Tipo | Descrição |
| :--- | :--- | :--- |
| `cliente_id` | ID | Removido (não preditivo) |
| `idade_norm` | Float | Idade após normalização |
| `is_premium` | Binary | 1 para clientes com plano superior |
## 5. Conclusões da Fase de Exploração
*O que aprenderam sobre o dataset que não sabiam no final do Milestone 1? Os dados são suficientes
para avançar para a modelação?*
---
*Data de última atualização: [DD/MM/AAAA]* 
