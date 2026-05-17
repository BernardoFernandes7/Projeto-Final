# Relatório de Conclusão e Entrega de Valor (Milestone 4)

## 1. Síntese de Resultados e Impacto

> **Nota:** Esta secção deve traduzir as métricas técnicas (F1-Score, RMSE, Silhouette, Lift) em
resultados compreensíveis para qualquer pessoa.

* **O Problema Resolvido:**  
O problema definido na Milestone 1 consistia em prever se um utilizador iria clicar ou não num anúncio digital,
utilizando dados do dataset Avazu CTR Prediction. Este problema é relevante porque, em publicidade digital e *Real-Time Bidding*,
os anunciantes precisam de decidir rapidamente em que impressões vale a pena investir.

O objetivo principal era desenvolver um modelo de classificação binária capaz de atingir pelo menos **AUC-ROC = 0.75** no conjunto de teste. 
Este objetivo foi alcançado, uma vez que o modelo final, baseado em **XGBoost otimizado**, obteve **AUC-ROC = 0.7509** no conjunto de teste.

Também foi cumprido o objetivo de identificar as variáveis mais importantes para a previsão do clique. A análise de importância dos atributos 
mostrou que variáveis como `banner_area`, `C16`, `device_type`, `C21` e `site_id` tiveram maior peso na decisão do modelo.

* **Interpretação dos Resultados:**  
Em linguagem simples, o modelo consegue distinguir anúncios com maior e menor probabilidade de clique de forma claramente superior a uma decisão aleatória.
O AUC-ROC de **0.7509** significa que o modelo tem boa capacidade para ordenar impressões publicitárias: quando compara uma impressão que gerou clique com outra
que não gerou clique, tende a atribuir maior probabilidade à impressão que realmente foi clicada.

O **F1-Score de 0.4223** mostra que o modelo consegue equilibrar, de forma moderada, a identificação de cliques reais com o controlo de previsões 
positivas erradas. Este valor deve ser interpretado tendo em conta o forte desequilíbrio do dataset, onde apenas cerca de 17% dos registos correspondem a cliques.

A matriz de confusão mostra que o modelo identificou corretamente **113.172 cliques reais**, mas ainda falhou **56.547 cliques** e classificou **253.084 impressões** como 
potenciais cliques quando, na realidade, não geraram interação. Isto significa que o modelo já tem utilidade prática para priorizar impressões, mas ainda não deve ser visto como uma solução perfeita de decisão automática.

Comparando com o modelo inicial, a melhoria foi significativa. A Regressão Logística obteve **AUC-ROC = 0.6412** no teste, enquanto o XGBoost otimizado 
atingiu **0.7509**, uma melhoria acumulada de **+0.1097**. Esta diferença justifica a escolha de um modelo mais complexo.

* **Valor para o Utilizador/Negócio:**  
O principal valor deste projeto está na capacidade de apoiar decisões de investimento em publicidade digital. Em vez de tratar todas as impressões como
igualmente relevantes, o modelo permite identificar quais têm maior probabilidade de gerar clique.

Na prática, esta solução pode ajudar um anunciante a concentrar orçamento em contextos mais promissores, reduzindo desperdício em impressões com menor 
probabilidade de interação. O modelo não decide sozinho se uma campanha terá sucesso, mas fornece uma pontuação útil para ordenar oportunidades e apoiar decisões em sistemas de *Real-Time Bidding*.

Além disso, a análise de importância das variáveis mostra que o formato visual do anúncio tem grande influência na probabilidade de clique. A variável `banner_area`, criada durante o projeto, foi a mais importante no modelo final, o que sugere que a dimensão e configuração visual do anúncio são fatores relevantes para otimizar campanhas digitais.

## 2. Análise Crítica e Limitações

> **Nota:** Identificar de forma honesta as fronteiras do projeto e onde o modelo pode falhar.

* **Limitações dos Dados:**  
Apesar de o dataset original Avazu conter cerca de 40 milhões de registos, o projeto trabalhou com uma amostra de **5.000.000 registos**, por razões de viabilidade computacional. Esta amostra permitiu desenvolver e testar os modelos, mas não representa a totalidade dos dados disponíveis.

Outra limitação importante é o forte desequilíbrio da variável alvo. Apenas cerca de **17%** dos registos correspondem a cliques, enquanto cerca de **83%** correspondem a não-cliques. Este desequilíbrio dificulta a aprendizagem do modelo, sobretudo na identificação correta da classe minoritária.

Também existem variáveis anónimas, como `C14` a `C21`, cujo significado real não é totalmente conhecido. Embora algumas destas variáveis tenham demonstrado importância para o modelo, a sua interpretação de negócio é limitada. Isto reduz a capacidade de explicar com precisão por que razão certas decisões são tomadas pelo algoritmo.

Além disso, os dados representam um período específico de observações. O comportamento dos utilizadores, os formatos de anúncios e as estratégias de campanha podem mudar ao longo do tempo, o que significa que o desempenho do modelo pode diminuir se for aplicado a dados futuros muito diferentes dos dados usados no treino.

* **Limitações do Modelo:**  
O modelo final, XGBoost otimizado, cumpre o objetivo definido, mas não é perfeito. O AUC-ROC de **0.7509** mostra boa capacidade de ordenação, mas não significa que o modelo acerte todas as previsões individuais.

A matriz de confusão evidencia que ainda existem muitos **Falsos Positivos**. Estes casos correspondem a impressões classificadas como potenciais cliques, mas que não geraram interação. Em contexto real, isto pode representar investimento publicitário pouco eficiente.

Também existem **Falsos Negativos**, ou seja, impressões que geraram clique mas que o modelo classificou como não-clique. Estes casos representam oportunidades reais que o modelo não conseguiu identificar.

Outra limitação é que o modelo identifica padrões estatísticos, mas não prova causalidade. Por exemplo, o facto de `banner_area` ser uma variável muito importante não significa, por si só, que aumentar sempre a área do anúncio irá causar mais cliques. O modelo mostra associação preditiva, não uma relação causal garantida.

* **Contextos de Falha:**  
O modelo pode falhar quando aplicado a contextos muito diferentes dos dados de treino, como novos sites, novas aplicações, novos dispositivos, novos formatos de anúncio ou campanhas com públicos-alvo diferentes.

Também pode ter menor fiabilidade em categorias pouco representadas no treino. Se uma determinada aplicação, site ou configuração de anúncio aparece poucas vezes nos dados, o modelo pode não ter exemplos suficientes para aprender um padrão robusto.

Outro contexto de falha está associado ao limiar de decisão utilizado. O modelo foi avaliado com um determinado *threshold*, mas este limiar pode não ser o ideal para todos os objetivos de negócio. Se o anunciante quiser captar o maior número possível de cliques, pode ser necessário reduzir o *threshold*, aceitando mais Falsos Positivos. Se o objetivo for reduzir desperdício de orçamento, pode ser preferível aumentar o *threshold*, aceitando perder alguns cliques reais.

Assim, o modelo não deve ser usado de forma isolada em decisões financeiras automáticas sem monitorização e ajuste ao objetivo concreto da campanha.

## 3. Considerações Éticas e de Viés

* **Privacidade:**  
O dataset utilizado é anonimizado e não contém nomes, contactos ou identificadores pessoais diretos. Além disso, variáveis como `id`, `device_id` e `device_ip` foram removidas durante o pré-processamento por terem elevada cardinalidade e baixo poder de generalização.

Esta remoção também reduz riscos associados ao uso de identificadores individuais. O modelo final trabalha com padrões agregados de contexto, dispositivo, anúncio e ambiente de exibição, sem procurar identificar pessoas concretas.

Ainda assim, deve existir cuidado na utilização deste tipo de solução, porque dados anonimizados podem continuar a representar padrões de comportamento. Por isso, o modelo deve ser usado apenas para otimização agregada de campanhas e não para tentar reconstruir perfis individuais de utilizadores.

* **Transparência:**  
Foram utilizadas técnicas de avaliação e interpretação para tornar o modelo mais compreensível. A matriz de confusão permitiu analisar os tipos de erro cometidos, enquanto a análise de *Feature Importance* permitiu identificar as variáveis com maior impacto na previsão.

Esta abordagem evita que o modelo seja tratado como uma “caixa negra” absoluta. Sabemos que variáveis como `banner_area`, `C16`, `device_type`, `C21` e `site_id` tiveram maior influência na decisão do XGBoost.

No entanto, a transparência é limitada pela existência de variáveis anónimas no dataset. Como algumas variáveis não têm significado de negócio totalmente conhecido, a explicação das decisões do modelo não é completa.

Também existe risco de viés, uma vez que o modelo aprende a partir de dados históricos. Se determinados dispositivos, sites, aplicações ou formatos de anúncio estiverem sobre-representados nos dados, o modelo pode favorecer esses contextos no futuro. Por isso, numa aplicação real, seria necessário monitorizar o desempenho por segmentos e garantir que a otimização para cliques não prejudica a experiência do utilizador nem conduz a decisões injustas ou excessivamente intrusivas.

## 4. Roadmap e Trabalhos Futuros

> **Nota:** Sugestões concretas para quem quiser continuar ou escalar este projeto.

1. **Melhoria Técnica:**  
Uma melhoria técnica prioritária seria testar diferentes *thresholds* de decisão. O modelo atual foi avaliado com um limiar fixo, mas o melhor valor depende da estratégia do anunciante. Se o objetivo for captar mais cliques reais, pode fazer sentido reduzir o *threshold* para aumentar o *Recall*. Se o objetivo for reduzir desperdício de orçamento, pode fazer sentido aumentar o *threshold* para melhorar a Precisão.

Também seria útil analisar a curva Precision-Recall, calibrar as probabilidades previstas pelo modelo e testar métodos adicionais de explicabilidade, como SHAP values. Estas melhorias ajudariam a compreender melhor as decisões do modelo e a adaptar a solução a diferentes contextos de negócio.

2. **Novas Variáveis:**  
O modelo poderia beneficiar da inclusão de novas variáveis relacionadas com a campanha e com o valor real do clique. Exemplos relevantes seriam: tipo de criativo, categoria do produto anunciado, custo por impressão, custo por clique, histórico de exposição do utilizador, frequência de anúncios mostrados ao mesmo utilizador e dia da semana.

Também seria importante incluir métricas posteriores ao clique, como conversão, compra, registo ou receita gerada. O modelo atual prevê cliques, mas um clique nem sempre representa valor final para o anunciante. Integrar dados de conversão permitiria evoluir de uma previsão de CTR para uma previsão mais próxima do retorno real da campanha.

3. **Escalabilidade (Deployment):**  
Como evolução futura, o modelo poderia ser colocado numa aplicação simples, por exemplo em **Streamlit**, permitindo que utilizadores não técnicos carregassem novos dados e obtivessem uma probabilidade estimada de clique para cada impressão.

Para isso, seria necessário guardar o modelo final, as etapas de pré-processamento e as transformações usadas, garantindo que novos dados são tratados da mesma forma que os dados de treino. A aplicação poderia incluir uma interface com a probabilidade prevista, as variáveis mais influentes e uma recomendação simples sobre priorizar ou não determinada impressão.

Numa fase mais avançada, a solução poderia evoluir para um dashboard de monitorização, com acompanhamento do desempenho do modelo ao longo do tempo, alertas de degradação de performance e análise dos principais fatores que influenciam os cliques em campanhas digitais.

---

**Data de Conclusão:** 15/05/2026  
**Versão do Projeto:** v4.0 Final
