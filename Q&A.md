# Perguntas e Respostas Assíncronas (Milestone 4)

## 1. Porque foi escolhido o AUC-ROC como métrica principal?

O AUC-ROC foi escolhido porque o problema é de classificação binária e a variável alvo `click` está fortemente desequilibrada. Cerca de 83% dos registos correspondem a não-cliques e cerca de 17% correspondem a cliques reais.

Neste contexto, a *Accuracy* seria uma métrica pouco adequada. Um modelo que previsse sempre “não clique” teria uma taxa de acerto aparentemente elevada, mas não teria utilidade prática para identificar impressões com maior probabilidade de gerar clique.

O AUC-ROC é mais adequado porque avalia a capacidade do modelo para distinguir e ordenar impressões com maior e menor probabilidade de clique, independentemente de um limiar de decisão específico. Isto é importante em publicidade digital, porque o objetivo não é apenas prever uma classe final, mas apoiar a priorização de oportunidades com maior potencial de interação.

O F1-Score foi utilizado como métrica secundária, porque permite avaliar o equilíbrio entre Precisão e *Recall*. Esta métrica ajuda a perceber se o modelo consegue identificar cliques reais sem gerar demasiadas previsões positivas erradas.

## 2. Porque foi escolhido o XGBoost otimizado como modelo final?

O XGBoost otimizado foi escolhido porque apresentou o melhor desempenho entre os modelos testados. A Regressão Logística foi usada como modelo de referência, o *Random Forest* apresentou uma melhoria clara, e o XGBoost foi o melhor modelo antes da otimização.

Após a otimização de hiperparâmetros, o XGBoost atingiu **AUC-ROC = 0.7509** no conjunto de teste, cumprindo o objetivo SMART definido na Milestone 1, que era atingir pelo menos **AUC-ROC = 0.75**. O modelo também obteve **F1-Score = 0.4223**.

A escolha do XGBoost foi justificada por três razões principais:

1. apresentou melhor desempenho do que a Regressão Logística e o *Random Forest*;
2. cumpriu o objetivo mínimo definido para o projeto;
3. manteve estabilidade em validação cruzada, com média de AUC-ROC próxima do resultado obtido no conjunto de teste.

Além disso, a análise de importância dos atributos permitiu interpretar quais variáveis tiveram maior peso na previsão. Entre as variáveis mais relevantes estiveram `banner_area`, `C16`, `device_type`, `C21` e `site_id`.

## 3. Quais são as principais limitações do modelo final?

A principal limitação é que o modelo prevê cliques, mas não prevê conversões, compras, registos ou receita final. Um clique pode indicar interesse, mas não garante retorno financeiro para o anunciante.

Outra limitação está relacionada com os erros do modelo. A matriz de confusão mostra que ainda existem Falsos Positivos e Falsos Negativos. Os Falsos Positivos representam impressões previstas como potenciais cliques, mas que não geraram clique. Os Falsos Negativos representam cliques reais que o modelo não conseguiu identificar.

Também existem limitações nos próprios dados. O projeto utilizou uma amostra de 5.000.000 registos, apesar de o conjunto original Avazu conter cerca de 40 milhões de registos. Além disso, algumas variáveis são anonimizadas, como `C14` a `C21`, o que limita a interpretação de negócio de algumas decisões do modelo.

Por fim, o modelo identifica padrões estatísticos, mas não prova causalidade. Por exemplo, `banner_area` foi a variável mais importante no modelo final, mas isso não significa que aumentar sempre a área do anúncio cause automaticamente mais cliques. Significa apenas que, nos dados analisados, essa variável teve valor preditivo relevante.

Assim, o modelo deve ser entendido como uma ferramenta de apoio à decisão e não como uma solução automática perfeita.
