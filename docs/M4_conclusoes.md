# Relatório de Conclusão e Entrega de Valor (Milestone 4)

## 1. Síntese de Resultados e Impacto

> **Nota:** Esta secção deve traduzir as métricas técnicas (F1-Score, RMSE, Silhouette, Lift) em resultados compreensíveis para qualquer pessoa.

* **O Problema Resolvido:** No Milestone 1, definimos como objetivo principal desenvolver um modelo de classificação binária capaz de prever se um utilizador irá clicar num anúncio digital, atingindo um AUC-ROC mínimo de 0,75 no conjunto de teste. O objetivo foi **alcançado**: o modelo final — XGBoost Otimizado — atingiu um **AUC-ROC de 0,7528** no conjunto de teste, superando o limiar definido. O segundo objetivo — identificar as 5 variáveis mais determinantes e fornecer recomendações acionáveis — foi igualmente cumprido dentro do prazo previsto no cronograma interno.

* **Interpretação dos Resultados:** O modelo consegue ordenar impressões publicitárias por probabilidade de clique com uma capacidade discriminativa de 75,28%. Em termos práticos: se apresentarmos ao modelo um par de impressões — uma que será clicada e outra que não será —, o modelo atribui uma pontuação mais alta à impressão clicada em **3 de cada 4 casos**. A melhoria face ao modelo base (Regressão Logística, AUC-ROC = 0,6412) foi de **+0,1097**, representando um ganho de 17,1% na capacidade preditiva. A variável `banner_area`, criada durante o projeto, revelou-se a mais importante do modelo com **32,6% da importância total**, validando retroativamente a decisão de *feature engineering*.

* **Valor para o Utilizador/Negócio:** Com esta solução, os anunciantes em plataformas de *Real-Time Bidding* podem concentrar os seus lances nas impressões com maior probabilidade de clique, reduzindo o desperdício de orçamento em impressões irrelevantes e aumentando o retorno sobre o investimento (ROI) das campanhas digitais. As recomendações acionáveis identificadas — privilegiar banners de maior área visual, concentrar lances nas primeiras horas da madrugada (0h–6h) e focar em dispositivos móveis (`device_type = 0`) — podem ser implementadas diretamente nas estratégias de *bidding* sem necessidade de alterações técnicas adicionais.

## 2. Análise Crítica e Limitações

> **Nota:** Identificar de forma honesta as fronteiras do projeto e onde o modelo pode falhar.

* **Limitações dos Dados:**
  * O volume de dados utilizado corresponde a apenas **12,4% do dataset original** (5.000.000 de 40,4 milhões de registos), por restrições técnicas e por sugestão da professora. Modelos treinados com a totalidade dos dados poderiam captar padrões mais raros e melhorar o AUC-ROC.
  * O dataset cobre apenas **10 dias**, o que impede o modelo de captar padrões sazonais (ex: variações de CTR em feriados, fins de semana ou eventos desportivos) e pode limitar a generalização para outros períodos.
  * Quase metade das variáveis são **anónimas (C14–C21)**, sem descrição do seu significado real, o que impede uma interpretação de negócio mais profunda e a identificação de novas oportunidades de *feature engineering*.
  * O **desequilíbrio severo (1:5)** entre cliques (17%) e não-cliques (83%) limita estruturalmente a capacidade do modelo de identificar todos os cliques reais, especialmente em segmentos de baixo volume.

* **Limitações do Modelo:**
  * Não foram aplicadas técnicas de reamostragem como **SMOTE** (*Synthetic Minority Over-sampling Technique*), o que poderia melhorar o *Recall* (atualmente 66,7%) e reduzir os 56.547 Falsos Negativos identificados na matriz de confusão.
  * O modelo trata cada impressão como um **evento independente**, sem contexto sequencial. Modelos como LSTMs ou Transformers poderiam captar padrões de comportamento do utilizador ao longo do tempo.
  * O XGBoost opera como uma **caixa semi-fechada**: a interação entre as 18 variáveis dentro do *ensemble* de 266 árvores não é diretamente interpretável por instância. Técnicas como SHAP permitiriam explicabilidade por instância.
  * A matriz de confusão foi calculada com o **threshold padrão de 0,5**, que pode não ser o ótimo para todas as estratégias de negócio — ajustar este limiar poderia melhorar o *trade-off* entre Precisão e *Recall*.

* **Contextos de Falha:**
  * O modelo **não é recomendado para novos publishers** (sites ou aplicações não presentes no treino), uma vez que o *Frequency Encoding* atribui frequência 0 a categorias desconhecidas, degradando a qualidade das previsões.
  * A análise da matriz de confusão revelou que os Falsos Negativos se concentram nas **impressões das 17h00**, um horário que o modelo sistematicamente subestima — possivelmente por variabilidade elevada do CTR nesse período.
  * O modelo não deve ser aplicado a **contextos fora da publicidade móvel** (ex: publicidade desktop ou *out-of-home*), uma vez que o dataset Avazu é 100% focado em dispositivos móveis.

## 3. Considerações Éticas e de Viés

* **Privacidade:** Todos os identificadores pessoais foram removidos, garantindo que o modelo analisa apenas padrões de comportamento anónimos. As colunas `id`, `device_id` e `device_ip` foram eliminadas da *pipeline* de modelação na fase de pré-processamento (Milestone 2). O dataset Avazu substitui estes identificadores por *hashes* criptográficas antes da disponibilização pública, impossibilitando a re-identificação de utilizadores reais. O projeto cumpre os requisitos do Regulamento Geral sobre a Proteção de Dados (RGPD).

* **Transparência:** Utilizámos técnicas de *Feature Importance* para garantir que as decisões do modelo são explicáveis e não operam como uma caixa negra: as top-5 variáveis concentram **70% do poder preditivo total**, tornando o modelo interpretável a nível agregado. A análise da matriz de confusão e a identificação dos contextos de falha fornecem igualmente um diagnóstico honesto das suas limitações. Para aplicações onde a explicabilidade por instância é um requisito legal ou regulatório, seria necessária a integração de técnicas como **SHAP** (*SHapley Additive exPlanations*) ou **LIME**.

## 4. Roadmap e Trabalhos Futuros

> **Nota:** Sugestões concretas para quem quiser continuar ou escalar este projeto.

1. **Melhoria Técnica:** Implementar técnicas de reamostragem (**SMOTE**) para lidar melhor com o desequilíbrio das classes (1:5). A geração de exemplos sintéticos da classe minoritária (cliques) poderia melhorar o *Recall* do modelo, reduzindo os 56.547 Falsos Negativos identificados. Em paralelo, otimizar o *threshold* de decisão (atualmente fixo em 0,5) através da análise da curva Precision-Recall, e integrar **SHAP** para explicabilidade por instância.

2. **Novas Variáveis:** Integrar dados sazonais e de calendário (dia da semana, feriados, eventos desportivos) para refinar as previsões de CTR ao longo do tempo — dimensão que o dataset de 10 dias não cobre. Incorporar o historial de interações do utilizador (*re-targeting*) e indicadores de qualidade do inventário publicitário (*viewability rate*) poderia igualmente aumentar o poder preditivo do modelo.

3. **Escalabilidade (Deployment):** Desenvolver uma **API REST** (FastAPI ou Flask) que exponha o modelo XGBoost para *scoring* em tempo real, integrável em sistemas de *Real-Time Bidding* com latência inferior a 10 ms por impressão. Em complemento, criar uma **interface web em Streamlit** para que gestores de marketing possam utilizar o modelo sem necessidade de código, e implementar **monitorização de *data drift*** (ex: Evidently AI) para detetar automaticamente degradação de desempenho em produção.

---

**Data de Conclusão:** 29 de abril de 2026  
**Versão do Projeto:** v4.0 Final
