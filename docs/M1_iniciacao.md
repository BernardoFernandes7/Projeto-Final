# Milestone 1: Iniciação e Definição do Projeto
## 1. Descrição Detalhada do Problema
A previsão de cliques em anúncios on-line, conhecida na literatura como Click-Through Rate Prediction (CTR Prediction), constitui uma área de estudo inserida no campo da ciência de dados aplicada ao marketing digital e à publicidade programática. Tem como principal objetivo estimar a probabilidade de um usuário clicar em um anúncio específico a partir de informações históricas, comportamentais e contextuais. Em termos metodológicos, o problema é geralmente formulado como uma tarefa de classificação binária, na qual se procura prever a ocorrência ou não de um clique diante de uma determinada impressão publicitária. O crescimento exponencial da publicidade digital, impulsionado pela expansão do comércio eletrônico e pela intensificação do uso de plataformas digitais, conferiu elevada relevância a esse tema. 

Atualmente, sistemas automatizados operam em larga escala e em tempo real para decidir quais anúncios devem ser exibidos a cada usuário. Tais decisões baseiam-se, em grande medida, em modelos preditivos capazes de estimar a probabilidade de interação, tornando a previsão de cliques um componente central da infraestrutura tecnológica que sustenta a publicidade on-line contemporânea. A importância deste problema é tanto econômica quanto técnica. A precisão na estimativa da taxa de cliques influencia diretamente a eficiência na alocação de recursos publicitários, o retorno sobre investimento das campanhas e a experiência do usuário, que tende a receber conteúdos mais alinhados ao seu perfil e às suas preferências.

Ademais, o desenvolvimento de modelos de previsão envolve desafios significativos, como o tratamento de grandes volumes de dados, a alta dimensionalidade das variáveis explicativas, o desbalanceamento entre classes e a necessidade de respostas em tempo real. Do ponto de vista metodológico, a área tem evoluído de abordagens estatísticas tradicionais, como a regressão logística, para técnicas mais avançadas de aprendizado de máquina, incluindo modelos baseados em árvores de decisão, métodos de fatoração e redes neurais profundas. Essa evolução reflete a crescente complexidade dos dados disponíveis e a necessidade de capturar padrões não lineares e interações sofisticadas entre características do usuário, do anúncio e do contexto de exibição. Assim, a previsão de cliques em anúncios on-line consolida-se como um campo interdisciplinar que integra conhecimentos de estatística, ciência da computação, economia e marketing, desempenhando papel fundamental na compreensão e na otimização dos processos decisórios que estruturam a publicidade digital moderna.



## 2. Objetivos SMART
*Defina os objetivos do projeto seguindo a lógica SMART (Específico, Mensurável, Atingível,
Relevante e Temporal):*
1. **Objetivo 1:** Desenvolver e avaliar um modelo preditivo capaz de estimar a probabilidade de clique em anúncios on-line a partir de dados históricos e contextuais

2. **Objetivo 2:** Estruturar um processo completo de modelagem, contemplando as etapas de exploração e tratamento dos dados, seleção e engenharia de atributos, treinamento de algoritmos de aprendizado de máquina e validação dos resultados obtidos.

3. **Objetivo 3:** Comparar diferentes abordagens de modelagem, identificando aquela que apresenta melhor desempenho em termos de capacidade de generalização e precisão preditiva.

4. **Objetivo 4:** Avaliar o desempenho do modelo por meio de métricas adequadas a problemas de classificação binária.

## 3. Perguntas de Investigação

1. **Quais são as variáveis que exercem maior influência na probabilidade de um utilizador clicar num anúncio?

2. **Qual dos modelos de aprendizagem automática testados apresenta melhor desempenho preditivo na estimativa da taxa de cliques?

3. **De que forma o contexto de exibição (dispositivo, horário ou posicionamento do anúncio) influencia a probabilidade de clique?

4. **O modelo desenvolvido mantém desempenho consistente quando aplicado a diferentes subconjuntos de dados?

5. **De que forma uma melhoria na capacidade preditiva pode contribuir para decisões mais eficientes na segmentação e alocação de recursos publicitários?

## 4. Metodologia de Gestão (PBL)
* **Divisão de Tarefas:**
 * **Hugo:** Responsável pelos Dados (limpeza, tratamento e preparação do dataset).
 * **Bernardo:** Responsável pela Modelação Estatística.
 * Ambos os membros poderão executar partes das outras tarefas, em colaboração um com o outro

* **Ferramentas de Colaboração:** [Ex: GitHub Projects, WhatsApp].

## 5. Análise de Viabilidade dos Dados
* **Disponibilidade:** [Os dados já foram descarregados, estão em base de dados e prontos para começarem a ser tratados]
  
* **Qualidade Inicial:** Numa análise preliminar, verifica-se que o dataset contém um elevado número de variáveis categóricas, muitas das quais codificadas numericamente. Será necessário avaliar a existência de valores nulos, tipos de dados incorretos e possíveis inconsistências.
Além disso, é expectável um forte desbalanceamento entre as classes (clique vs. não clique), o que poderá exigir técnicas específicas de tratamento, como ajuste de métricas de avaliação ou técnicas de reamostragem.

* **Ética:** Os dados encontram-se anonimizados, não incluindo identificação direta dos utilizadores. Serão respeitados os princípios do RGPD e da ética em ciência de dados, garantindo que os dados são utilizados exclusivamente para fins académicos e que não existe tentativa de reidentificação de indivíduos.

## 6. Cronograma Interno
| Fase | Data Limite | Entregável Esperado |
| :--- | :--- | :--- |
| M1: Iniciação | [24/02/2026] | Repositório estruturado e Plano de Projeto. |
| M2: Exploração | [Data] | Notebook de EDA e Dados Processados. |
| M3: Modelação | [Data] | Comparação de algoritmos e métricas. |
| M4: Finalização| [Data] | Pitch e Relatório Final. |
---
*Data de última atualização: [18/02/2026]*
