# Milestone 1: Iniciação e Definição do Projeto

## 1. Descrição Detalhada do Problema Previsão de Clicks em Anúncios On-line

A previsão de cliques em anúncios on-line, conhecida na literatura como Click-Through Rate Prediction (CTR Prediction), constitui uma área de estudo inserida no campo da ciência de dados aplicada ao marketing digital e à publicidade programática. Tem como principal objetivo estimar a probabilidade de um usuário clicar em um anúncio específico a partir de informações históricas, comportamentais e contextuais. Em termos metodológicos, o problema é geralmente formulado como uma tarefa de classificação binária, na qual se procura prever a ocorrência ou não de um clique diante de uma determinada impressão publicitária.

O crescimento exponencial da publicidade digital, impulsionado pela expansão do comércio eletrônico e pela intensificação do uso de plataformas digitais, conferiu elevada relevância a esse tema. Atualmente, sistemas automatizados operam em larga escala e em tempo real para decidir quais anúncios devem ser exibidos a cada usuário. Tais decisões baseiam-se, em grande medida, em modelos preditivos capazes de estimar a probabilidade de interação, tornando a previsão de cliques um componente central da infraestrutura tecnológica que sustenta a publicidade on-line contemporânea.

A importância deste problema é tanto econômica quanto técnica. A precisão na estimativa da taxa de cliques influencia diretamente a eficiência na alocação de recursos publicitários, o retorno sobre investimento das campanhas e a experiência do usuário, que tende a receber conteúdos mais alinhados ao seu perfil e às suas preferências. Ademais, o desenvolvimento de modelos de previsão envolve desafios significativos, como o tratamento de grandes volumes de dados, a alta dimensionalidade das variáveis explicativas, o desbalanceamento entre classes e a necessidade de respostas em tempo real.

Do ponto de vista metodológico, a área tem evoluído de abordagens estatísticas tradicionais, como a regressão logística, para técnicas mais avançadas de aprendizado de máquina, incluindo modelos baseados em árvores de decisão, métodos de fatoração e redes neurais profundas. Essa evolução reflete a crescente complexidade dos dados disponíveis e a necessidade de capturar padrões não lineares e interações sofisticadas entre características do usuário, do anúncio e do contexto de exibição.

Assim, a previsão de cliques em anúncios on-line consolida-se como um campo interdisciplinar que integra conhecimentos de estatística, ciência da computação, economia e marketing, desempenhando papel fundamental na compreensão e na otimização dos processos decisórios que estruturam a publicidade digital moderna.


## 2. Objetivos SMART

1. **Objetivo 1:** Desenvolver e avaliar um modelo preditivo capaz de estimar a probabilidade de clique em anúncios on-line a partir de dados históricos e contextuais
2. **Objetivo 2:** Estruturar um processo completo de modelagem, contemplando as etapas de exploração e tratamento dos dados, seleção e engenharia de atributos, treinamento de algoritmos de aprendizado de máquina e validação dos resultados obtidos.
3.**Objetivo 3:**Comparar diferentes abordagens de modelagem, identificando aquela que apresenta melhor desempenho em termos de capacidade de generalização e precisão preditiva.
4.**Objetivo 4:** Avaliar o desempenho do modelo por meio de métricas adequadas a problemas de classificação binária


## 3. Perguntas de Investigação

A formulação de perguntas de investigação é essencial para orientar a análise empírica e delimitar o foco do estudo. No contexto da previsão de cliques em anúncios on-line, o presente trabalho procura responder às seguintes questões:

A primeira pergunta consiste em identificar quais variáveis exercem maior influência na probabilidade de um usuário clicar em um anúncio, Busca-se compreender quais características relacionadas ao perfil do usuário, ao conteúdo do anúncio ou ao contexto de exibição apresentam maior poder explicativo no modelo preditivo.

A segunda questão investiga qual modelo de aprendizado de máquina apresenta melhor desempenho na previsão da taxa de cliques, comparando diferentes abordagens metodológicas com base em métricas apropriadas para problemas de classificação binária.

A terceira pergunta procura analisar como diferentes contextos de exibição (como horário, dispositivo ou posicionamento do anúncio) influenciam a probabilidade de clique. O objetivo é compreender se existem padrões comportamentais associados a determinadas condições de visualização que impactam significativamente a interação do usuário.

A quarta questão busca verificar se o modelo desenvolvido mantém desempenho consistente quando aplicado a diferentes subconjuntos de dados, permitindo avaliar sua capacidade de generalização e robustez.

Por fim, a quinta pergunta investiga em que medida a melhoria na capacidade preditiva do modelo pode contribuir para decisões mais eficientes no contexto da publicidade digital, considerando possíveis impactos na segmentação e na alocação de recursos.


## 4. Identificação das Ferramentas e Bibliotecas Python

Para o desenvolvimento do presente trabalho, prevê-se a utilização de um conjunto de bibliotecas amplamente empregadas em projetos de ciência de dados e aprendizado de máquina em Python. Contudo, importa salientar que a escolha definitiva das ferramentas dependerá dos resultados obtidos ao longo das diferentes etapas do projeto, podendo ser ajustada conforme as necessidades identificadas durante a análise e modelagem dos dados.

Numa fase inicial, é expectável a utilização de bibliotecas voltadas para a manipulação e tratamento de dados, como Pandas e NumPy, que permitem estruturar, limpar e transformar conjuntos de dados de forma eficiente. Essas ferramentas são particularmente adequadas para a realização de análises exploratórias, tratamento de valores ausentes e preparação das variáveis para a modelagem.

Para a análise exploratória e visualização de dados, poderão ser utilizadas bibliotecas como Matplotlib e Seaborn, que possibilitam a construção de gráficos estatísticos e a identificação de padrões, distribuições e possíveis relações entre variáveis. No entanto, a adoção dessas ferramentas dependerá da complexidade dos dados e da necessidade de análises visuais mais detalhadas.

No que se refere à modelagem preditiva, é provável a utilização da biblioteca Scikit-learn, dada a sua versatilidade na implementação de algoritmos de classificação e na avaliação de modelos por meio de métricas apropriadas. Ainda assim, caso o desempenho obtido não seja satisfatório, poderão ser consideradas alternativas mais avançadas, como bibliotecas especializadas em métodos de boosting ou redes neurais, sempre em função dos resultados empíricos alcançados.

O ambiente de desenvolvimento previsto é o Jupyter Notebook, por facilitar a integração entre código, visualizações e documentação descritiva. Todavia, essa escolha também poderá ser adaptada caso se revele necessária outra solução mais adequada às exigências do projeto.

Dessa forma, a seleção das ferramentas não deve ser entendida como definitiva, mas sim como uma proposta inicial orientada pelas boas práticas da área, sujeita a revisão conforme o progresso e os desafios encontrados ao longo do desenvolvimento do trabalho.


## 5. Metodologia de Gestão (PBL)
* **Divisão de Tarefas:**
Ao longo deste trabalho, todas as tarefas serão divididas entre os dois membros, e iremos nos organizando continuamente para garantir colaboração e eficiência.
* **Ferramentas de Colaboração:** [Ex: WhatsApp, Discord].


## 6. Análise de Viabilidade dos Dados
* **Disponibilidade:** [Os dados serão carregados no dia 18/02, devido a duvidas a esclarecer com a professora]
* **Qualidade Inicial:** [--.]
* **Ética:** [--]


## 7. Cronograma Interno
| Fase | Data Limite | Entregável Esperado |
| :--- | :--- | :--- |
| M1: Iniciação | [Data] | Repositório estruturado e Plano de Projeto. |
| M2: Exploração | [Data] | Notebook de EDA e Dados Processados. |
| M3: Modelação | [Data] | Comparação de algoritmos e métricas. |
| M4: Finalização| [Data] | Pitch e Relatório Final. |
---
*Data de última atualização: [18/02/2026]*
