# Otimização de Campanhas Publicitárias Digitais: Previsão de Taxas de Clique (*Click-Through Rate*)

## Identificação da Equipa
* **Grupo nº:** 6
* **Membros:**
 * Bernardo Fernandes - 2023132747
 * Hugo Grou - 2023137127

## Organização do Repositório
A estrutura deste projeto segue as boas práticas de Ciência de Dados e Engenharia de Software:
* **`data/`**: Armazenamento de dados (dados brutos em `raw/` e processados em `processed/`).
* **`docs/`**: Documentação técnica detalhada dividida por *Milestones* (M1, M2 e M3).
* **`notebooks/`**: *Jupyter Notebooks* para experimentação, limpeza e modelação. 
* **`src/`**: Código-fonte modular (scripts `.py`) para funções reutilizáveis.
* **`reports/`**: Relatórios finais, apresentações e exportação de figuras (`figures/`).
* **`requirements.txt`**: Ficheiro de configuração com as bibliotecas necessárias e respetiva descrição.

## 1. Iniciação (Milestone 1)

### Contexto e Problema de Negócio
No ecossistema da publicidade *online*, a eficiência de uma campanha depende da precisão com que um anúncio é direcionado ao utilizador. O desafio proposto pelo parceiro Avazu consiste em mitigar o desperdício orçamental em impressões publicitárias que não geram interação. O problema de negócio foca-se em prever o comportamento do utilizador (clique ou não clique) perante variáveis contextuais como o tipo de dispositivo, o horário e o ambiente de navegação. Ao antecipar esta probabilidade, as organizações podem otimizar a alocação de recursos em tempo real, maximizando o retorno sobre o investimento publicitário (*Return on Ad Spend*).

### Objetivos do Projeto (SMART)
O objetivo central deste trabalho é desenvolver e validar, até ao final do semestre letivo, um modelo de classificação binária capaz de prever cliques em anúncios com uma métrica de desempenho *AUC-ROC* superior a 0.75. Através da análise do *dataset* da Avazu, pretende-se criar uma ferramenta preditiva que suporte a decisão de compra de inventário publicitário em tempo real.

### Perguntas de Investigação
1. Quais são as janelas horárias que apresentam uma maior propensão para a conversão de cliques por parte dos utilizadores?
2. De que forma o tipo de dispositivo (*device type*) e as suas características técnicas influenciam a eficácia da entrega do anúncio?
3. Que categorias de aplicações ou domínios de internet geram tráfego com maior taxa de interação real face ao volume de impressões?

### Fonte de Dados
* **Dataset:** [Avazu Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction) (Disponível no *Kaggle*).
* **Dimensão:** O conjunto de dados original contém 40.428.967 registos e 24 colunas. Para viabilidade técnica no ambiente *Kaggle*, os testes iniciais são realizados sobre uma amostra aleatória estratificada.

## 2. Exploração (Milestone 2)
### Limpeza e Preparação
* [Fase em desenvolvimento: Incluirá tratamento de *outliers*, conversão de tipos de dados e gestão de valores mascarados. Detalhes em `docs/M2_exploracao.md`]

### Principais Conclusões (EDA)
* **Ponto-chave:** Durante a análise exploratória inicial (*EDA*), identificou-se um forte desequilíbrio de classes (apenas ~17% de cliques), o que exigirá o uso de métricas de avaliação como *AUC-ROC* e *F1-Score* em detrimento da *Accuracy*.

## 3. Modelação (Milestone 3)
### Abordagem Técnica
* **Modelos:** [A definir: Candidatos principais *XGBoost* e *Random Forest*]
* **Métrica Principal:** *AUC-ROC* e *Log Loss*

## 4. Finalização (Milestone 4)
### Resposta ao Problema
[A preencher na conclusão do projeto com base nos resultados obtidos.]

### Recomendações de Inovação
1. [Sugestão prática baseada nos resultados finais da modelação]

## Como Reproduzir este Projeto
1. Clone o repositório: `git clone https://github.com/BernardoFernandes7/Projeto-Final.git`
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute os *notebooks* na pasta `notebooks/` seguindo a ordem numérica, garantindo a ligação ao *dataset* oficial no *Kaggle*.

---
**Instituição:** Coimbra Business School | ISCAC  
**Curso:** Licenciatura em Ciência de Dados para a Gestão  
**Unidade Curricular:** Projeto em Ciência de Dados  
**Professor Responsável:** Dora Melo (dmelo@iscac.pt)
