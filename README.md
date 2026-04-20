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
No ecossistema da publicidade *online*, a eficiência de uma campanha depende da precisão com que um anúncio é direcionado ao utilizador. O desafio proposto pelo *dataset* Avazu consiste em mitigar o desperdício orçamental em impressões publicitárias que não geram interação. O problema de negócio foca-se em prever o comportamento do utilizador (clique ou não clique) perante variáveis contextuais como o tipo de dispositivo, o horário e o ambiente de navegação. Ao antecipar esta probabilidade, as organizações podem otimizar a alocação de recursos em tempo real, maximizando o retorno sobre o investimento publicitário (*Return on Ad Spend*).

### Objetivos do Projeto (SMART)
O objetivo principal é desenvolver um modelo de classificação (XGBoost/Random Forest) capaz de prever a probabilidade de clique em anúncios online (CTR) no dataset Avazu com 40 milhões de registos, analisando de que forma as variáveis contextuais (dispositivo, ambiente de navegação e tempo) influenciam essa decisão, garantindo um AUC-ROC superior a 0.75 e um Log Loss inferior a 0.40, de forma a permitir às empresas otimizar o seu investimento publicitário até ao fim do semestre letivo.

### Perguntas de Investigação
1. Quais são as janelas horárias que apresentam uma maior propensão para a conversão de cliques por parte dos utilizadores?
2. De que forma o tipo de dispositivo (*device type*) e as suas características técnicas influenciam a eficácia da entrega do anúncio?
3. Que categorias de aplicações ou domínios de internet geram tráfego com maior taxa de interação real face ao volume de impressões?

### Fonte de Dados e Dicionário
* **Dataset:** [Avazu Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction) (Disponível no *Kaggle*).
* **Dimensão:** O conjunto de dados original contém 40.428.967 registos e 24 colunas. Para viabilidade técnica no ambiente *Kaggle*, os testes iniciais são realizados sobre uma amostra aleatória estratificada.
* **Variável Objetivo:** `click` (Indica se o anúncio foi clicado: 0 para não, 1 para sim).

| Coluna | Tipo de Dado | Descrição Breve |
| :--- | :--- | :--- |
| **id** | Decimal | Identificador único do anúncio |
| **click** | Inteiro | Variável alvo (0: não clicou, 1: clicou) |
| **hour** | Inteiro | Formato AAAMMDDHH (Data e hora) |
| **C1** | Inteiro | Variável anónima (Atributo do anúncio) |
| **banner_pos** | Inteiro | Posição do *banner* no ecrã |
| **site_id** | Texto | Identificador do sítio *web* |
| **site_domain** | Texto | Domínio do sítio *web* |
| **site_category** | Texto | Categoria do sítio *web* |
| **app_id** | Texto | Identificador da aplicação |
| **app_domain** | Texto | Domínio da aplicação |
| **app_category** | Texto | Categoria da aplicação |
| **device_id** | Texto | Identificador do dispositivo |
| **device_ip** | Texto | Endereço IP do dispositivo |
| **device_model** | Texto | Modelo do dispositivo |
| **device_type** | Inteiro | Tipo de dispositivo |
| **device_conn_type** | Inteiro | Tipo de ligação à rede |
| **C14 - C21** | Inteiro | Variáveis anónimas contextuais |

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
