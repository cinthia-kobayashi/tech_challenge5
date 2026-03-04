
# 📊 Tech Challenge 5 — Análise de Reclamações Financeiras com NLP

Projeto desenvolvido para o **Tech Challenge 5 da Pós Tech FIAP** com o objetivo de analisar reclamações de consumidores do setor financeiro utilizando **Processamento de Linguagem Natural (NLP)** e **Machine Learning / Deep Learning**.

A análise utiliza a base pública **Consumer Complaint Database** do **Consumer Financial Protection Bureau (CFPB)**.

---

# 🎯 Objetivo do Projeto

O objetivo deste trabalho é:

- Processar milhões de reclamações de consumidores
- Identificar **padrões de insatisfação**
- Criar um **modelo de classificação de sentimento**
- Identificar **principais dores dos clientes por categoria de produto**
- Construir um **dashboard interativo para visualização dos resultados**

---

# 📊 Base de Dados

Fonte:  
Consumer Complaint Database  
https://www.consumerfinance.gov/data-research/consumer-complaints/

Download realizado em:

```
18/02/2026
```

Após limpeza inicial da base:

```
Total de registros com narrativa: 3.717.193
```

Os produtos foram consolidados em **5 grandes categorias**:

| Produto | Descrição |
|------|------|
| credit reporting | relatórios de crédito |
| debt collection | cobrança de dívidas |
| retail banking | serviços bancários |
| mortgages and loans | empréstimos e financiamentos |
| credit cards | cartões de crédito |

---

# 📈 Distribuição de Reclamações

| Produto | Percentual aproximado |
|------|------|
| Credit Reporting | ~67% |
| Debt Collection | ~11% |
| Retail Banking | ~8% |
| Mortgages and Loans | ~8% |
| Credit Cards | ~6% |

Observa-se que **problemas em relatórios de crédito dominam amplamente as reclamações**.

---

# 🧠 Pipeline de Processamento

O pipeline completo de NLP inclui as seguintes etapas.

## 1️⃣ Limpeza de texto

Foram aplicadas as seguintes técnicas:

- remoção de pontuação
- remoção de números
- remoção de stopwords
- normalização de texto
- lematização utilizando **spaCy**

Exemplo:

```
Texto original:
"I had paid off this card within one year..."

Texto processado:
"pay card year avoid interest statement"
```

---

## 2️⃣ Vetorização

Para transformar texto em dados numéricos foi utilizado **TF-IDF**.

Parâmetros principais:

```python
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=5
)
```

---

## 3️⃣ Criação da variável alvo

Como a base contém **apenas reclamações**, foi necessário criar uma variável de sentimento.

Foi utilizada a variável:

```
Company response to consumer
```

Classificação utilizada:

| Classe | Critério |
|------|------|
| Negativo | reclamação sem compensação |
| Positivo | houve compensação ao consumidor |

Distribuição final:

| Classe | Percentual |
|------|------|
| Negativo | 76.83% |
| Positivo | 23.17% |

---

# 🤖 Modelos Treinados

Foram avaliados três modelos.

## Logistic Regression
Modelo baseline tradicional para vetores **TF-IDF**.

## Random Forest
Modelo de comparação.

## Deep Learning (MLP — Keras)

Arquitetura utilizada:

```
Input TF-IDF
Dense (128)
Dropout
Dense (64)
Dropout
Sigmoid Output
```

---

# 📊 Resultados do Modelo de Deep Learning

```
Accuracy: 78.3%
ROC-AUC: 0.77
```

Classification Report:

```
Classe 0 (negativa)
precision: 0.81
recall: 0.93

Classe 1 (positiva)
precision: 0.56
recall: 0.29
```

Observações:

- O modelo possui **alta capacidade de identificar reclamações negativas**
- A classe positiva é **mais difícil de identificar**
- Isso ocorre devido ao **desbalanceamento do dataset**

---

# 🔎 Análise das Dores dos Clientes

Foram extraídos os **termos mais relevantes por produto** utilizando TF-IDF.

### Credit Reporting
Problemas com:

- credit report
- inaccurate information
- disputes

Indicam erros em **histórico de crédito e score**.

### Debt Collection
Termos frequentes:

- collection
- debt
- letter
- company

Indicam **cobranças indevidas ou contestadas**.

### Retail Banking
Principais termos:

- account
- money
- funds
- check

Relacionados a **movimentação financeira e saldo**.

### Mortgages and Loans
Termos principais:

- loan
- payment
- mortgage

Relacionados a **pagamentos e contratos de financiamento**.

### Credit Cards
Termos relevantes:

- charge
- payment
- dispute

Relacionados a **cobranças indevidas e contestação de transações**.

---

# 📊 Dashboard Interativo

O projeto inclui um **dashboard desenvolvido em Streamlit** para visualização dos resultados.

O dashboard apresenta:

- distribuição de produtos
- distribuição de sentimento
- termos relevantes por produto
- nuvens de palavras
- métricas do modelo de Deep Learning

---

# 👩‍💻 Grupo

- Agnes Miki Magario  
- Cinthia Mayumi Kobayashi  
- Lina Satie Kobata Felippe  

---

# 🏫 Pós Tech FIAP

Projeto desenvolvido para o **Tech Challenge 5** da **Pós Tech FIAP**.
