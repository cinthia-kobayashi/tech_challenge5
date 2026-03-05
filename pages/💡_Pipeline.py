import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Tech Challenge - FIAP",
    page_icon='📊',
    layout="wide"

)

#Colocar o nome do grupo no Barra Lateral.
with st.sidebar:
    st.markdown(''' Grupo:  
        Agnes Miki Magario  
        Cinthia Mayumi Kobayashi  
        Lina Satie Kobata Felippe
 ''')

st.title("POSTECH - FIAP (Data Analytics - BB)")
st.caption("Essa página tem como objetivo explicar o código utilizado no tratamento de dados do trabalho do TechChallenge 5")
st.divider()

st.subheader("TechChallenge Fase 5 - DeepLearning",)
st.image("images/logo_210.png")

st.markdown('''
Analisaremos agora as fases de processamento do texto e análise de sentimentos e contidos na base de reclamações "Consumer Complaint Database
API docs".  
De acordo com site, os dados vem de uma coleção de reclamações sobre produtos e serviços financeiros de consumo.
Os dados são enviados para empresas para resposta. 
As reclamações são publicadas após a empresa responder, confirmando uma relação comercial com o consumidor, ou após 15 dias.
O banco de dados geralmente é atualizado diariamente.
Para mais informações e downloads dos dados direto no site do [CFPB, clique aqui](https://cfpb.github.io/api/ccdb/index.html).
        ''')


st.text("")
st.markdown("**Os dados utilizados nesse trabalho foram baixados em 18/02/2026**")
st.info(
    """
    O código completo e os arquivos utilizados estão no [GitHub!](https://github.com/cinthia-kobayashi/tech_challenge5)
    """,
    icon="🔗",
)

st.header("Bloco 1 — Importação de bibliotecas")
st.markdown("""
Aqui fizemos a importação de bibliotecas para:
- Manipulação de dados (pandas, numpy)
- Expressões regulares (re)
- Vetorização TF-IDF
- Métricas e avaliação
- Modelos clássicos e Deep Learning
- Pipeline Transformer (HuggingFace) para comparação exploratória
""")

st.code("""
#Sistema
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#NLP
import spacy
from transformers import pipeline
from tqdm import tqdm

#Scikit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
)

#Deep Learning (TensorFlow / Keras)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Utils
import joblib
""")
st.divider()


# ==========================================================
# BLOCO 2 — Leitura do CSV + backup
# ==========================================================
st.header("Bloco 2 — Carregamento da base")
st.markdown("""
Carregamos o arquivo completo `complaints.csv` e armazenamos uma cópia (`df_backup`),
permitindo retornar ao estado original caso necessário.
""")
st.info("O arquivo completo `complaints.csv` não está disponível em nosso *Github* devido ao seu tamanho (7,5gb)")

st.code("""
df = pd.read_csv("complaints.csv")
df_backup = df.copy()
""")

st.markdown("**Saída (shape da base original):**")
st.code("(13682539, 18)")
st.divider()



st.header("Bloco 3 — Remoção de registros sem narrativa")
st.markdown("""
A coluna **Consumer complaint narrative** contém o texto da reclamação.
Como há milhões de registros sem texto, removemos linhas nulas para reduzir custo computacional e ruído.
""")

st.code("""
df = df[df["Consumer complaint narrative"].notnull()].copy()
""")

st.markdown("**Saída (shape após remover narrativas nulas):**")
st.code("(3717193, 18)")
st.divider()


st.header("Bloco 4 — Consolidação de produtos (Product → 5 classes)")
st.markdown("""
Para reduzir desbalanceamento e agrupar produtos similares, consolidamos os produtos em 5 classes:

- credit reporting  
- debt collection  
- mortgages and loans  
- credit cards  
- retail banking
""")

st.code("""
def consolidar_produtos(df):
    mapping = {
        # credit reporting
        "Credit reporting": "credit reporting",
        "Credit reporting or other personal consumer reports": "credit reporting",
        "Credit reporting, credit repair services, or other personal consumer reports": "credit reporting",

        # debt collection
        "Debt collection": "debt collection",
        "Debt or credit management": "debt collection",

        # mortgages and loans
        "Mortgage": "mortgages and loans",
        "Vehicle loan or lease": "mortgages and loans",
        "Consumer Loan": "mortgages and loans",
        "Student loan": "mortgages and loans",
        "Payday loan": "mortgages and loans",
        "Payday loan, title loan, or personal loan": "mortgages and loans",
        "Payday loan, title loan, personal loan, or advance loan": "mortgages and loans",

        # credit cards
        "Credit card": "credit cards",
        "Credit card or prepaid card": "credit cards",
        "Prepaid card": "credit cards",

        # retail banking
        "Checking or savings account": "retail banking",
        "Bank account or service": "retail banking",
        "Money transfers": "retail banking",
        "Money transfer, virtual currency, or money service": "retail banking",
        "Virtual currency": "retail banking",
        "Other financial service": "retail banking"
    }

    df = df.copy()
    df["Product Consolidated"] = df["Product"].map(mapping)
    return df

df = consolidar_produtos(df)
""")

st.markdown("**Saída: distribuição por classe consolidada**")
df_consolidated = pd.DataFrame({
    "Product Consolidated": ["credit reporting", "debt collection", "retail banking", "mortgages and loans", "credit cards"],
    "Count": [2497507, 408350, 295820, 288388, 227128]
})
st.dataframe(df_consolidated, use_container_width=True, hide_index=True)
st.divider()



st.header("Bloco 5 — Amostragem balanceada (80k por classe)")
st.markdown("""
Como a base é muito grande, selecionamos uma amostra balanceada com **80.000 registros por classe**.
Isso garante comparabilidade e reduz o custo computacional, evitando que a classe mais frequente domine o treino.
""")

st.code("""
sample_size = 80000

df_sample = (
    df.groupby("Product Consolidated", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), sample_size), random_state=42))
      .reset_index(drop=True)
)

df_sample["Product Consolidated"].value_counts()
""")

st.markdown("**Saída: contagem por classe e shape da amostra**")
df_sample_counts = pd.DataFrame({
    "Product Consolidated": ["credit cards", "credit reporting", "debt collection", "mortgages and loans", "retail banking"],
    "Count": [80000, 80000, 80000, 80000, 80000]
})
st.dataframe(df_sample_counts, use_container_width=True, hide_index=True)
st.code("df_sample shape: (400000, 19)")
st.divider()


st.header("Bloco 6 — Sentimento exploratório por palavras-chave (sentiment_rule)")
st.markdown("""
Construímos um rótulo exploratório de sentimento baseado em regras simples:
- conta palavras-chave negativas e positivas dentro do texto
- se negativo > positivo, marca como 0; se positivo > negativo, marca como 1; caso contrário, NaN.

⚠️ Esse método é simples e não interpreta contexto/negação perfeitamente.
""")

st.code("""
negative_keywords = [
    "fraud", "unauthorized", "error", "wrong", "charged",
    "denied", "complaint", "issue", "problem",
    "not resolved", "never received", "scam",
    "harassment", "threat", "foreclosure", "late fee",
    "debt", "collection", "dispute", "inaccurate",
    "overcharge", "misleading"
]

positive_keywords = [
    "resolved", "thank you", "satisfied",
    "helpful", "fixed", "appreciate",
    "closed my complaint", "solved"
]

def create_sentiment_rule(text):
    text = str(text).lower()
    neg_count = sum(k in text for k in negative_keywords)
    pos_count = sum(k in text for k in positive_keywords)

    if neg_count > pos_count:
        return 0
    elif pos_count > neg_count:
        return 1
    return np.nan

df_sample["sentiment_rule"] = df_sample["Consumer complaint narrative"].apply(create_sentiment_rule)
df_rule = df_sample[df_sample["sentiment_rule"].notnull()].copy()

df_rule["sentiment_rule"].value_counts(normalize=True)
""")

st.markdown("**Saída: distribuição (proporção) do sentiment_rule**")
st.code("""sentiment_rule
0.0    0.960316
1.0    0.039684
Name: proportion, dtype: float64""")
st.divider()


st.header("Bloco 7 — Transformer (HuggingFace) para comparação exploratória")
st.markdown("""
Aplicamos um Transformer do HuggingFace como comparação exploratória:
`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`.

⚠️ Observação importante: o modelo é treinado em **notícias financeiras** (domínio diferente),
então a performance/consistência pode ser limitada.
""")

st.code("""
df_compare = df_rule.sample(15000, random_state=42).copy()

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    device=0  # se der erro, use -1
)

batch_size = 32
texts = df_compare["Consumer complaint narrative"].tolist()

predictions = []
for i in tqdm(range(0, len(texts), batch_size), desc="HF sentiment"):
    batch = texts[i:i+batch_size]
    outputs = sentiment_pipeline(batch, truncation=True, max_length=512)
    predictions.extend(outputs)

df_compare["hf_label"] = [p["label"] for p in predictions]
df_compare["hf_score"] = [p["score"] for p in predictions]

df_compare["sentiment_hf"] = df_compare["hf_label"].map({
    "positive": 1,
    "negative": 0,
    "neutral": np.nan
})

df_compare = df_compare[df_compare["sentiment_hf"].notnull()].copy()
""")

st.markdown("**Saída: comparação Heurística vs Transformer**")
st.code(r"""Comparação Heurística vs Transformer (binário, sem neutral):
Distribuição Heurística (rule):
sentiment_rule
0.0    0.971443
1.0    0.028557
Name: proportion, dtype: float64

Distribuição Transformer (hf):
sentiment_hf
0.0    0.957544
1.0    0.042456
Name: proportion, dtype: float64

Matriz de Confusão (rule como referência):
[[3695  149]
 [  94   19]]

Classification report (rule como referência):
              precision    recall  f1-score   support

         0.0       0.98      0.96      0.97      3844
         1.0       0.11      0.17      0.14       113

    accuracy                           0.94      3957
   macro avg       0.54      0.56      0.55      3957
weighted avg       0.95      0.94      0.94      3957""")
st.divider()



st.header("Bloco 8 — Variável alvo final (sentiment_final) baseada em outcome")
st.markdown("""
Definimos um alvo supervisionado **binário** com base no desfecho reportado pela empresa:
- **1 (positivo)**: quando houve algum tipo de *relief* ao consumidor
  - Closed with monetary relief
  - Closed with non-monetary relief
- **0 (negativo)**: demais respostas

Essa abordagem produz um target balanceado o suficiente para modelagem e interpretação.
""")

st.code("""
def create_sentiment_final(response):
    if response in [
        "Closed with monetary relief",
        "Closed with non-monetary relief"
    ]:
        return 1
    return 0

df_sample["sentiment_final"] = df_sample["Company response to consumer"].apply(create_sentiment_final)

df_sample["Company response to consumer"].value_counts()
df_sample["sentiment_final"].value_counts(normalize=True)
""")

st.markdown("**Saída: distribuição Company response to consumer e sentiment_final**")
st.code("""Company response to consumer
Closed with explanation            304426
Closed with non-monetary relief     65795
Closed with monetary relief         26872
Untimely response                    1931
Closed                                859
In progress                           117
Name: count, dtype: int64

sentiment_final
0    0.768332
1    0.231667
Name: proportion, dtype: float64""")
st.divider()



st.header("Bloco 9 — Pré-processamento de texto (limpeza, stopwords, lematização)")
st.markdown("""
Atendendo ao requisito de pré-processamento, aplicamos:
- normalização (lowercase)
- remoção de tokens de anonimização (`xx`, `xxxx`)
- remoção de números isolados
- lematização (spaCy)
- remoção de stopwords e pontuação (via filtros no pipeline)
O resultado final é a coluna **text_processed**.
""")

st.code(r"""
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def normalize_for_spacy(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b[x]{2,}\b", " ", text)  # remove xx/xxxx
    text = re.sub(r"\b\d+\b", " ", text)     # remove números isolados
    text = re.sub(r"\s+", " ", text).strip() # normaliza espaços
    return text

df_sample["text_norm"] = df_sample["Consumer complaint narrative"].apply(normalize_for_spacy)

processed = []
for doc in tqdm(nlp.pipe(df_sample["text_norm"].tolist(), batch_size=1000),
                total=len(df_sample), desc="spaCy lemmatization"):
    tokens = [
        t.lemma_
        for t in doc
        if t.is_alpha
        and not t.is_stop
        and not t.is_punct
        and len(t) > 2
    ]
    processed.append(" ".join(tokens))

df_sample["text_processed"] = processed
""")

st.markdown("**Saída: exemplo de textos (narrative → text_processed)**")
df_example = pd.DataFrame({
    "Consumer complaint narrative": [
        "Review the attached documents. I ask the burea...",
        "I had paid off this card within one year of my...",
        "This is my second complaint against this compa..."
    ],
    "text_processed": [
        "review attach document ask bureau commence inq...",
        "pay card year card use avoid interest statemen...",
        "second complaint company open issue second acc..."
    ]
})
st.dataframe(df_example, use_container_width=True, hide_index=True)
st.divider()



st.header("Bloco 10 — Vetorização TF-IDF (5.000 features) + treino/teste")
st.markdown("""
Transformamos o texto em vetores numéricos com **TF-IDF** (unigramas e bigramas),
usando como entrada `text_processed`.

O parâmetro `max_features=5000` foi usado para reduzir custo de memória e permitir o treino do modelo Deep Learning.
""")

st.code(r"""
RANDOM_STATE = 42

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words=None,                 # stopwords já removidas no text_processed
    ngram_range=(1, 2),
    min_df=5,
    token_pattern=r"\b[a-zA-Z]{3,}\b"
)

X = vectorizer.fit_transform(df_sample["text_processed"])
y = df_sample["sentiment_final"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)
""")

st.markdown("**Saída:** (as métricas são apresentadas nos próximos blocos)")
st.divider()


st.header("Bloco 11 — Modelo baseline: Logistic Regression")
st.markdown("""
Treinamos Logistic Regression como baseline forte para TF-IDF e avaliamos com:
- precision, recall, f1-score
- accuracy
- ROC-AUC
""")

st.code("""
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

y_proba = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
""")

st.markdown("**Saída: Logistic Regression (teste)**")
st.code(r"""              precision    recall  f1-score   support

           0       0.81      0.94      0.87     61467
           1       0.56      0.26      0.35     18533

    accuracy                           0.78     80000
   macro avg       0.68      0.60      0.61     80000
weighted avg       0.75      0.78      0.75     80000

Accuracy: 0.7814625
ROC-AUC: 0.7691599135994273""")
st.divider()



st.header("Bloco 12 — Modelo de comparação: Random Forest (50 árvores)")
st.markdown("""
Treinamos Random Forest para comparação com Logistic Regression.
Observação: Random Forest pode ser mais lenta em TF-IDF (alta dimensionalidade).
""")

st.code("""
rf = RandomForestClassifier(
    n_estimators=50,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\\n=== Random Forest (50 trees) ===")
print(classification_report(y_test, y_pred_rf))
""")

st.markdown("**Saída: Random Forest (teste)**")
st.code(r"""              precision    recall  f1-score   support

           0       0.80      0.95      0.87     61467
           1       0.58      0.21      0.31     18533

    accuracy                           0.78     80000
   macro avg       0.69      0.58      0.59     80000
weighted avg       0.75      0.78      0.74     80000""")
st.divider()



st.header("Bloco 13 — Modelagem com Deep Learning: Keras MLP em TF-IDF")
st.markdown("""
Para atender ao requisito de **Deep Learning**, treinamos uma rede neural do tipo MLP (Multi-Layer Perceptron)
com camadas densas e dropout, usando como entrada o TF-IDF (5.000 features).

Como o Keras não treina diretamente com matrizes esparsas no formato CSR,
convertemos `X_train` e `X_test` para arrays densos (`toarray()`), usando `float32`.
""")

st.code(r"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# sparse -> dense
X_train_dense = X_train.toarray().astype("float32")
X_test_dense  = X_test.toarray().astype("float32")

y_train_np = y_train.to_numpy(dtype="int32")
y_test_np  = y_test.to_numpy(dtype="int32")

model_dl = keras.Sequential([
    layers.Input(shape=(X_train_dense.shape[1],)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model_dl.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc")
    ]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=2,
        restore_best_weights=True
    )
]

history = model_dl.fit(
    X_train_dense, y_train_np,
    validation_split=0.2,
    epochs=10,
    batch_size=512,
    callbacks=callbacks,
    verbose=1
)

y_proba_dl = model_dl.predict(X_test_dense, batch_size=1024).ravel()
y_pred_dl = (y_proba_dl >= 0.5).astype(int)

print("\\n=== Deep Learning (Keras MLP) ===")
print(classification_report(y_test_np, y_pred_dl))
print("Accuracy:", accuracy_score(y_test_np, y_pred_dl))
print("ROC-AUC:", roc_auc_score(y_test_np, y_proba_dl))
""")

st.markdown("**Saída: logs de treino e métricas (teste)**")
st.code(r"""Epoch 1/10
500/500  9s 14ms/step - accuracy: 0.7746 - auc: 0.7452 - loss: 0.4708 - val_accuracy: 0.7767 - val_auc: 0.7662 - val_loss: 0.4570
Epoch 2/10
500/500  6s 12ms/step - accuracy: 0.7862 - auc: 0.7842 - loss: 0.4448 - val_accuracy: 0.7806 - val_auc: 0.7710 - val_loss: 0.4546
Epoch 3/10
500/500  6s 12ms/step - accuracy: 0.8003 - auc: 0.8143 - loss: 0.4209 - val_accuracy: 0.7776 - val_auc: 0.7681 - val_loss: 0.4631
Epoch 4/10
500/500  6s 12ms/step - accuracy: 0.8208 - auc: 0.8536 - loss: 0.3828 - val_accuracy: 0.7720 - val_auc: 0.7573 - val_loss: 0.4854

=== Deep Learning (Keras MLP) ===
              precision    recall  f1-score   support

           0       0.81      0.93      0.87     61467
           1       0.56      0.29      0.38     18533

    accuracy                           0.78     80000
   macro avg       0.69      0.61      0.62     80000
weighted avg       0.75      0.78      0.76     80000

Accuracy: 0.7828875
ROC-AUC: 0.7703426040412755""")
st.divider()


st.header("Bloco 14 — Interpretabilidade: termos associados ao target")
st.markdown("""
A Logistic Regression permite interpretar coeficientes:
- coeficientes maiores → termos mais associados ao **positivo (relief)**
- coeficientes menores → termos mais associados ao **negativo (sem relief)**
""")

st.code("""
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]

top_positive = coefs.argsort()[-20:]
top_negative = coefs.argsort()[:20]

print("Top termos associados a POSITIVO (relief):")
print([feature_names[i] for i in top_positive])

print("\\nTop termos associados a NEGATIVO (sem relief):")
print([feature_names[i] for i in top_negative])
""")

st.markdown("**Saída: Top termos**")
st.code(r"""Top termos associados a POSITIVO (relief):
['direct express', 'bluebird', 'bank america', 'penfe', 'rush', 'midland', 'credit management', 'spring', 'comenity', 'bpo', 'information service', 'consultant', 'mrs', 'mcm', 'elan', 'portfolio recovery', 'resource management', 'transunion', 'credence resource', 'credence']

Top termos associados a NEGATIVO (sem relief):
['cash app', 'cashapp', 'zelle', 'nelnet', 'vanilla', 'cooper', 'netspend', 'robinhood', 'loancare', 'ocwen', 'inc', 'fcra experian', 'transworld', 'hsbc', 'nationstar', 'westlake', 'edfinancial', 'freedom mortgage', 'aidvantage', 'avant']""")
st.divider()


st.header("Bloco 15 — Dores nas reclamações negativas por categoria de produto")
st.markdown("""
Para identificar as “dores” por categoria, filtramos apenas os casos negativos (`sentiment_final == 0`).
Em seguida, calculamos TF-IDF médio por categoria e extraímos os 15 termos mais relevantes.
""")

st.code(r"""
# Filtrar apenas negativos
df_neg = df_sample[df_sample["sentiment_final"] == 0].copy()

vectorizer_neg = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1,2),
    min_df=10,
    token_pattern=r"\b[a-zA-Z]{3,}\b"
)

results = []

for product in df_neg["Product Consolidated"].unique():
    df_temp = df_neg[df_neg["Product Consolidated"] == product]

    X_temp = vectorizer_neg.fit_transform(df_temp["text_processed"])
    mean_tfidf = X_temp.mean(axis=0).A1
    feature_names_temp = vectorizer_neg.get_feature_names_out()

    top_indices = mean_tfidf.argsort()[-15:]

    for idx in top_indices:
        results.append({
            "Product": product,
            "Term": feature_names_temp[idx],
            "Score": float(mean_tfidf[idx])
        })

df_terms = pd.DataFrame(results).sort_values(["Product","Score"], ascending=[True, False])
df_terms
""")

st.markdown("**Saída: top termos por produto (prints + tabela parcial)**")
st.code(r"""====== CREDIT CARDS ======
['balance', 'dispute', 'call', 'tell', 'time', 'receive', 'report', 'pay', 'bank', 'credit card', 'charge', 'payment', 'account', 'credit', 'card']

====== CREDIT REPORTING ======
['inaccurate', 'item', 'request', 'section', 'remove', 'dispute', 'payment', 'inquiry', 'reporting', 'consumer', 'credit report', 'information', 'account', 'report', 'credit']

====== DEBT COLLECTION ======
['owe', 'letter', 'request', 'send', 'receive', 'credit report', 'pay', 'information', 'call', 'company', 'collection', 'report', 'account', 'credit', 'debt']

====== MORTGAGES AND LOANS ======
['bank', 'report', 'send', 'month', 'call', 'company', 'receive', 'time', 'tell', 'credit', 'account', 'mortgage', 'pay', 'loan', 'payment']

====== RETAIL BANKING ======
['zelle', 'receive', 'transfer', 'deposit', 'send', 'cash app', 'tell', 'app', 'transaction', 'cash', 'check', 'fund', 'money', 'bank', 'account']""")

df_terms_preview = pd.DataFrame({
    "Product": [
        "credit cards","credit cards","credit cards","credit cards","credit cards",
        "credit cards","credit cards","credit cards","credit cards","credit cards",
        "credit cards","credit cards","credit cards","credit cards","credit cards",
        "credit reporting","credit reporting","credit reporting","credit reporting","credit reporting"
    ],
    "Term": [
        "card","credit","account","payment","charge",
        "credit card","bank","pay","report","receive",
        "time","tell","call","dispute","balance",
        "credit","report","account","information","credit report"
    ],
    "Score": [
        0.066575,0.064052,0.062020,0.048476,0.040904,
        0.033873,0.032930,0.031644,0.029885,0.028277,
        0.026721,0.026369,0.025630,0.025482,0.025264,
        0.073658,0.071903,0.068642,0.047688,0.043507
    ]
})
st.dataframe(df_terms_preview, use_container_width=True, hide_index=True)


st.divider()
