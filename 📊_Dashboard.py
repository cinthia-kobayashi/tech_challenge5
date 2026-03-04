import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="POSTECH - FIAP | Dashboard",
    page_icon="📈",
    layout="wide"
)

with st.sidebar:
    st.markdown('''**Grupo:**  
Agnes Miki Magario  
Cinthia Mayumi Kobayashi  
Lina Satie Kobata Felippe
''')

st.title("Dashboard — TechChallenge 5")
st.caption("Resultado das avaliações do banco de dados..")
st.divider()

st.markdown("""
Esta página tem como objetivo mostrar o resultado da análise do banco de dados de reclamações "Consumer Complaint Database API docs".  
Aqui mostraremos e analisaremos os outputs gerados pelo [pipeline de dados](/Pipeline) para tratamento e treinamento de um modelo para interpretação das reclamações.
""")
st.markdown("**Os dados utilizados nesse trabalho foram baixados em 18/02/2026**")
st.info(
    """
    O código completo e os arquivos utilizados estão no [GitHub!](https://github.com/cinthia-kobayashi/tech_challenge5)
    """,
    icon="🔗",
)
st.divider()

st.subheader("1 - Distribuição de produtos na base")
st.markdown("""
Percebe-se que o produto que domina as reclamações é o Relatório ou Cadastro de crédito (credit reporting), que engloba reclamações erros no histórico, negativação indevida, erros de score, entre outros. Em segundo lugar vem as reclamações sobre cobranças de dívidas atrasadas (debt collection). 
Do terceiro ao quinto percebe-se uma semelhança percentual entre as reclamações de serviços bancários no geral (retail banking), financiamentos (mortgages and loans) e cartões de crédito (credit cards), nessa ordem respectivamente.
""")


data_dist = [
    {"Product Consolidated": "credit reporting",       "Count": 2497507},
    {"Product Consolidated": "debt collection",        "Count": 408350},
    {"Product Consolidated": "retail banking",         "Count": 295820},
    {"Product Consolidated": "mortgages and loans",    "Count": 288388},
    {"Product Consolidated": "credit cards",           "Count": 227128},
]

df_dist = pd.DataFrame(data_dist)
total = df_dist["Count"].sum()
df_dist["Percent"] = (df_dist["Count"] / total) * 100
df_dist = df_dist.sort_values("Percent", ascending=False)
col1, col2 = st.columns(2)
col2.dataframe(
    df_dist[["Product Consolidated","Percent"]].assign(Percent=df_dist["Percent"].round(2)),
    use_container_width=True,
    hide_index=True
)

plt.figure(figsize=(10, 5))
plt.barh(df_dist["Product Consolidated"], df_dist["Percent"])
plt.xlabel("Percentual (%)")
plt.title("Reclamações por Categoria")
plt.gca().invert_yaxis()
plt.tight_layout()

col1.pyplot(plt)

st.divider()



st.subheader("2 - Distribuição do sentimento")
st.markdown("""
Ao analisarmos a base, por ser uma base de reclamação, mesmo quando o grupo fez **manualmente** a análise de sentimentos de uma amostra de 1500 reclamações, todas foram negativas.  
Assim, decidimos usar como base para categorizar como positivos as reclamções que tiveram como solução alguma compensação financeira (relief), ficando:

- **Reclamação negativa** (sem relief)
- **Reclamação positiva** (com algum tipo de relief ao consumidor)

Com esse parâmetro obtemos a distribuição entre positivos e negativos na base de acordo com o gráfico abaixo.
""")
st.markdown("")
col1, col2 = st.columns(2)
data_sentiment = pd.DataFrame({
    "Sentimento": ["Negativo (sem relief)", "Positivo (com relief)"],
    "Percentual": [76.83, 23.17]
})

col2.dataframe(data_sentiment, hide_index=True, use_container_width=True)


plt.figure(figsize=(10, 5))
plt.bar(data_sentiment["Sentimento"], data_sentiment["Percentual"])
plt.ylabel("Percentual (%)")
plt.title("Distribuição do sentimento na base")

col1.pyplot(plt)




st.divider()

st.subheader("3 -  Termos mais relevantes por produto")
col1, col2, col3 = st.columns([10,2,8])
col1.markdown("""
Pela análise dos termos mais relevantes nas reclamações negativas, segmentado por categoria de produto, evidenciamos padrões consistentes de insatisfação do consumidor.    
                          
- Em *credit reporting*, os termos como “credit report”, “information” e “inaccurate” nos diz que o foco dos erros no tem impacto direto no histórico do consumidor e há uma grande solicitação de correção, o que indica negativação indevida entre outros erros.
            
- Na categoria *debt collection*, percebe-se pelos termos uma grande quantidade de correspondências (letters received) com cobranças indevidas, gerando essa grande quantidade de reclamações.

- Na parte de serviços financeiros (*retail banking*), percebe-se que os serviços que tem maior ocorrência são “account”, “money”, “fund” e “check”, sinalizando problemas de movimentação e disponibilidade de saldo, reforçando que as principais dores se concentram em atividades financeiras críticas (cobrança, registro de crédito e acesso a recursos).

- Em mortgages and loans (empréstimos, financiamentos e hipotecas), os termos mais recorrentes (**payment, pay, account*) destacam falhas no processamento de parcelas e inconsistências contratuais.
            
- Por fim, em credit cards, predominam termos associados a cobrança e pagamentos (“charge”, “payment”, “account”), sugerindo recorrência de problemas como cobranças indevidas, contestação de transações e divergências em faturas. 
""")

df_terms = pd.read_csv("outputs/df_terms.csv")
col3.markdown("### Tabela de score de termos")
col3.dataframe(df_terms, use_container_width=True)


st.divider()

st.subheader("4 - Gráficos de termos por produtos")
st.markdown("""
Abaixo, exibimos em mais detalhes os gráficos que corroboram a análise do item 3.
""")

products = [
    "credit_cards",
    "credit reporting",
    "debt collection",
    "mortgages and loans",
    "retail banking"
]
cols = st.columns(2)
i = 0
for p in products:
    filename = f"outputs/top_terms_{p.replace(' ', '_')}.png"
    with cols[i % 2]:
        st.markdown(f"**{p.title()}**")
        try:
            st.image(filename)
        except Exception:
            st.info(f"Imagem não encontrada: `{filename}`")
    i += 1

st.divider()

st.subheader("4 - Nuvem de palavras por produto")
st.markdown("""
Exibe wordclouds salvos no pipeline (ex.: `outputs/wordcloud_<product>.png`).
""")

cols = st.columns(2)
i = 0
for p in products:
    filename = f"outputs/wordcloud_{p.replace(' ', '_')}.png"
    with cols[i % 2]:
        st.markdown(f"**{p.title()}**")
        try:
            st.image(filename)
        except Exception:
            st.info(f"Imagem não encontrada: `{filename}`")
    i += 1

st.divider()

st.subheader("5 -  Métricas do modelo de DeepLearning")

st.markdown("""
O modelo de Deep Learning (MLP) apresentou acurácia de **78,3%** e ROC-AUC de **0,77**, o que indica uma boa capacidade de discriminação entre reclamações positivas e negativas.
Observamos, devido à natureza dos dados, que o modelo apresenta melhor desempenho na classe **negativa**, com **recall de 0,93**, enquanto a classe **positiva** possui recall de **0,29**, indicando uma maior dificuldade na identificação de casos em que houve resolução favorável ao consumidor.
Comportamento esperado, como falado acima, devido ao desbalanceamento do *dataset*, onde aproximadamente **77% das observações pertencem à classe negativa** e só tem esse patamar devido a imposição do grupo por outra variável. A base de dados é, em si, toda negativa.
""")


with open("outputs/metrics_deep_learning.txt", "r") as f:
        st.code(f.read())