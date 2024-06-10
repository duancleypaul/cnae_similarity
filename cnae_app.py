import pandas    as pd
import streamlit as st
from sentence_transformers    import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#-------------------------------- LAYOUT CONFIG
st.set_page_config(
    page_title="Simulação CNAE",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Developed by Duan Cleypaul."}
)

#-------------------------------- CACHED FUNCTIONS
@st.cache_data(show_spinner=False)
def get_model():
    return SentenceTransformer('distiluse-base-multilingual-cased-v1', truncate_dim=128)

@st.cache_data(show_spinner=False)
def get_recommendations(df, cnae_cliente, razao_social_cliente):
    model = get_model()
    razao_social_todos = razao_social_cliente + df.loc[df['CNAE FISCAL PRINCIPAL']==cnae_cliente, 'RAZÃO SOCIAL'].values.tolist()
    client_embeddings  = model.encode(razao_social_todos)
    df_rec             = df[(df['CNAE FISCAL PRINCIPAL']==cnae_cliente) & (df['CNPJ']!=cnpj_cliente)].copy()
    df_rec['SIMILARIDADE'] = cosine_similarity(client_embeddings)[1:,0]
    df_rec = df_rec.sort_values(by='SIMILARIDADE', ascending=False)
    return df_rec

#-------------------------------- DATA LOAD
df = pd.read_csv('pre-processed-data-730k.csv')
df_clientes = pd.read_csv('clientes.csv')

#-------------------------------- SIDEBAR
with st.sidebar:
    st.write('# Simulação')
    # top_k = st.number_input("Qtd máxima de recomendações:", min_value=1, max_value=15, value=5, key="top-k")
    top_k = st.selectbox("Qtd máxima de recomendações:", options=[10, 20, 30, 50, 100, 200], index=0)
    cnpj_cliente = st.selectbox("CNPJ Base:", df_clientes['CNPJ'].unique())
    df_cliente = df_clientes[df_clientes['CNPJ']==cnpj_cliente].copy()
    cnae_cliente = df_cliente['CNAE FISCAL PRINCIPAL'].values[0]
    razao_social_cliente = df_clientes.loc[df_clientes['CNPJ']==cnpj_cliente,'RAZÃO SOCIAL'].values.tolist()

st.write("# Info Cliente:")
st.write(f"**CNPJ Base:** {cnpj_cliente}")
st.write(f"**Razão Social:** {df_cliente['RAZÃO SOCIAL'].values[0]}")
st.write(f"**CNAE principal:** {cnae_cliente}")
st.write(f"**CNAE secundária:** {df_cliente['CNAE FISCAL SECUNDÁRIA'].values[0]}")

st.divider()

st.write("# Recomendações:")

# with st.spinner('Buscando recomendações...'):
df_rec = get_recommendations(df, cnae_cliente, razao_social_cliente)

st.data_editor(
    df_rec.head(top_k).reset_index(drop=True),
    column_config={
        "ID": st.column_config.NumberColumn(
            format="%s",
        ),
        "CNPJ": st.column_config.NumberColumn(
            help="8 primeiros dígitos do CNPJ",
            format="%s",
        ),
        "CNAE FISCAL PRINCIPAL": st.column_config.NumberColumn(
            format="%s",
        ),
        "NAT. JURIDICA": st.column_config.NumberColumn(
            format="%s",
        ),
        "INICIO DA ATIVIDADE": st.column_config.NumberColumn(
            format="%s",
        ),
        "MUNICÍPIO": st.column_config.NumberColumn(
            format="%s",
        ),
    },
)