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
@st.cache_resource(show_spinner=False)
def get_model():
    return SentenceTransformer('distiluse-base-multilingual-cased-v1')

@st.cache_data(show_spinner="Buscando recomendações...")
def get_recommendations(df, cnae_cliente, nome_cliente):
    model = get_model()
    nome_todos = nome_cliente + df.loc[df['CNAE FISCAL PRINCIPAL']==cnae_cliente, 'NOME'].values.tolist()
    client_embeddings  = model.encode(nome_todos)
    df_rec             = df[(df['CNAE FISCAL PRINCIPAL']==cnae_cliente) & (df['CNPJ']!=cnpj_cliente)].copy()
    df_rec['SIMILARIDADE'] = cosine_similarity(client_embeddings)[1:,0]
    df_rec = df_rec.sort_values(by='SIMILARIDADE', ascending=False)
    return df_rec

@st.cache_data
def read_data():
    return pd.read_csv('pre-processed-data-100k.csv'), pd.read_csv('preprocessed-clientes.csv')

#-------------------------------- DATA LOAD
df, df_clientes = read_data()

#-------------------------------- SIDEBAR
with st.sidebar:
    st.write('# Simulação')
    top_k = st.selectbox("Qtd máxima de recomendações:", options=[10, 20, 30, 50, 100, 200], index=0)
    cnpj_cliente = st.selectbox("CNPJ Base:", df_clientes['CNPJ'].sort_values().unique(), index=74) 
    df_cliente = df_clientes[df_clientes['CNPJ']==cnpj_cliente].copy()
    cnae_cliente = df_cliente['CNAE FISCAL PRINCIPAL'].values[0]
    nome_cliente = df_clientes.loc[df_clientes['CNPJ']==cnpj_cliente,'NOME'].values.tolist()

st.write("# Info Cliente:")
st.write(f"**CNPJ Base:** {cnpj_cliente}")
st.write(f"**Nome Fantasia:** {df_cliente['NOME FANTASIA'].values[0]}")
st.write(f"**Razão Social:** {df_cliente['RAZÃO SOCIAL'].values[0]}")
st.write(f"**CNAE principal:** {cnae_cliente}")
st.write(f"**CNAE secundária:** {df_cliente['CNAE FISCAL SECUNDÁRIA'].values[0]}")

st.divider()

st.write("# Recomendações:")

df_rec = get_recommendations(df, cnae_cliente, nome_cliente)

st.data_editor(
    df_rec.drop(columns='NOME').head(top_k).reset_index(drop=True),
    column_config={
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