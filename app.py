import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import lime
import lime.lime_tabular
 

# Importando as bibliotecas necessárias
from sklearn.pipeline import Pipeline  # Para criar pipelines de transformação
from sklearn.compose import ColumnTransformer  # Para aplicar transformações diferentes a grupos de colunas

from sklearn.preprocessing import StandardScaler  # Para padronizar variáveis numéricas
from sklearn.preprocessing import OneHotEncoder  # Para codificar variáveis categóricas
from sklearn.impute import SimpleImputer  # Para lidar com valores faltantes
from xgboost import XGBClassifier

df_dados_input = pd.read_csv('DataBaseDengue.csv',header=0)
df_dados_processados = pd.read_csv('DataBaseDengueProcessada.csv',header=0)
df_dados_processados.drop('Class', axis=1, inplace=True)
# Carregar modelo
modelo = joblib.load('modelo_dengue.pkl')

st.title("Previsão de Surto de Dengue")


# Inputs do usuário — adapte aos seus features!
# Inputs na sidebar
st.sidebar.header("Selecione a cidade")

# Seleciona cidade
lista_cidades = df_dados_input["MunicipioResidencia"].sort_values().unique()
cidade_selecionada = st.sidebar.selectbox("Escolha uma cidade", lista_cidades)

# Filtrando o DataFrame para encontrar linhas com o valor desejado
df_filtrado = df_dados_input[df_dados_input['MunicipioResidencia'] == cidade_selecionada]
# Obtendo o índice da primeira ocorrência
indice = df_filtrado.index[0]


# Quando o botão for clicado
if st.sidebar.button("Prever"):
    # Filtra os dados da cidade selecionada
    entrada = df_dados_processados
    # EXIBE OS DADOS DA CIDADE AQUI
    # Mostra título da seção
    st.subheader(f"📊 Indicadores da cidade de {cidade_selecionada}")
    col1, col2,col3,ColCategorica = st.columns(4)

    with col1:
        st.metric("População Total", f"{df_filtrado['POP_TOT'].values[0]:}")##.replace(",", "."))
        st.metric("PIB per capita", f"R$ {df_filtrado['PIB_PER_CAPITA'].values[0]}")##.replace(",", "X").replace(".", ",").replace("X", "."))
        st.metric("Índice de desenvolvimento humano do município (IDH-M)", f"{df_filtrado['IDH-M'].values[0]}")
        st.metric("Índice de GINI da renda domiciliar per capita - GINI", f"{df_filtrado['GINI'].values[0]}")

    with col2:
        st.metric("Índice de pavimentação das vias públicas (%)", f"{df_filtrado['Índice de pavimentação das vias públicas'].values[0]}")
        st.metric("IN055_AE - Índice da população total com atendimento de água (%)", f"{df_filtrado['IN055_AE'].values[0]}")
        st.metric("IN056_AE - Índice da população total com atendimento de esgoto(%)", f"{df_filtrado['IN056_AE'].values[0]}")
        st.metric("IN015_RS - Taxa da população coberta com serviço de coleta de resíduos(%)", f"{df_filtrado['IN015_RS'].values[0]}")
    with col3:
        st.metric("IN024_AE - Índice da população urbana com atendimento de esgoto (%)", f"{df_filtrado['IN024_AE'].values[0]}")
        st.metric("IN015_AE - Índice de volume de esgoto coletado (%)", f"{df_filtrado['IN015_AE'].values[0]}")
        st.metric("IN022_AE - Consumo médio per capita de água", f"{df_filtrado['IN022_AE'].values[0]}")
        st.metric("IN049_AE - Índice de perdas na distribuição de água(%)", f"{df_filtrado['IN049_AE'].values[0]}")
        st.metric("IN016_AE - Indice de volume de esgoto tratado (%)", f"{df_filtrado['IN016_AE'].values[0]}")
    with ColCategorica:
        st.write(f"📊 Indicadores categóricos da cidade de {cidade_selecionada}")
        status = "✅ Sim" if df_filtrado['CS001'].values[0] == "Sim" else "❌ Não"
        st.write(f"📍Coleta seletiva de resíduos no município: {status}")
        status = "✅ Sim" if df_filtrado['Msau28'].values[0] == "Sim" else "❌ Não"
        st.write(f"📍 Programa de Agentes Comunitários de Saúde - existência: {status}")
        status = "✅ Sim" if df_filtrado['Mgrd06'].values[0] == "Sim" else "❌ Não"
        st.write(f"📍 O município foi atingido por alagamentos nos últimos 4 anos: {status}")
        status = "✅ Sim" if df_filtrado['Mgrd08'].values[0] == "Sim" else "❌ Não"
        st.write(f"📍O município foi atingido por enchentes ou inundações graduais nos últimos 4 anos: {status}")
        status = "✅ Sim" if df_filtrado['Mgrd11'].values[0] == "Sim" else "❌ Não"
        st.write(f"📍O município foi atingido por enxurradas ou inundações bruscas nos últimos 4 anos: {status}")        


    # Realiza a previsão
    pred = modelo.predict(entrada)[indice]
    st.success(f"A previsão para **{cidade_selecionada}** é: {'Surto' if pred == 1 else 'Sem Surto'}")
    st.set_page_config(layout="wide")
    col4, col5 = st.columns(2)
    with col4:
    ####  Explicação dos resultados ####
    ### Shap
        st.subheader(" Explicação com SHAP")
        explainer = shap.Explainer(modelo)
        shap_values = explainer(df_dados_processados)  # seu dataframe X com as features

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[indice],show=False)
        st.pyplot(fig)

    ### Lime
    with col5:
        st.subheader(" Explicação com LIME")
        # Criar o explicador    
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=df_dados_processados.values,
            feature_names=df_dados_processados.columns.tolist(),
            class_names=['Sem Surto', 'Com Surto'],
            mode='classification'
        )

        i = indice
        exp = explainer.explain_instance(
            data_row=df_dados_processados.iloc[i].values,
            predict_fn=modelo.predict_proba,  # se for XGBClassifier do sklearn
            num_features=10
        )

        # Visualizar no notebook ou em HTML

        st.components.v1.html(exp.as_html(), height=400, scrolling=True)





