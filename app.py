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

df_dados_input = pd.read_csv('DatabaseDengue.csv',header=0)
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

    # Realiza a previsão
    pred = modelo.predict(entrada)[indice]
    st.success(f"A previsão para **{cidade_selecionada}** é: {'Surto' if pred == 1 else 'Sem Surto'}")
    st.set_page_config(layout="wide")
    col1, col2 = st.columns(2)
    with col1:
    ####  Explicação dos resultados ####
    ### Shap
        st.subheader("Explicação com SHAP")
        explainer = shap.Explainer(modelo)
        shap_values = explainer(df_dados_processados)  # seu dataframe X com as features

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[indice],show=False)
        st.pyplot(fig)

    ### Lime
    with col2:
        st.subheader("Explicação com LIME")
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





