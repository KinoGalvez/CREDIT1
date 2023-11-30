import streamlit as st
import pandas as pd
from PIL import Image
import streamlit.components.v1 as c
import os
import pickle


directorio_actual = os.getcwd()
with open(os.path.join(directorio_actual, '..', 'models','final_model_gbc_roc_auc271123.pkl'), 'rb') as file:
    modelo_GBC = pickle.load(file)

df_test = pd.read_csv(os.path.join(directorio_actual, '..', 'data','test', 'test_balmix_cat_mean.csv'))

st.set_page_config(page_title='CREDITO',
                   page_icon="💵")

st.title('DAMOS CREDITO?')

imagen = Image.open('img/logo_lending_club.png')

st.image(imagen)

predecir = st.sidebar.button('Damos crédito?')

if predecir:
    registro_aleatorio = df_test.sample(n=1)
    prediccion = modelo_GBC.predict(registro_aleatorio)
    st.title(f'La predicción es :{prediccion}')
