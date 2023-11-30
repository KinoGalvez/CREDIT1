

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



imagen = Image.open('img/logo_lending_club.png')

st.image(imagen)





def main():
    st.title('DAMOS CREDITO?')

    st.sidebar.header('Datos cliente:')

    def user_input_parameters():
        term = st.sidebar.slider('term', 0.27, 0.56)
        int_rate = st.sidebar.slider('int_rate', 6.00, 24.89)
        installment = st.sidebar.slider('installment', 21.62, 1380.63)
        emp_length = st.sidebar.slider('emp_length', 0.45, 0.53)
        home_ownership = st.sidebar.slider('home_ownership', 0.38, 0.52)
        annual_inc = st.sidebar.slider('annual_inc', 4800, 80000)
        purpose = st.sidebar.slider('purpose', 0.39, 0.63)
        dti = st.sidebar.slider('dti', 0.00, 35.00)
        pub_rec = st.sidebar.slider('pub_rec', 0.00, 0.50)
        revol_bal = st.sidebar.slider('revol_bal', 0.00, 19000.00)
        revol_util = st.sidebar.slider('revol_util', 0.00, 78.70)
        total_acc = st.sidebar.slider('total_acc', 0.36, 0.51)
        initial_list_status = st.sidebar.slider('initial_list_status', 0.49, 0.52)
        mor_acc = st.sidebar.slider('mor_acc', 0.33, 0.50)
        pub_rec_bankruptcies = st.sidebar.slider('pub_rec_bankruptcies', 0.00, 0.50)
        
 
        data = {'term' : term, 'int_rate' : int_rate, 'installment' : installment,
                 'emp_length' : emp_length , 'home_ownership' : home_ownership,'annual_inc' : annual_inc, 
                 'purpose' : purpose, 'dti' : dti, 'pub_rec' : pub_rec, 'revol_bal':revol_bal, 'revol_util' : revol_util, 
                 'total_acc' : total_acc ,'initial_list_status': initial_list_status,'mor_acc': mor_acc,'pub_rec_bankruptcies': pub_rec_bankruptcies}
        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_parameters()

  
    
    st.subheader('Parametros seleccionados')
    st.subheader(modelo_GBC)
    st.write(df)
    

    if st.button('DAMOS CRÉDITO'):
    
        prediccion = modelo_GBC.predict(df)
    
        
        resultado = "Préstamo Aprobado" if prediccion[0] == 1 else "Préstamo Denegado"

    
        st.success(resultado)

if __name__ == '__main__':
    main()

