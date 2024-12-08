import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Função para carregar os modelos de Adult Mortality
@st.cache_resource
def load_models_mortality():
    regressao_linear = joblib.load('regressao_linearAM.pkl')
    arvore_decisao = joblib.load('arvore_decisaoAM.pkl')
    random_forest = joblib.load('random_forestAM.pkl')
    return regressao_linear, arvore_decisao, random_forest

# Função para carregar os modelos de Life Expectancy
@st.cache_resource
def load_models_life_expectancy():
    regressao_linear = joblib.load('regressao_linear.pkl')
    arvore_decisao = joblib.load('arvore_decisao.pkl')
    random_forest = joblib.load('random_forest.pkl')
    return regressao_linear, arvore_decisao, random_forest

# Função para realizar a previsão
def predict(model, input_data):
    return model.predict(input_data)

# Função para verificar as características de entrada
def check_input_features(input_data, model):
    expected_features = model.coef_.shape[0] if hasattr(model, 'coef_') else len(model.feature_importances_)
    input_features = input_data.shape[1]
    if input_features != expected_features:
        st.error(f"Erro: O modelo espera {expected_features} características, mas recebeu {input_features}.")
        return False
    return True

# Menu principal
st.title("Sistema de Apoio à Decisão")
menu = st.radio("Selecione uma opção", ["Adult Mortality", "Life Expectancy"])

if menu == "Adult Mortality":
    st.header("Previsão de Mortalidade Adulta")
    
    # Carregar modelos de Mortalidade Adulta
    regressao_linear, arvore_decisao, random_forest = load_models_mortality()
    
    # Selecionar modelo
    model_option = st.selectbox('Selecione o Modelo', ['Regressão Linear', 'Árvore de Decisão', 'Random Forest'])

    # Ajustar variáveis
    perc_hiv_aids = st.slider('Ajustar HIV/AIDS (%)', min_value=0, max_value=100, value=100, step=1)
    perc_thin_1_19 = st.slider('Ajustar valor de thinnss 1-19 (%)', min_value=0, max_value=100, value=100, step=1)
    perc_thin_5_9 = st.slider('Ajustar valor de thinnss 5-9 (%)', min_value=0, max_value=100, value=100, step=1)


    
    input_data = {
        ' HIV/AIDS': 2 * (perc_hiv_aids / 100),
        ' thinness 1-19 years' : 5 * (perc_thin_1_19 / 100),
        ' thinness 5-9 years' : 5 * (perc_thin_1_19 / 100),
    }

    # Preencher com valores padrão para as outras variáveis
    num_faltando = 17 - len(input_data)
    for i in range(num_faltando):
        input_data[f'Feature_{i+1}'] = 1.0
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    # Previsão
    prediction = None
    if model_option == 'Regressão Linear':
        if check_input_features(input_array, regressao_linear):
            prediction = predict(regressao_linear, input_array)
    elif model_option == 'Árvore de Decisão':
        if check_input_features(input_array, arvore_decisao):
            prediction = predict(arvore_decisao, input_array)
    else:
        if check_input_features(input_array, random_forest):
            prediction = predict(random_forest, input_array)
    
    if prediction is not None:
        st.write(f'Previsão de {model_option}: {prediction[0]:.2f}')
        fig, ax = plt.subplots()
        ax.bar(model_option, prediction[0], color='red' if prediction[0] < 50 else 'blue')
        ax.set_ylabel('Previsão de Mortalidade Adulta')
        st.pyplot(fig)

elif menu == "Life Expectancy":
    st.header("Previsão de Expectativa de Vida")
    
    # Carregar modelos de Expectativa de Vida
    regressao_linear, arvore_decisao, random_forest = load_models_life_expectancy()
    
    # Selecionar modelo
    model_option = st.selectbox('Selecione o Modelo', ['Regressão Linear', 'Árvore de Decisão', 'Random Forest'])

    # Ajustar variáveis

    perc_mortalidade = st.slider('Ajustar Adult Mortality (%)', min_value=0, max_value=100, value=100, step=1)
    perc_hiv_aids = st.slider('Ajustar HIV/AIDS (%)', min_value=0, max_value=100, value=100, step=1)
    perc_thin_1_19 = st.slider('Ajustar valor de thinnss 1-19 (%)', min_value=0, max_value=100, value=100, step=1)
    perc_thin_5_9 = st.slider('Ajustar valor de thinnss 5-9 (%)', min_value=0, max_value=100, value=100, step=1)

    input_data = {
        'Adult Mortality': 16821 * (perc_mortalidade / 100),
        ' HIV/AIDS': 198 * (perc_hiv_aids / 100),
        ' thinness 1-19 years' : 485 * (perc_thin_1_19 / 100),
        ' thinness 5-9 years' : 490 * (perc_thin_1_19 / 100),
    }

    # Preencher com valores padrão para as outras variáveis
    num_faltando = 17 - len(input_data)
    for i in range(num_faltando):
        input_data[f'Feature_{i+1}'] = 1.0
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    # Previsão
    prediction = None
    if model_option == 'Regressão Linear':
        if check_input_features(input_array, regressao_linear):
            prediction = predict(regressao_linear, input_array)
    elif model_option == 'Árvore de Decisão':
        if check_input_features(input_array, arvore_decisao):
            prediction = predict(arvore_decisao, input_array)
    else:
        if check_input_features(input_array, random_forest):
            prediction = predict(random_forest, input_array)
    
    if prediction is not None:
        st.write(f'Previsão de {model_option}: {prediction[0]:.2f}')
        fig, ax = plt.subplots()
        ax.bar(model_option, prediction[0], color='red' if prediction[0] < 50 else 'blue')
        ax.set_ylabel('Previsão de Expectativa de Vida')
        st.pyplot(fig)
