import streamlit as st
import h2o
import pandas as pd
import numpy as np

# Iniciar o H2O
h2o.init()

# Carregar o modelo salvo
model_path = "/Users/altonfreitas/Documents/LSIG/Metodos_Tecnicas_Analise_Dados/Def" 
model = h2o.load_model(model_path)

# Função para realizar a previsão
def make_prediction():
    # Capturar os inputs do usuário
    input_data = {
        "Adult Mortality": st.session_state.adult_mortality,
        "Alcohol": st.session_state.alcohol,
        "Hepatitis B": st.session_state.hepatitis_b,
        " BMI ": st.session_state.bmi,
        " HIV/AIDS": st.session_state.hiv_aids,
        "GDP": st.session_state.gdp,
        "Income composition of resources": st.session_state.income_composition,
        "Schooling": st.session_state.schooling,
    }

    # Criar um DataFrame com os inputs
    input_df = pd.DataFrame([input_data])
    input_h2o = h2o.H2OFrame(input_df)

    # Realizar a predição
    prediction = model.predict(input_h2o).as_data_frame()
    st.session_state["prediction_label"] = prediction.iloc[0, 0]

# Título da aplicação
st.title("Previsão de Expectativa de Vida :sunglasses:")

# Sliders para as variáveis de entrada
st.slider("Adult Mortality", 0.0, 500.0, 150.0, key="adult_mortality", on_change=make_prediction)
st.slider("Alcohol", 0.0, 20.0, 5.0, key="alcohol", on_change=make_prediction)
st.slider("Hepatitis B (Cobertura Vacinal)", 0.0, 100.0, 80.0, key="hepatitis_b", on_change=make_prediction)
st.slider("BMI", 0.0, 50.0, 25.0, key="bmi", on_change=make_prediction)
st.slider("HIV/AIDS", 0.0, 10.0, 1.0, key="hiv_aids", on_change=make_prediction)
st.slider("GDP (Produto Interno Bruto)", 0.0, 100000.0, 5000.0, key="gdp", on_change=make_prediction)
st.slider("Income Composition of Resources", 0.0, 1.0, 0.5, key="income_composition", on_change=make_prediction)
st.slider("Schooling (Anos de Escolaridade)", 0.0, 20.0, 10.0, key="schooling", on_change=make_prediction)

# Exibir a previsão
if "prediction_label" in st.session_state:
    st.write(f"**Expectativa de Vida Prevista:** :red[**{st.session_state['prediction_label']}**] anos")
