import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Função para carregar o modelo salvo
@st.cache_resource
def load_model():
    model = joblib.load('modelo_h2o.pkl')  # Certifique-se de que o caminho está correto
    return model

# Função para realizar a previsão
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Inicializar a interface do Streamlit
st.title('Sistema de Apoio à Decisão')
st.write("Ajuste os valores e preveja a **Life Expectancy** com base no modelo Random Forest.")

# Carregar o modelo Random Forest
random_forest = load_model()

# Interface para ajustar as variáveis
perc_life_exp = st.slider('Ajustar Life expectancy (%)', 0, 200, 100)
perc_hiv_aids = st.slider('Ajustar HIV/AIDS (%)', 0, 200, 100)
perc_schooling = st.slider('Ajustar Schooling (%)', 0, 200, 100)
perc_bmi = st.slider('Ajustar BMI (%)', 0, 200, 100)

# Valores originais das variáveis
valor_life_exp_original = 69
valor_hiv_aids_original = 2
valor_schooling_original = 12
valor_bmi_original = 38

# Ajustar os valores com base na percentagem
input_data = np.array([
    valor_life_exp_original * (perc_life_exp / 100),
    valor_hiv_aids_original * (perc_hiv_aids / 100),
    valor_schooling_original * (perc_schooling / 100),
    valor_bmi_original * (perc_bmi / 100),
]).reshape(1, -1)

# Verificar o número de características esperado pelo modelo
n_features_model = random_forest.n_features_in_
if input_data.shape[1] != n_features_model:
    st.error(f"O modelo espera {n_features_model} características, mas recebeu {input_data.shape[1]}.")
else:
    # Realizar a previsão
    prediction = predict(random_forest, input_data)
    st.write(f"**Previsão de Life Expectancy:** {prediction[0]:.2f}")

    # Gerar gráfico de barras
    fig, ax = plt.subplots()
    ax.bar('Life Expectancy', prediction[0], color='blue')
    ax.set_ylabel('Valor Previsto')
    ax.set_title('Previsão com Random Forest')
    st.pyplot(fig)
