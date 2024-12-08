import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carregar os modelos salvos
@st.cache_resource
def load_models():
    regressao_linear = joblib.load('regressao_linear.pkl')
    arvore_decisao = joblib.load('arvore_decisao.pkl')
    random_forest = joblib.load('random_forest.pkl')
    return regressao_linear, arvore_decisao, random_forest

# Função para previsão
def predict(model, input_data):
    return model.predict(input_data)

# Função para verificar as colunas do modelo
def check_input_features(input_data, model):
    # Verificar o número de características do modelo
    expected_features = model.coef_.shape[0] if hasattr(model, 'coef_') else len(model.feature_importances_)  # Para Regressão Linear ou RandomForest
    input_features = input_data.shape[1]
    
    if input_features != expected_features:
        st.error(f"Erro: O modelo espera {expected_features} características, mas recebeu {input_features}.")
        return False
    return True

# Carregar os modelos
regressao_linear, arvore_decisao, random_forest = load_models()

# Definir as variáveis de entrada
st.title('Sistema de Apoio à Decisão')

st.write("Escolha os valores para as variáveis ajustados por percentagem e preveja a Mortalidade Adulta ou Expectativa de Vida.")

# Exemplo de variáveis independentes
model_option = st.selectbox('Selecione o Modelo', ['Regressão Linear', 'Árvore de Decisão', 'Random Forest'])

# Exibir os campos de entrada com base na seleção do modelo
input_data = {}
if model_option in ['Regressão Linear', 'Árvore de Decisão', 'Random Forest']:
    # Percentagem de ajuste para as variáveis
    perc_mortalidade = st.slider('Ajustar Adult Mortality (%)', min_value=0, max_value=100, value=100, step=1)
    perc_hiv_aids = st.slider('Ajustar HIV/AIDS (%)', min_value=0, max_value=100, value=100, step=1)



    # Valores originais das variáveis
    valor_mortalidade_original = 169 
    valor_hiv_aids_original = 2
    valor_thinness1_19_original = 5
    valor_IoR_original = 2  

    # Ajustar os valores conforme a percentagem
    input_data['Adult Mortality'] = valor_mortalidade_original * (perc_mortalidade / 100)
    input_data[' HIV/AIDS'] = valor_hiv_aids_original * (perc_hiv_aids / 100)
    
    # Preencher as outras variáveis com valores padrão (podem ser ajustados conforme necessário)
    num_faltando = 17 - len(input_data)
    for i in range(num_faltando):
        input_data[f'Feature_{i+1}'] = 1.0  # Ajuste conforme os valores padrão adequados para o seu modelo

# Certifique-se de que input_array tem a forma (1, n_features) para um único exemplo
input_array = np.array(list(input_data.values())).reshape(1, -1)

# Atualizar a previsão automaticamente
prediction = None  # Inicializar a variável prediction

# Verificar a entrada para garantir que tem o número correto de características
if model_option == 'Regressão Linear':
    if check_input_features(input_array, regressao_linear):
        prediction = predict(regressao_linear, input_array)
elif model_option == 'Árvore de Decisão':
    if check_input_features(input_array, arvore_decisao):
        prediction = predict(arvore_decisao, input_array)
else:
    if check_input_features(input_array, random_forest):
        prediction = predict(random_forest, input_array)

# Gerar o gráfico automaticamente quando o valor for ajustado
if prediction is not None:
    st.write(f'Previsão de {model_option}: {prediction[0]:.2f}')
    
    # Definir a cor do gráfico com base na previsão
    half_value = 50  # Valor de referência para "metade"
    if prediction[0] < half_value:
        color = 'red'  # Cor vermelha se a previsão for menor que a metade
    else:
        color = 'blue'  # Cor azul se a previsão for maior que a metade
    
    # Gerar gráfico de barras
    fig, ax = plt.subplots()
    ax.bar(model_option, prediction[0], color=color)
    ax.set_ylabel('Previsão de Expectativa de Vida')
    ax.set_title(f'Previsão para o modelo: {model_option}')
    
    # Exibir o gráfico da previsão
    st.pyplot(fig)

else:
    st.warning("Não foi possível realizar a previsão. Verifique os valores de entrada.")
