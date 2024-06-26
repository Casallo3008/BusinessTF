import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import requests
import joblib
import os
import requests
import pickle

import requests
from io import BytesIO

url = 'modelo_random_forest_TF.pkl'

try:
    # Descargar el archivo del modelo desde GitHub
    response = requests.get(url)
    response.raise_for_status()  # Esto levantará una excepción si hay un error en la solicitud

    # Cargar el modelo desde los bytes descargados
    modelo = joblib.load(BytesIO(response.content))
    print("Modelo cargado exitosamente.")

except requests.exceptions.RequestException as e:
    print(f"Error al descargar el modelo: {e}")

except joblib.externals.loky.process_executor._RemoteTraceback as e:
    print(f"Error al cargar el modelo con joblib: {e}")

except Exception as e:
    print(f"Ha ocurrido un error inesperado: {e}")


# Definir el título de la aplicación
st.title('Predicción de Cliente Permanece')

# Definir las características seleccionadas según el ranking
selected_features = ['Escala de Servicio', 'Edad']

# Agregar controles de entrada para las características seleccionadas
escala_servicio = st.slider('Escala de Servicio', min_value=0, max_value=10, value=5)
edad = st.slider('Edad', min_value=18, max_value=100, value=30)

# Crear un dataframe con las características de entrada
input_data = pd.DataFrame({'Escala de Servicio': [escala_servicio], 'Edad': [edad]})

# Seleccionar solo las características especificadas en selected_features
input_data = input_data[selected_features]

# Agregar un botón para hacer la predicción cuando se presiona
if st.button('Predecir'):
    prediction = modelo.predict(input_data)
    proba = modelo.predict_proba(input_data)[:, 1]
    st.write(f'La predicción es: {"Permanece" if prediction[0] == 1 else "No Permanece"}')
    st.write(f'Probabilidad de permanecer: {proba[0]:.2f}')

