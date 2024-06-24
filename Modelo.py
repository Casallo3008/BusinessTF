import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path

# URL del archivo en GitHub
url = 'https://raw.githubusercontent.com/Casallo3008/BusinessTF/main/modelo_random_forest_TF.pkl'


# Ruta local donde se guardará el archivo descargado
output_path = 'modelo_random_forest_TF.pkl'

# Descargar el archivo desde la URL
response = requests.get(url)
if response.status_code == 200:
    # Guardar el contenido descargado en un archivo local
    with open(output_path, 'wb') as f:
        f.write(response.content)
else:
    print(f"No se pudo descargar el archivo desde {url}. Status code: {response.status_code}")
    exit(1)

# Intentar cargar el modelo usando joblib
modelo = joblib.load(output_path)

# Cargar el modelo desde el archivo descargado
modelo = joblib.load(output_path)
print("Modelo cargado correctamente.")

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



# Opcional: eliminar el archivo después de usarlo
Path(output_path).unlink()
