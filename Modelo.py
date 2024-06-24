import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import requests
import joblib

output_path = 'modelo_random_forest_TF.pkl'

try:
    # Intentar cargar el modelo usando joblib
    with open(output_path, 'rb') as f:
        modelo = joblib.load(f)
    print("Modelo cargado correctamente.")
    # Aquí puedes continuar con el uso del modelo cargado

except FileNotFoundError:
    print(f"El archivo {output_path} no se encontró.")

except ImportError:
    print(f"No se pudo importar el módulo necesario para cargar el modelo desde {output_path}.")

except KeyError as e:
    print(f"Error de clave al cargar el modelo desde {output_path}: {e}")

except Exception as e:
    print(f"Error desconocido al cargar el modelo desde {output_path}: {str(e)}")





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
