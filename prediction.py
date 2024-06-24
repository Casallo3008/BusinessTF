import joblib

def predict(data):
    modelo = joblib.load('modelo_random_forest_TF.pkl')
    return modelo.predict(data)
