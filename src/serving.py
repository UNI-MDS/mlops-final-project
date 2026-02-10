from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# 1. Inicializar la App
app = FastAPI(title="API de Predicción de Churn", description="Servicio para predecir si un cliente abandonará la empresa")

# 2. Cargar el modelo entrenado
MODEL_PATH = "models/churn_model.pkl"
model = joblib.load(MODEL_PATH)

# 3. Definir el esquema de datos de entrada (Pydantic)
# Nota: Aquí deberías incluir las columnas que quedaron después del One-Hot Encoding
# Para simplificar, aceptaremos un diccionario genérico o una lista.
class CustomerData(BaseModel):
    data: dict

@app.get("/")
def home():
    return {"message": "API de Churn funcionando correctamente. Ve a /docs para probar."}

@app.post("/predict")
def predict(input_data: CustomerData):
    # Convertir JSON de entrada a DataFrame
    df_input = pd.DataFrame([input_data.data])
    
    # Realizar predicción
    prediction = model.predict(df_input)
    probability = model.predict_proba(df_input)
    
    result = "Abandona (Churn)" if prediction[0] == 1 else "Se queda (No Churn)"
    
    return {
        "prediction": int(prediction[0]),
        "label": result,
        "probability": float(probability[0][prediction[0]])
    }