import pandas as pd
import os

def prepare_data(input_path, output_path):
    """Lee el raw data, lo limpia y lo guarda en la carpeta training."""
    # Leer
    df = pd.read_csv(input_path)
    
    # Limpieza básica
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df.drop('customerID', axis=1, inplace=True)
    
    # Encoding
    df_final = pd.get_dummies(df)
    
    # Crear carpeta de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar resultado
    df_final.to_csv(output_path, index=False)
    print(f"Archivo procesado guardado en: {output_path}")

if __name__ == "__main__":
    # Definimos las rutas siguiendo la estructura del proyecto
    RAW_DATA = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    PROCESSED_DATA = "data/training/churn_processed.csv"
    
    prepare_data(RAW_DATA, PROCESSED_DATA)