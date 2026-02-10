import pandas as pd
import numpy as np

# 1. Cargar datos desde la carpeta raw
df = pd.read_csv('C:\\Users\\XD\\Documents\\GitHub\\mlops-final-project\\data\\raw\\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Limpiar TotalCharges
# Reemplazamos espacios vacíos por NaN y convertimos a float
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Llenamos los nulos (que son pocos) con la mediana o 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# 3. Transformar Variable Objetivo (Churn) a binario
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 4. Eliminar ID innecesario
df.drop('customerID', axis=1, inplace=True)

# 5. Convertir variables categóricas (One-Hot Encoding)
df_final = pd.get_dummies(df)

print("Datos listos para entrenar:")
print(df_final.head())