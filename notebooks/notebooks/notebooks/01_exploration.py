import pandas as pd

# Cargar los datos
df = pd.read_csv('C:\\Users\\XD\\Documents\\GitHub\\mlops-final-project\\data\\raw\\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Ver las primeras filas y tipos de datos
print(df.head())
print(df.info())

# Verificar la variable objetivo
print(df['Churn'].value_counts())