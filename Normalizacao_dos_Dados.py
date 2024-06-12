import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

df = pd.read_csv('Datasets/Dataset_03.csv')


def histograma(df):
    for col in df.columns:
        fig = px.histogram(df, x=col)
        fig.show()

#histograma(df)


#Normalização dos dados, após a análise dos gráficos de barras

#Aplicação da Normalização Log Transformation na variável 'price'
df['price'] = np.log1p(df['price'])

#Aplicação da Normalização Z-Score na variável 'bedrooms'
scaler_standard = StandardScaler()
df_scaled_standard = scaler_standard.fit_transform(df[['bedrooms']])
df['bedrooms'] = df_scaled_standard

#Aplicação da Normalização Log Transformation na variável 'bathrooms'
df['bathrooms'] = np.log1p(df['bathrooms'])

#Aplicação da Normalização Log Transformation na variável 'livingArea'
df['livingArea'] = np.log1p(df['livingArea'])

#Aplicação da Normalização IQR na variável 'lotSize'
scaler_robust = RobustScaler()
df_scaled_robust = scaler_robust.fit_transform(df[['lotSize']])
df['lotSize'] = df_scaled_robust

#Aplicação da Normalização IQR na variável 'yearBuilt'
scaler_robust = RobustScaler()
df_scaled_robust = scaler_robust.fit_transform(df[['yearBuilt']])
df['yearBuilt'] = df_scaled_robust

#Aplicação da Normalização Z-Score na variável 'dateSold'
scaler_standard = StandardScaler()
df_scaled_standard = scaler_standard.fit_transform(df[['dateSold']])
df['dateSold'] = df_scaled_standard

#Salvar o dataset normalizado
df.to_csv('Datasets/Dataset_04.csv', index=False)