import pandas as pd
import plotly.express as px
import numpy as np

#Fazer uma analise grafica da distribuicao de cada variavel (histograma, grafico de disperção), para verificar se existem outliers, e escolher melhor metodo de remoção de outliers

#Leitura do dataset
df = pd.read_csv('Datasets/Dataset_02.csv')

def histograma(df):
    for col in df.columns:
        fig = px.histogram(df, x=col)
        fig.show()

def grafico_dispercao(df):
    for col in df.columns:
        if col == 'price':
            continue
        fig = px.scatter(df, x=col, y='price')
        fig.show()

#Verificando a distribuicao de cada variavel
#histograma(df)

#eliminando os outliers, com metodo IQR por haver assimetria nos dados (à exceçaõ da variavel 'dateSold', que não possui outliers)
def remover_outliers(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[coluna] > (Q1 - 1.5 * IQR)) & (df[coluna] < (Q3 + 1.5 * IQR))]
    return df

for col in df.columns:
    if col == 'dateSold':
        continue
    df = remover_outliers(df, col)


#Verificar se os outliers foram removidos
#histograma(df)

#Retirar valores irrealistas
#Por exemplo, a variavel 'livingArea' tem minimo de 400, e o lotSize tem minimo de 0-99 (Pelos histogramas), ambos os valores são irrealistas
df = df[df['livingArea'] > 400]
df = df[df['lotSize'] > 100]

#Verificar os imoveis onde o preço estao entre 0 e 10000
df1 = df[df['price'] < 10000]
#print(df1) #Tendo em conta a situação atual do mercado imobiliario, é irrealista que um imovel (entre T2 a T4) seja vendido por um preço inferior a 10000, logo, esses valores são retirados

df = df[df['price'] > 10000]

#Ver os imoveis com 0, 0.5 e 0.75 casas de banho
df2 = df[df['bathrooms'] == 0]
df3 = df[df['bathrooms'] == 0.5]
df4 = df[df['bathrooms'] == 0.75]
#print(df2)
#print(df3)
#print(df4)

#Nestas situações, as casas de banho são incompletas, mas para cumprir com a normalidade dos dados, é retirado os imoveis com 0, 0.5 e 0.75 casas de banho, pois estes casos podem não estar com regulação legal
df = df[df['bathrooms'] != 0]
df = df[df['bathrooms'] != 0.5]
df = df[df['bathrooms'] != 0.75]

#Verificar os imoveis com preço inferior a 30000
df5 = df[df['price'] < 30000]
#print(df5) #Tendo em conta a situação, em relação as datas de venda (2014-2015 e 2020-2022) do mercado imobiliario, é irrealista que um imovel (entre T2 a T4, com terreno bastante espaçoso) seja vendido por um preço inferior a 30000, logo, esses valores são retirados

df = df[df['price'] > 30000]

#Verificar se os valores irrealistas foram retirados
#histograma(df)

#Salvar o dataset sem outliers
df.to_csv('Datasets/Dataset_03.csv', index=False)



