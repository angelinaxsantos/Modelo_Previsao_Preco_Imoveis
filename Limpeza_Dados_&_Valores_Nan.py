import pandas as pd

#Indetificação e eliminação de valores NaN, tratar de variaveis onde os valores presentes não são aceites pelo modelo

#Leitura do dataset
df = pd.read_csv('Datasets/Dataset_01.csv')

#Verificação de valores NaN
#print(df.isnull().sum())

#Eliminação de valores NaN
df.dropna(subset=['lotSize', 'livingArea', 'bathrooms', 'bedrooms', 'yearBuilt'], inplace=True)

#Verificação se ainda existe valores NaN
#print(df.isnull().sum())


#A variavel dateSold não é aceite pelo modelo, por isso é necessário transformar a variavel para um formato aceite

#Transformação da coluna dateSold para datetime "ms" para o modelo aceitar os dados
df['dateSold'] = pd.to_datetime(df['dateSold'], format='%m/%d/%Y')
df['dateSold'] = df['dateSold'].astype('int64') // 10**9

#Verificação do tipo de dados
#print(df.dtypes)

#Exportação do dataset limpo de valores NaN, e com a variavel dateSold transformada
file = 'Datasets/Dataset_02.csv'
df.to_csv(file, index=False)