import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

#Leitura do dataset

df = pd.read_csv('Datasets/Dataset_04.csv')

#Analise de correlação entre as variaveis
matriz_correlação = df.corr()
plt.figure(figsize=(10, 7))
sns.heatmap(matriz_correlação, annot=True)
plt.show()
#Pessima correlaçao com todos menos com livingArea, bathrooms

#Criar datasets de treino, avaliação e validação

#Dividir o dataset em 3 datasets: treino, avaliação e validação
X = df.drop('price', axis=1)
y = df['price']



#X_treino, X_temp, y_treino, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
#X_avaliacao, X_validacao, y_avaliacao, y_validacao = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#Exportar os datasets
#X_treino.to_csv('Datasets_Treino/X_treino.csv', index=False)
#y_treino.to_csv('Datasets_Treino/y_treino.csv', index=False)
#X_avaliacao.to_csv('Datasets_Treino/X_teste.csv', index=False)
#y_avaliacao.to_csv('Datasets_Treino/y_teste.csv', index=False)
#X_validacao.to_csv('Datasets_Treino/X_validacao.csv', index=False)
#y_validacao.to_csv('Datasets_Treino/y_validacao.csv', index=False)
