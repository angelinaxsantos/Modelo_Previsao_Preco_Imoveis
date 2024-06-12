import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt

# Definindo o modelo sequencial
model = Sequential()

# Adicionando a camada de entrada e a primeira camada oculta
model.add(Dense(units=1, activation='linear', input_shape=(6,)))

# Compilando o modelo (usaremos o otimizador Adam e a perda de erro quadrático médio)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mape'])

# Visualizando o resumo do modelo
model.summary()

# Importando os datasets treino

X_treino = pd.read_csv('Datasets_Treino/X_treino.csv')
y_treino = pd.read_csv('Datasets_Treino/y_treino.csv')
X_validacao = pd.read_csv('Datasets_Treino/X_validacao.csv')
y_validacao = pd.read_csv('Datasets_Treino/y_validacao.csv')

# Treinando o modelo

resultado = model.fit(X_treino, y_treino, epochs=100, validation_split=0.1)

# Visualizando o histórico de treinamento

plt.plot(resultado.history['loss'])
plt.plot(resultado.history['val_loss'])
plt.title('Model_01 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Salvando o log de treinamento
with open('Logs_Treinamento/training_log_01.txt', 'w') as f:
    for epoch in range(len(resultado.history['loss'])):
        f.write(f'Epoch {epoch+1}: Loss - {resultado.history['loss'][epoch]}, MAPE - {resultado.history["mape"][epoch]}\n')

# Avaliando o modelo com os dados de validação
loss, mape = model.evaluate(X_validacao, y_validacao)

with open('Logs_Treinamento/validation_log_01.txt', 'w') as f:
    f.write("Resultados da Validação\n")
    f.write(f'Loss - {loss}, MAPE - {mape}\n')



