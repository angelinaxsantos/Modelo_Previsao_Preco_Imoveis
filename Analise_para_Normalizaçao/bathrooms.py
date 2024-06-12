import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


# Simulação de dados de exemplo
data = pd.read_csv('../Datasets/Dataset_03.csv')

# Visualizar a distribuição da variável
plt.figure(figsize=(10, 6))
sns.histplot(data['bathrooms'], kde=True)
plt.title('Distribuição de Casas de Banho')
plt.xlabel('Casas de Banho')
plt.ylabel('Contagem')
plt.show()


# Aplicar Log Transformation
data['bathrooms_log'] = np.log1p(data['bathrooms'])

# Aplicar Standardization
scaler_standard = StandardScaler()
data_scaled_standard = scaler_standard.fit_transform(data[['bathrooms']])
data['bathrooms_standard'] = data_scaled_standard

# Aplicar Robust Scaling
scaler_robust = RobustScaler()
data_scaled_robust = scaler_robust.fit_transform(data[['bathrooms']])
data['bathrooms_robust'] = data_scaled_robust

# Comparar as distribuições após a normalização
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
sns.histplot(data['bathrooms_log'], kde=True, bins=50)
plt.title('Log Transformation')
plt.xlabel('Casas de Banho (Log)')
plt.ylabel('Contagem')

plt.subplot(1, 3, 2)
sns.histplot(data['bathrooms_standard'], kde=True)
plt.title('Standardization')
plt.xlabel('Casas de Banho (Standard)')
plt.ylabel('Contagem')

plt.subplot(1, 3, 3)
sns.histplot(data['bathrooms_robust'], kde=True)
plt.title('Robust Scaling')
plt.xlabel('Casas de Banho (Robust)')
plt.ylabel('Contagem')

plt.tight_layout()
plt.show()
