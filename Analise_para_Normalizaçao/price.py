import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler


# Simulação de dados de exemplo
np.random.seed(0)
prices = np.random.exponential(scale=400000, size=10000)

data = pd.read_csv('../Datasets/Dataset_03.csv')

# Visualizar a distribuição da variável original
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], kde=True, bins=50)
plt.title('Distribuição dos Preços')
plt.xlabel('Preços')
plt.ylabel('Contagem')
plt.show()

# Aplicar Log Transformation
data['price_log'] = np.log1p(data['price'])

# Aplicar Standardization após Log Transformation
scaler_standard = StandardScaler()
data['price_log_standard'] = scaler_standard.fit_transform(data[['price_log']])

# Aplicar Robust Scaling
scaler_robust = RobustScaler()
data['price_robust'] = scaler_robust.fit_transform(data[['price']])

# Comparar as distribuições após a normalização
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
sns.histplot(data['price_log'], kde=True, bins=50)
plt.title('Log Transformation')
plt.xlabel('Preços (Log)')
plt.ylabel('Contagem')

plt.subplot(1, 3, 2)
sns.histplot(data['price_log_standard'], kde=True, bins=50)
plt.title('Log + Standardization')
plt.xlabel('Preços (Log + Standard)')
plt.ylabel('Contagem')

plt.subplot(1, 3, 3)
sns.histplot(data['price_robust'], kde=True, bins=50)
plt.title('Robust Scaling')
plt.xlabel('Preços (Robust)')
plt.ylabel('Contagem')

plt.tight_layout()
plt.show()
