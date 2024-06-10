import pandas as pd

df = pd.read_csv('Datasets/Dataset_01.csv')
#print(df.isnull().sum())

df.dropna(subset=['lotSize', 'livingArea', 'bathrooms', 'bedrooms', 'yearBuilt'], inplace=True)

print(df.isnull().sum())