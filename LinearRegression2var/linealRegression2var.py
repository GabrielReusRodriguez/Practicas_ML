#!./venv/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import  linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,  r2_score


from sklearn.preprocessing import StandardScaler


DATA_LOCATION = './data/articulos_ml.csv'

NUM_RESUME_LINES = 10

# Cargamos el dataFrame.

df_articles = pd.read_csv(
    filepath_or_buffer= DATA_LOCATION
    )

#print(df_articles.head(NUM_RESUME_LINES))
#print(df_articles.describe())

# Histograma de los datos.
#fig, ax = plt.subplots(1,1, figsize = (20,10))
#df_articles.drop(['Title', 'url', 'Elapsed days'], axis = 'columns').hist(ax= ax)
#df_articles.drop(['Title', 'url', 'Elapsed days'], axis = 'columns').hist()
# Nos quedamos con la mayoria de datos


df_filtered_data = df_articles[(df_articles['Word count'] <= 3500) & (df_articles['# Shares'] <= 80000)]

# Generamos una tercera variable que es la suma de enlaces , comentarios e imagenes.

suma = (df_filtered_data['# of Links'] + df_filtered_data['# of comments'].fillna(0) + df_filtered_data['# Images video'])

# Creamos el nuevo dataFrame.
dataX2 = pd.DataFrame()
dataX2["Word count"] = df_filtered_data["Word count"]
dataX2["suma"] = suma

# Generamos el juego de datos de entrenamiento
XY_train = np.array(dataX2)
# Genero los resultados de las muestras
z_train =  df_filtered_data['# Shares'].values

# Una vez generado el set de entenamiento, generamos el regresos lineal de 2 dimensiones.
regr2 = linear_model.LinearRegression()
# Entrenamos el regresor.
regr2.fit(XY_train, z_train)
# Hacemos la predicción.
z_pred = regr2.predict(XY_train)
# Printamos el modelo.
print(f"Coeficientes: \n {regr2.coef_}")

# Calculamos el rendimiento
print(f"Mean squared error : \n {mean_squared_error(z_train, z_pred):.2f}")
print(f"Variance score : \n {r2_score(z_train, z_pred):.2f}")
print(f"Scores : \n {regr2.score(z_train, z_pred)}")

