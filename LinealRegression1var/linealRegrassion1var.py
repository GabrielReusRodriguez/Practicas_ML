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

print("Init App")

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
#plt.show()

df_filtered_data = df_articles[(df_articles['Word count'] <= 3500) & (df_articles['# Shares'] <= 80000)]
#plt.show()

colores = ['orange', 'blue']
tamanios = [30,60]

f1 = df_filtered_data['Word count'].values
f2 = df_filtered_data['# Shares'].values

asignar = []

for index, row in df_filtered_data.iterrows():
    if (row['Word count'] > 1808):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
plt.scatter(f1, f2, c= asignar, s= tamanios[0])


# Entrenamos la regresion lineal.
# El linar regresor necesita un array de n _samples , n_features es decir que si hay 400 muestras  y cada muestra tiene 1 seria 400 arrays de 1 valor
dataX = df_filtered_data[['Word count']]
dataY = df_filtered_data[['# Shares']]

"""
# Dividimos los juegos de test en 80% entreno y 20% de test


X_train, X_test, y_train, y_test = train_test_split(
        dataX,
        dataY,
        test_size = 0.01
    )
"""
X_train = dataX
X_test = dataX
y_train = dataY
y_test = dataY

"""
# Instantiate StandardScaler.
scaler = StandardScaler()

# Fit and transform training data.
X_train = scaler.fit_transform(X_train)

# Also transform test data.
X_test = scaler.transform(X_test)
"""

# Creamos el regresor lineal
regr = linear_model.LinearRegression()

# Lo entrenamos
regr.fit(X_train,  y_train)

# Ya solo quedan las predicciones.
y_pred = regr.predict(X_test)

# Check del rendimiento del modelo.
print(f"Mean squared_error :{mean_squared_error(y_test , y_pred):.2f}", end ="\n")
print(f"Variance score ={r2_score(y_test , y_pred):.2f}", end ="\n")

#print(f"Score: {regr.score(X_test, y_test)}", end = "\n")
print(f"Score: {regr.score(X_test, y_test)}", end = "\n")

#Generamos la recta 
x_line = [ x for x in range(dataX.values.min(), dataX.values.max())]
y_line = [ x * regr.coef_[0] + regr.intercept_ for x in x_line]

plt.plot(x_line, y_line, color = 'red')
plt.show()

print("End App")

