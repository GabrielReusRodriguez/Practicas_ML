import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import statsmodels.api as sm

"""
Obtenemos la matriz de correlacion que nos indica en una tabla la correlacion de todas las variables con el resto de variables.
"""

# La url a la que accedemos...
url = "https://raw.githubusercontent.com/lorey/list-of-countries/master/csv/countries.csv"
# Llamamso a la funcion de lectura y además especificamos el separador que en este caso es un ; forzamos com indice la columna alpha_3
dataFrame = pd.read_csv(url,  sep=";", index_col='alpha_3' )
# Ojo que tienen valores de texto por lo que hay que forzar el numeric_only = True
corr = dataFrame.corr(numeric_only=True)
#print(f"Correlaticon :\n{corr}")
sm.graphics.plot_corr(corr, xnames= list(corr.columns))
plt.show()