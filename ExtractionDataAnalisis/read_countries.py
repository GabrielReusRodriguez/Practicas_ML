import pandas as pd
import numpy as py
import matplotlib.pyplot as plt

# La url a la que accedemos...
url = "https://raw.githubusercontent.com/lorey/list-of-countries/master/csv/countries.csv"
# Llamamso a la funcion de lectura y además especificamos el separador que en este caso es un ;
dataFrame = pd.read_csv(url,  sep=";", index_col='alpha_3' )
# Printamos el dataFrame.
print(f"Cantidad de Filas y columnas: {dataFrame.shape}")
print(f"Información de la dataFrame: ")
dataFrame.info()
print(end="\n\n")
print(f"Resume: {dataFrame.describe()}")