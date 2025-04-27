
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

URL = "https://raw.githubusercontent.com/lorey/list-of-countries/master/csv/countries.csv"
anomalies = []

# Funcion que encuentra los outliers ( anomalias )
def find_outliers(data: pd.DataFrame)-> list[int]:
    # Primero establecemos los limites que se calculan el max y el min +- 2 veces la desvaicion standard
    # es lo que suele considerar aceptable la gente.
    # Primero calculamos la std.
    data_sd = data.std()
    data_mean = data.mean()
    #Esto es un Series /Dataframe con el minimo de cada columna
    lower_limit = data_mean - data_sd * 2
    #Esto es un Series /Dataframe con el maximo de cada columna
    upper_limit = data_mean + data_sd * 2
    # Para cada fila, revisamos si sus valores pertenecen a outliers.
    for index, row  in data.iterrows():
        # Buscamos el outlier del area
        if row.iloc[0] > upper_limit.iloc[0] or row.iloc[0] < lower_limit.iloc[0]:
            anomalies.append(index)
    return (anomalies)

if __name__ == "__main__":
    # Leemos todos los datos.
    dataFrame = pd.read_csv(URL, sep=";", index_col = "alpha_3")
    # Substituimosl os Nan por ''
    df_espanol = dataFrame.replace(np.nan, '', regex= True)
    # Filtramos por lenguaje español.
    df_espanol = df_espanol[df_espanol['languages'].str.contains('es')]
    #Nos quedamos solo los numericos para calcular los outliers.
    data_numeric = df_espanol.select_dtypes(include=np.number)
    # Los calculamos.
    anomalies = find_outliers(data_numeric)
    # Ahora eliminamos las anmalias.
    clean_dataFrame = df_espanol.drop(anomalies)
    # Aqui obtenemos los campos population y area, ordenamos por poblacion y mandamos a plotar.
    clean_dataFrame[['population', 'area']].sort_values(['population']).plot(kind="bar", rot = 65, figsize=(20,10))
    plt.show()
