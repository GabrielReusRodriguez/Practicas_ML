
import sys
import pandas as pd
import matplotlib.pyplot as plt


URL = "https://raw.githubusercontent.com/DrueStaples/Population_Growth/master/countries.csv"

def compare_2_countries(country_1: str, country_2: str):
    # Leo el dataFrame
    dataFrame = pd.read_csv(URL)
    # Checks y validaciones.
    if not country_1 in dataFrame['country'].values:
        print(f"Error, {country_1} is not a valid country code.")
        return (1)
    if not country_2 in dataFrame['country'].values:
        print(f"Error, {country_2} is not a valid country code.")
        return (1)
    # Obtengo los datos del pais 1
    df_country_1 = dataFrame[dataFrame['country'] == country_1]
    # Obtengo los datos del pais 2
    df_country_2 = dataFrame[dataFrame['country'] == country_2]
    
    #usamos la escala de años del primero como barra x de la grafica.
    anios = df_country_1['year'].unique()
    # Como tenemos los dos paises, eliminamos la columna del literal de country.
    df_country_1= df_country_1.drop(['country'], axis = 'columns')
    df_country_2.drop(['country'], axis = 'columns')
    # Ya tenemos los dos paises en dataFrames distintos, tenemos que juntarlso en un unico dataFrame.
    df_plot = pd.DataFrame({country_1: df_country_1['population'].values, country_2 : df_country_2['population'].values}, index = anios)
    # Plotamos a los dos.
    df_plot.plot(kind= 'bar')
    # Aunque usemos el plot del dataFrame hay que llamar al show de plt.
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"ERROR: this program requires Two alpha_3 country code to compare")
        exit(1)
    compare_2_countries(sys.argv[1], sys.argv[2])
