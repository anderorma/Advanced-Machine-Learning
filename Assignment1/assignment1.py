import pandas as pd
import os

import torch
import torch.nn as nn
import torch.nn.init as init

import sklearn
from sklearn.impute import KNNImputer

from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Check if an NVIDIA GPU (CUDA) is available. 
# If yes, we use 'cuda'; otherwise, we fall back to the 'cpu'.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device choice so we know if the code is running on the GPU or CPU
print(f"Using device: {DEVICE}\n")

stations = [
    "Aotizhongxin", "Changping", "Dingling", "Dongsi", 
    "Guanyuan", "Gucheng", "Huairou", "Nongzhanguan", 
    "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"
]

df_list = [pd.read_csv(f"Assignment1/Datasets/PRSA_Data_{name}_20130301-20170228.csv") for name in stations]

# We load all dataframes into a single one
beijing_total = pd.concat(df_list, ignore_index=True)

# Add the coordinates of the stations to the dataframe
beijing_coords = pd.read_csv("Assignment1/Datasets/beijingCoordenadas.csv", sep = ';')
beijing_total = pd.merge(beijing_total, beijing_coords, on='station', how='left')

print("GENERAL INFORMATION ABOUT THE DATASET:")
print("\nFirst 5 rows:")
print(beijing_total.head())
print("\nLast 5 rows:")
print(beijing_total.tail())
print("\nData type of each column:")
print(beijing_total.dtypes)

# DATA IMPUTATION ------------------------------------------------------------------------------------------------------------------------
print("\n*** DATA IMPUTATION ***")
print("\nNULL VALUES IN THE DATASET:\n")
print(beijing_total.isnull().sum())

# Linear interpolation (df_interp)

cols_num = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
df_interp = beijing_total.copy()
df_interp = df_interp.sort_values(by=['station', 'year', 'month', 'day', 'hour'])

# 2. Aplicamos la interpolación por estación
# Usamos groupby para que no intente interpolar el final de una estación con el principio de otra
for col in cols_num:
    df_interp[col] = df_interp.groupby('station')[col].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )

# 3. Para la variable categórica 'wd' (texto), usamos ffill/bfill
# Ya que no se puede hacer una "línea" entre direcciones de viento
df_interp['wd'] = df_interp.groupby('station')['wd'].transform(
    lambda x: x.ffill().bfill()
)

print("\nNULL VALUES AFTER INTERPOLATION:\n")
print(df_interp.isnull().sum())


# KNN imputation (beijing_knn)

def imputar_knn_geografico_estricto(df_principal, perfil):
    cols_datos = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']
    df_res = df_principal.copy()
    
    # Agrupamos por mes y hora para cumplir tu restricción temporal
    for (mes, hora), grupo_instante in df_res.groupby(['month', 'hour']):
        
        # 1. Obtener el perfil de estaciones para este mes/hora
        perfil_instante = perfil[(perfil['month'] == mes) & (perfil['hour'] == hora)].copy()
        if perfil_instante.empty: continue

        # 2. "ENTRENAR" EL MODELO KNN (Solo con Latitud y Longitud)
        # Aquí es donde garantizamos que la distancia sea SOLO geográfica
        knn = NearestNeighbors(n_neighbors=len(perfil_instante), metric='euclidean')
        knn.fit(perfil_instante[['Latitud', 'Longitud']])

        # 3. Para cada fila con nulos en este grupo
        for col in cols_datos:
            nulos = grupo_instante[grupo_instante[col].isnull()]
            if nulos.empty: continue
            
            # Encontrar los vecinos más cercanos para estas estaciones con nulos
            distancias, indices = knn.kneighbors(nulos[['Latitud', 'Longitud']])
            
            # 'indices' nos dice qué estaciones del perfil son las más cercanas
            for i, idx_en_nulos in enumerate(nulos.index):
                # Buscamos en los vecinos (del más cercano al más lejano)
                for vecino_idx_en_perfil in indices[i]:
                    valor_vecino = perfil_instante.iloc[vecino_idx_en_perfil][col]
                    
                    if pd.notna(valor_vecino):
                        df_res.at[idx_en_nulos, col] = valor_vecino
                        break
    return df_res


# Definimos el Perfil Maestro
perfil_maestro = beijing_total.groupby(['month', 'hour', 'station']).agg({
    # Variables que queremos promediar para usar como relleno
    'PM2.5': 'mean',
    'PM10': 'mean',
    'SO2': 'mean',
    'NO2': 'mean',
    'CO': 'mean',
    'O3': 'mean',
    'TEMP': 'mean',
    'PRES': 'mean',
    'DEWP': 'mean',
    'RAIN': 'mean',
    'wd': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
    'WSPM': 'mean',
 # Coordenadas (usamos 'first' porque no cambian para una misma estación)
    'Latitud': 'first',
    'Longitud': 'first'
}).reset_index()

# Importante: Si una estación estuvo rota todo un mes a una hora específica, 
# tendrá un NaN en el perfil. El KNN lo ignorará y pasará a la siguiente estación más cercana.

# Ejecución
print("\nImputation of values using KNN with strict geographic distance...")
beijing_knn = imputar_knn_geografico_estricto(beijing_total, perfil_maestro)
print("\nNULL VALUES AFTER KNN IMPUTATION:\n")
print(beijing_knn.isnull().sum())

# Compare both methods
print("\nCOMPARISON OF BOTH IMPUTATION METHODS:\n")

# Comparar la desviación estándar para ver cuál preserva mejor la variabilidad
print(pd.DataFrame({
    'Original': beijing_total[cols_num].std(),
    'Interpolacion': df_interp[cols_num].std(),
    'KNN': beijing_knn[cols_num].std()
}))
# ambos han respetado el limite fisico del dataset
print("\nCOMPARISON OF MAX VALUES (SHOULD BE SIMILAR OR EQUAL TO ORIGINAL):\n")
print(df_interp['PM2.5'].max())
print(beijing_knn['PM2.5'].max())
print(beijing_total['PM2.5'].max())


# 1. Create a copy of rows that have NO nulls to use as our "Ground Truth"
df_complete = beijing_total.dropna(subset=cols_num).sample(n=1000, random_state=42).copy()
real_values = df_complete[cols_num].copy()

# 2. Simulate missing values (hide 20% of the data in our sample)
df_simulated = df_complete.copy()
for col in cols_num:
    df_simulated.loc[df_simulated.sample(frac=0.2, random_state=42).index, col] = np.nan

# --- Apply Interpolation on the simulated gap ---
df_sim_interp = df_simulated.copy()
df_sim_interp[cols_num] = df_sim_interp[cols_num].interpolate(method='linear', limit_direction='both')

# For the string column 'wd', we use ffill/bfill just like in your main code
df_sim_interp['wd'] = df_sim_interp['wd'].ffill().bfill()

# --- Apply your KNN function on the simulated gap ---
# Note: Ensure 'perfil_maestro' is available in your environment
df_sim_knn = imputar_knn_geografico_estricto(df_simulated, perfil_maestro)

# 3. Calculate RMSE for a key variable (e.g., PM2.5)
rmse_interp = np.sqrt(mean_squared_error(real_values['PM2.5'], df_sim_interp['PM2.5']))
rmse_knn = np.sqrt(mean_squared_error(real_values['PM2.5'], df_sim_knn['PM2.5']))

print("\nRMSE COMPARISON FOR PM2.5:")
print(f"RMSE PM2.5 (Interpolation): {rmse_interp:.2f}")
print(f"RMSE PM2.5 (KNN Geográfico): {rmse_knn:.2f}")

# Compare how well the correlation between PM2.5 and TEMP is preserved
corr_orig = beijing_total['PM2.5'].corr(beijing_total['TEMP'])
corr_interp = df_interp['PM2.5'].corr(df_interp['TEMP'])
corr_knn = beijing_knn['PM2.5'].corr(beijing_knn['TEMP'])

print("\nCORRELATION PRESERVATION (PM2.5 vs TEMP):")
print(f"Original:      {corr_orig:.4f}")
print(f"Interpolation: {corr_interp:.4f}")
print(f"KNN:           {corr_knn:.4f}")

# 1. Apply the DatetimeIndex to ALL dataframes involved
for df in [beijing_total, df_interp, beijing_knn, df_complete, df_sim_interp, df_sim_knn]:
    # Only convert if the index isn't already a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

# 1. Filter a specific station from your simulated sample
# (Ensure this station exists in your df_complete sample)
sample_station = df_complete['station'].unique()[0]

# 2. Filter and explicitly SORT by the index (assuming your index is the DatetimeIndex)
truth_plot = df_complete[df_complete['station'] == sample_station].sort_index()
knn_plot = df_sim_knn[df_sim_knn['station'] == sample_station].sort_index()
interp_plot = df_sim_interp[df_sim_interp['station'] == sample_station].sort_index()

plt.figure(figsize=(15, 6))

# 3. PLOT USING THE INDEX VALUES
# We use truth_plot.index explicitly to force the date axis
plt.plot(truth_plot.index, truth_plot['PM2.5'], 
         label='GROUND TRUTH', color='black', linewidth=2.5, zorder=1)

plt.plot(knn_plot.index, knn_plot['PM2.5'], 
         label='KNN Prediction', color='blue', alpha=0.8, zorder=3)

plt.plot(interp_plot.index, interp_plot['PM2.5'], 
         label='Linear Interpolation', color='red', alpha=0.7, linestyle='--', zorder=2)

# 4. Format the date axis
plt.title(f"Validation over Time: {sample_station}")
plt.ylabel("PM2.5 Concentration")
plt.xlabel("Date")

# This is the "Magic" line that formats the X-axis into readable dates
plt.gcf().autofmt_xdate() 

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
print("------------------------------------------------------------------------------\n"
"------------------------------------------------------------------------------\n")

# ----------------------------------------------------------------------------------------------------------------------------------------
#we choose: beijing_knn because it has a lower RMSE and preserves better the correlation between PM2.5 and TEMP, which is an important relationship in air quality data.    
# 1. Seleccionamos la estación
nombre_estacion = beijing_knn['station'].unique()[0] 

# 2. Filtramos por estación, ordenamos Y filtramos solo el año 2015
# El uso de .loc['2015'] es la forma más rápida y limpia de hacerlo con DatetimeIndex
df_estacion_2015 = beijing_knn[beijing_knn['station'] == nombre_estacion].sort_index().loc['2015']

plt.figure(figsize=(15, 7))

# 3. Graficamos los datos horarios de 2015
plt.plot(df_estacion_2015.index, df_estacion_2015['PM2.5'], 
         color='royalblue', linewidth=0.8, alpha=0.8, label='Datos Horarios 2015')

# 4. Media móvil (mantenemos las 168 horas para ver la tendencia semanal en 2015)
plt.plot(df_estacion_2015['PM2.5'].rolling(window=168, center=True).mean(), 
         color='orange', linewidth=2.5, label='Tendencia Semanal')

# Configuración
plt.title(f"PM2.5 en 2015 - Estación: {nombre_estacion}", fontsize=15)
plt.ylabel(r"Concentración de PM2.5 ($\mu g/m^3$)")
plt.xlabel("Meses de 2015")
plt.legend()
plt.grid(True, alpha=0.2)

# Formatear eje X para que se vean bien los meses
plt.gcf().autofmt_xdate()

plt.show()

