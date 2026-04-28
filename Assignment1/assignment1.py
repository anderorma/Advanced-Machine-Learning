
# INSTALLED LIBRARIES
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

# We load all the dataframes for each station into a list using a list 
df_list = [pd.read_csv(f"Assignment1/Datasets/PRSA_Data_{name}_20130301-20170228.csv") for name in stations]

# We load all dataframes into a single one
beijing_total = pd.concat(df_list, ignore_index=True)

# Add the coordinates of the stations to the dataframe from the beijingCoordenadas.csv file, which contains the latitude and longitude of each station.
# We will use this information for the KNN imputation later on.
beijing_coords = pd.read_csv("Assignment1/Datasets/beijingCoordenadas.csv", sep = ';')
beijing_total = pd.merge(beijing_total, beijing_coords, on='station', how='left')

print("GENERAL INFORMATION ABOUT THE DATASET:")
print("\nFirst 5 rows:")
print(beijing_total.head())
print("\nLast 5 rows:")
print(beijing_total.tail())
print("\nData type of each column:")
print(beijing_total.dtypes)

# =================================================================
# DATA IMPUTATION ------------------------------------------------------------------------------------------------------------------------
# =================================================================

print("\n================================")
print("*** DATA IMPUTATION ***")
print("================================")
print("\nNULL VALUES IN THE DATASET:\n")
print(beijing_total.isnull().sum())

# LINEAR INTERPOLATION (df_interp)
cols_num = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
df_interp = beijing_total.copy()
df_interp = df_interp.sort_values(by=['station', 'year', 'month', 'day', 'hour'])

# We apply interpolation separately for each station to avoid mixing data from different stations
for col in cols_num:
    df_interp[col] = df_interp.groupby('station')[col].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )

# For the variable 'wd' (categorical), we use ffill/bfill
# Since we can't create a "line" between wind direction values, 
# we propagate the last valid observation forward and backward to fill the gaps.
df_interp['wd'] = df_interp.groupby('station')['wd'].transform(
    lambda x: x.ffill().bfill()
)

print("\nNULL VALUES AFTER INTERPOLATION:\n")
print(df_interp.isnull().sum())


# KNN IMPUTATION (beijing_knn)
def imputar_knn_geografico_estricto(df_principal, perfil):
    cols_datos = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']
    df_res = df_principal.copy()
    
    # We group by month and hour to ensure we only compare stations that are in the same temporal context
    for (mes, hora), grupo_instante in df_res.groupby(['month', 'hour']):
        
        # We filter the profile to only include stations that have data for this specific month and hour,
        # this way, we ensure that the KNN will only consider stations that are actually comparable in terms of time, 
        # and we avoid using data from stations that might be completely missing for that time period.
        perfil_instante = perfil[(perfil['month'] == mes) & (perfil['hour'] == hora)].copy()
        if perfil_instante.empty: continue

        # Train the KNN model using only the geographic coordinates (Latitud, Longitud) of the stations in this profile
        # We guarantee that the KNN will only consider the distance between stations, and not the values of the variables, to find the nearest neighbors.
        knn = NearestNeighbors(n_neighbors=len(perfil_instante), metric='euclidean')
        knn.fit(perfil_instante[['Latitud', 'Longitud']])

        # Now we look for the rows in 'grupo_instante' that have null values in any of the columns we want to impute.
        for col in cols_datos:
            nulos = grupo_instante[grupo_instante[col].isnull()]
            if nulos.empty: continue
            
            # For each row with a null value, we find the nearest neighbors in the profile based on geographic distance.
            distancias, indices = knn.kneighbors(nulos[['Latitud', 'Longitud']])
            
            # 'indices' gives us the positions of the nearest neighbors in 'perfil_instante', ordered from closest to farthest.
            for i, idx_en_nulos in enumerate(nulos.index):
                # Buscamos en los vecinos (del más cercano al más lejano)
                for vecino_idx_en_perfil in indices[i]:
                    valor_vecino = perfil_instante.iloc[vecino_idx_en_perfil][col]
                    
                    if pd.notna(valor_vecino):
                        df_res.at[idx_en_nulos, col] = valor_vecino
                        break
    return df_res


# Before running the imputation function, we create a reference table called 'perfil_maestro'.
# 
# 1. Grouping: We group the entire dataset by 'month', 'hour', and 'station'.
# 2. Averaging: For each specific station, we calculate the historical average of 
#    pollutants and weather conditions for that specific hour and month.
# 3. Result: This creates an "average snapshot" of the typical values for each station 
#    (e.g., the typical temperature at "Aotizhongxin" station during January at 10:00 AM).
#    This master profile serves as the reliable source to fill in missing gaps based 
#    on historical patterns.
perfil_maestro = beijing_total.groupby(['month', 'hour', 'station']).agg({
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
 # Coordinates (we take the first value since they are the same for each station)
    'Latitud': 'first',
    'Longitud': 'first'
}).reset_index()

# Execution of the KNN imputation function
print("\nImputation of values using KNN with strict geographic distance...")
beijing_knn = imputar_knn_geografico_estricto(beijing_total, perfil_maestro)
print("\nNULL VALUES AFTER KNN IMPUTATION:\n")
print(beijing_knn.isnull().sum())

# Compare both methods
print("\nCOMPARISON OF BOTH IMPUTATION METHODS:\n")

# Compares standard deviation of the original data with the imputed data (both linear interpolationand KNN),
# to see if the variability is preserved.
print(pd.DataFrame({
    'Original': beijing_total[cols_num].std(),
    'Interpolacion': df_interp[cols_num].std(),
    'KNN': beijing_knn[cols_num].std()
}))
# Compares the maximum values of PM2.5 in the original data with the imputed data, to see if extreme values are preserved.
print("\nCOMPARISON OF MAX VALUES (SHOULD BE SIMILAR OR EQUAL TO ORIGINAL):")
print(df_interp['PM2.5'].max())
print(beijing_knn['PM2.5'].max())
print(beijing_total['PM2.5'].max())


# VALIDATION OF IMPUTED VALUES
# --- STEP 1: PREPARE GROUND TRUTH ---
# We select 1,000 rows that have NO missing values to act as our "gold standard" (Ground Truth).
# This allows us to compare the imputed values against the actual known values later.
df_complete = beijing_total.dropna(subset=cols_num).sample(n=1000, random_state=42).copy()
real_values = df_complete[cols_num].copy()

# --- STEP 2: SIMULATE DATA LOSS ---
# We intentionally "break" our clean dataset by hiding 20% of the values in each column.
# These NaNs are what our models will try to predict.
df_simulated = df_complete.copy()
for col in cols_num:
    df_simulated.loc[df_simulated.sample(frac=0.2, random_state=42).index, col] = np.nan

# --- STEP 3: APPLY IMPUTATION METHODS ---
# Method A: Linear Interpolation (guesses values based on the mathematical slope between points).
df_sim_interp = df_simulated.copy()
df_sim_interp[cols_num] = df_sim_interp[cols_num].interpolate(method='linear', limit_direction='both')
df_sim_interp['wd'] = df_sim_interp['wd'].ffill().bfill() # Handle categorical wind direction

# Method B: Geographic KNN (your custom function using nearby stations' averages).
df_sim_knn = imputar_knn_geografico_estricto(df_simulated, perfil_maestro)

# --- STEP 4: QUANTITATIVE EVALUATION (METRICS) ---
# Calculate RMSE (Root Mean Squared Error): Lower is better. 
# It measures the average "distance" between the imputed value and the real value.
rmse_interp = np.sqrt(mean_squared_error(real_values['PM2.5'], df_sim_interp['PM2.5']))
rmse_knn = np.sqrt(mean_squared_error(real_values['PM2.5'], df_sim_knn['PM2.5']))

print("\nRMSE COMPARISON FOR PM2.5:")
print(f"RMSE PM2.5 (Interpolation): {rmse_interp:.2f}")
print(f"RMSE PM2.5 (KNN Geográfico): {rmse_knn:.2f}")

# Correlation Preservation: Checks if the relationship between PM2.5 and Temperature 
# is still maintained after imputation. A good method shouldn't distort these physics relationships.
corr_orig = beijing_total['PM2.5'].corr(beijing_total['TEMP'])
corr_interp = df_interp['PM2.5'].corr(df_interp['TEMP'])
corr_knn = beijing_knn['PM2.5'].corr(beijing_knn['TEMP'])

# --- STEP 5: TIME SERIES CONVERSION ---
# Ensure all DataFrames use a proper DatetimeIndex for chronological plotting.
for df in [beijing_total, df_interp, beijing_knn, df_complete, df_sim_interp, df_sim_knn]:
    if not isinstance(df.index, pd.DatetimeIndex):
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

# --- STEP 6: VISUAL VALIDATION ---
# Pick one specific station and plot the results to see which line follows the "Ground Truth" 
# more closely.
sample_station = df_complete['station'].unique()[0]
truth_plot = df_complete[df_complete['station'] == sample_station].sort_index()
knn_plot = df_sim_knn[df_sim_knn['station'] == sample_station].sort_index()
interp_plot = df_sim_interp[df_sim_interp['station'] == sample_station].sort_index()

plt.figure(figsize=(15, 6))
plt.plot(truth_plot.index, truth_plot['PM2.5'], label='GROUND TRUTH', color='black', linewidth=2.5, zorder=1)
plt.plot(knn_plot.index, knn_plot['PM2.5'], label='KNN Prediction', color='blue', alpha=0.8, zorder=3)
plt.plot(interp_plot.index, interp_plot['PM2.5'], label='Linear Interpolation', color='red', alpha=0.7, linestyle='--', zorder=2)

plt.title(f"Validation over Time: {sample_station}")
plt.ylabel("PM2.5 Concentration")
plt.xlabel("Date")
plt.gcf().autofmt_xdate() # Auto-rotate dates for readability
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# ----------------------------------------------------------------------------------------------------------------------------------------
# VISUALIZATION OF PM2.5 IN 2015 FOR ONE STATION

# We choose 'beijing_knn' because it achieved a lower RMSE (higher accuracy) 
# and better preserved the physical correlation between PM2.5 and Temperature 
# compared to linear interpolation.

# We select the first available station name from the dataset.
nombre_estacion = beijing_knn['station'].unique()[0] 

# We filter the data for this station, sort it chronologically, 
# and specifically extract only the year 2015 using the DatetimeIndex.
df_estacion_2015 = beijing_knn[beijing_knn['station'] == nombre_estacion].sort_index().loc['2015']

plt.figure(figsize=(15, 7))

# We plot the hourly PM2.5 concentrations. 
plt.plot(df_estacion_2015.index, df_estacion_2015['PM2.5'], 
         color='royalblue', linewidth=0.8, alpha=0.8, label='Hourly Data 2015')

plt.plot(df_estacion_2015['PM2.5'].rolling(window=168, center=True).mean(), 
         color='orange', linewidth=2.5, label='Weekly Trend')

plt.title(f"PM2.5 in 2015 - Station: {nombre_estacion}", fontsize=15)
plt.ylabel(r"PM2.5 Concentration ($\mu g/m^3$)")
plt.xlabel("Months of 2015")
plt.legend()
plt.grid(True, alpha=0.2)

# Automatically format and rotate the date labels on the X-axis for readability.
plt.gcf().autofmt_xdate()

plt.show()

# =================================================================
# TIME SERIES ------------------------------------------------------------------------------------------------------------------------
# =================================================================
print("\n================================")
print("\n*** TIME SERIES ANALYSIS ***")
print("\n================================")
# we have to make sure that the data is stationary before applying the SARIMA model, so we will
# apply the Dickey-Fuller test to check for stationarity. 
# We filter the data for one station and one year 
estacion_un_año = beijing_knn[
    (beijing_knn['station'] == sample_station) & 
    (beijing_knn.index.year == 2014)
]['PM2.5'].resample('D').mean()

# 2. Manejo de posibles nulos (importante para la descomposición)
estacion_un_año = estacion_un_año.ffill().bfill()

# Decomposition of the time series into its components: observed, trend, seasonal, and residual.
res = seasonal_decompose(estacion_un_año, model='additive', period=7)

# We plot the original series, the trend, the seasonality, and the residuals to visually inspect the components of the time series.
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
res.observed.plot(ax=ax1, title=f'Análisis PM2.5 - {sample_station} (2014)')
ax1.set_ylabel('Original')
res.trend.plot(ax=ax2)
ax2.set_ylabel('Tendencia')
res.seasonal.plot(ax=ax3)
ax3.set_ylabel('Estacionalidad')
res.resid.plot(ax=ax4)
ax4.set_ylabel('Residuos (Ruido)')

plt.tight_layout()
plt.show()

# we confirm the stationarity of the series using the Dickey-Fuller test
def test_estacionariedad(serie):
    print("\nRESULTS OF THE DICKEY-FULLER TEST:")
    resultado = adfuller(serie)
    print(f'ADF Statistic: {resultado[0]:.4f}')
    print(f'p-value: {resultado[1]:.4f}')
    if resultado[1] <= 0.05:
        print("\nCONCLUSION: The series is STATIONARY")
    else:
        print("\nCONCLUSION: The series is NOT STATIONARY")

test_estacionariedad(estacion_un_año)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# ACF help us see the parameter 'q' (moving average) and the overall autocorrelation structure of the series.
plot_acf(estacion_un_año, lags=40, ax=ax1)
# PACF help us see the parameter 'p'
plot_pacf(estacion_un_año, lags=40, ax=ax2)

plt.tight_layout()
plt.show()


# 1. DATA PREPARATION (Multi-annual Logic)
# Target: Train with 2014-2015 (2 full years) | Test with 2016 (1 full year)
# This allows the model to learn historical seasonal cycles (winter vs winter).

columnas_num = ['PM2.5', 'TEMP', 'WSPM', 'PRES', 'RAIN']

# --- Training Set: Years 2014 and 2015 ---
datos_entrenamiento = beijing_knn[
    (beijing_knn['station'] == 'Gucheng') & 
    (beijing_knn.index.year.isin([2014, 2015]))
][columnas_num].resample('D').mean().ffill()

train_y = datos_entrenamiento['PM2.5']
train_exog = datos_entrenamiento[['TEMP', 'WSPM', 'PRES', 'RAIN']]

# --- Testing Set: Year 2016 ---
datos_test = beijing_knn[
    (beijing_knn['station'] == 'Gucheng') & 
    (beijing_knn.index.year == 2016)
][columnas_num].resample('D').mean().ffill()

test_y = datos_test['PM2.5']
test_exog = datos_test[['TEMP', 'WSPM', 'PRES', 'RAIN']]


# 2. SARIMA MODEL (Univariate - Only looks at past PM2.5)
# We fit the model using only the PM2.5 target variable.
model_sarima = sm.tsa.statespace.SARIMAX(train_y,
                                        order=(1, 0, 1),
                                        seasonal_order=(1, 1, 1, 7),
                                        enforce_stationarity=False)
res_sarima = model_sarima.fit(disp=False)

# Forecast for the 366 days of 2016 (Leap year)
pred_sarima = res_sarima.get_forecast(steps=len(test_y)).predicted_mean.clip(lower=0)


# 3. SARIMAX MODEL (Multivariate - Past PM2.5 + Weather)
# We fit the model using PM2.5 AND weather variables as exogenous features.
model_sarimax = sm.tsa.statespace.SARIMAX(train_y,
                                         exog=train_exog,
                                         order=(1, 0, 1),
                                         seasonal_order=(1, 1, 1, 7),
                                         enforce_stationarity=False)
res_sarimax = model_sarimax.fit(disp=False)

# Forecast for 2016 using the 2016 weather data as exogenous input
pred_sarimax = res_sarimax.get_forecast(steps=len(test_y), exog=test_exog).predicted_mean.clip(lower=0)


# 4. COMPARATIVE VISUALIZATION
plt.figure(figsize=(15, 8))

# Ground Truth (Real 2016 data)
plt.plot(test_y.index, test_y, label='Actual 2016 Data', color='black', alpha=0.3, linewidth=1)

# SARIMA Prediction (Green line)
plt.plot(pred_sarima.index, pred_sarima, label='SARIMA (Historical patterns only)', color='green', linewidth=2)

# SARIMAX Prediction (Red line)
plt.plot(pred_sarimax.index, pred_sarimax, label='SARIMAX (Historical + Weather)', color='red', linewidth=2)

plt.title("Model Battle: Predicting 2016 (Trained on 2014-2015)", fontsize=14)
plt.ylabel("PM2.5 Concentration")
plt.xlabel("Date")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()


# 5. FINAL PERFORMANCE EVALUATION (RMSE)
rmse_sarima = np.sqrt(mean_squared_error(test_y, pred_sarima))
rmse_sarimax = np.sqrt(mean_squared_error(test_y, pred_sarimax))

print(f"\n--- 2016 FINAL EVALUATION RESULTS ---")
print(f"SARIMA RMSE (Univariate): {rmse_sarima:.2f}")
print(f"SARIMAX RMSE (Multivariate): {rmse_sarimax:.2f}")

# Direct comparison to prove if weather data adds value
if rmse_sarimax < rmse_sarima:
    print(f"\nCONCLUSION: SARIMAX is superior. Weather data reduced the error by {rmse_sarima - rmse_sarimax:.2f} points.")
else:
    print("\nCONCLUSION: SARIMA performed better. Weather data might have introduced noise in this specific period.")

# 6. RESIDUAL ANALYSIS (SARIMAX)
# We calculate and plot residuals to check for any remaining patterns.
residuals = test_y - pred_sarimax
plt.figure(figsize=(12, 4))
plt.scatter(residuals.index, residuals, color='purple', alpha=0.4, s=10)
plt.axhline(0, color='black', linestyle='--')
plt.title("Residual Analysis (SARIMAX) - Error Distribution in 2016")
plt.ylabel("Error (Actual - Predicted)")
plt.show()

# FUTURE FORECASTING (MARCH 2017)
# 1. PREPARE HISTORICAL DATA (Multi-annual context)
# We use all available data from the Gucheng station to train our final models
serie_datos = beijing_knn[beijing_knn['station'] == 'Gucheng'][columnas_num].resample('D').mean().ffill()

# Define the training variables
y_historia = serie_datos['PM2.5']
exog_historia = serie_datos[['TEMP', 'WSPM', 'PRES', 'RAIN']]

# 2. CREATE THE "SYNTHETIC CLIMATE" FOR THE FUTURE
# We find the typical weather for each month/day based on 2014-2015 patterns
clima_estacional = serie_datos.groupby([serie_datos.index.month, serie_datos.index.day]).mean()

# Define the forecast period: 30 days after your last data point (March 2017)
fecha_inicio_futuro = serie_datos.index[-1] + pd.Timedelta(days=1)
indice_futuro = pd.date_range(start=fecha_inicio_futuro, periods=30, freq='D')

# Map the historical average weather to these future dates
exog_futura = pd.DataFrame([clima_estacional.loc[(d.month, d.day)] for d in indice_futuro], 
                          index=indice_futuro)[['TEMP', 'WSPM', 'PRES', 'RAIN']]

# 3. TRAIN FINAL MODELS ON FULL HISTORY
# --- Model A: SARIMA (Univariate - Only looks at past PM2.5) ---
modelo_final_sarima = sm.tsa.statespace.SARIMAX(y_historia, 
                                               order=(1, 0, 1),
                                               seasonal_order=(1, 1, 1, 7),
                                               enforce_stationarity=False)
res_final_sarima = modelo_final_sarima.fit(disp=False)

# --- Model B: SARIMAX (Multivariate - PM2.5 + Synthetic Climate) ---
modelo_final_sarimax = sm.tsa.statespace.SARIMAX(y_historia,
                                                exog=exog_historia,
                                                order=(1, 0, 1),
                                                seasonal_order=(1, 1, 1, 7),
                                                enforce_stationarity=False)
res_final_sarimax = modelo_final_sarimax.fit(disp=False)

# 4. GENERATE THE 30-DAY FORECAST (MARCH 2017)

# SARIMA Forecast
forecast_sarima = res_final_sarima.get_forecast(steps=30)
puntos_sarima = forecast_sarima.predicted_mean.clip(lower=0)

# SARIMAX Forecast (Using the Synthetic Climate)
forecast_sarimax = res_final_sarimax.get_forecast(steps=30, exog=exog_futura)
puntos_sarimax = forecast_sarimax.predicted_mean.clip(lower=0)
intervalos_x = forecast_sarimax.conf_int()

# 5. MASTER COMPARATIVE VISUALIZATION

# Prepare the "Mirror" data: Real March 2016 values to compare with March 2017 forecast
fecha_inicio_anterior = fecha_inicio_futuro - pd.DateOffset(years=1)
indice_anterior = pd.date_range(start=fecha_inicio_anterior, periods=30, freq='D')
datos_año_anterior = y_historia.reindex(indice_anterior)
datos_año_anterior.index = indice_futuro  # Align to 2017 timeline for overlay

plt.figure(figsize=(15, 8))

# Context: The last 60 days of real history
plt.plot(y_historia.index[-60:], y_historia[-60:], label='Actual History (Context)', color='black', alpha=0.3)

# Mirror: Last year's reality (March 2016)
plt.plot(datos_año_anterior.index, datos_año_anterior, color='blue', linestyle='--', alpha=0.5, label='Actual March 2016 (The Mirror)')

# Forecast: SARIMA (Green)
plt.plot(puntos_sarima.index, puntos_sarima, color='green', linewidth=2, label='SARIMA Forecast (Blind to Weather)')

# Forecast: SARIMAX (Red)
plt.plot(puntos_sarimax.index, puntos_sarimax, color='red', linewidth=2, label='SARIMAX Forecast (Driven by Climate)')

# Uncertainty Shading for SARIMAX
plt.fill_between(puntos_sarimax.index, 
                 intervalos_x.iloc[:, 0].clip(lower=0), 
                 intervalos_x.iloc[:, 1], color='red', alpha=0.07)

plt.title("2017 Air Quality Forecast: SARIMA vs SARIMAX (Multi-annual Analysis)", fontsize=14)
plt.ylabel("PM2.5 Concentration")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()
