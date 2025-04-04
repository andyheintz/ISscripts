import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

file_path = "/Users/andyheintz/Desktop/ISenergy/data/mergedTexSolar.csv"
df = pd.read_csv(file_path, header=1)
df = df.loc[:, ~df.columns.str.contains('Unnamed')]
print(df.columns)

df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce')
df = df.drop(columns=['Year', 'Day', 'Hour', 'Minute'])  
df = df.sort_values(by='Timestamp')

df['GHI'] = pd.to_numeric(df['GHI'], errors='coerce')
for col in ['Temperature', 'Wind Speed']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Time_Index'] = np.arange(len(df))
T_year, T_day, K_high = 8760, 24, 15 
for k in range(1, K_high + 1):
    df[f'sin_{k}_year'] = np.sin(2 * np.pi * k * df['Time_Index'] / T_year)
    df[f'cos_{k}_year'] = np.cos(2 * np.pi * k * df['Time_Index'] / T_year)
    df[f'sin_{k}_day'] = np.sin(2 * np.pi * k * df['Time_Index'] / T_day)
    df[f'cos_{k}_day'] = np.cos(2 * np.pi * k * df['Time_Index'] / T_day)

df = df.dropna(subset=['GHI', 'Temperature', 'Wind Speed'])

X = df[[col for col in df.columns if col.startswith('sin_') or col.startswith('cos_')]].copy()
#X['Temperature'] = df['Temperature']
#X['Wind Speed'] = df['Wind Speed']
#X['Month'] = df['Month']
y = df['GHI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=25, random_state=42)
rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
model_summary = {
    "Train R² Score": r2_score(y_train, y_train_pred),
    "Test R² Score": r2_score(y_test, y_test_pred),
    "Test MSE": mean_squared_error(y_test, y_test_pred),
    "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    "Test MAE": mean_absolute_error(y_test, y_test_pred),
}
n, k = len(y_test), X_test.shape[1]
rse = np.sqrt(np.sum((y_test - y_test_pred) ** 2) / (n - k - 1))

print("\nModel Summary:")
for key, value in model_summary.items():
    print(f"{key}: {value:.2f}")
print(f"Test RSE: {rse:.2f}")
test_residuals = y_test - y_test_pred

rse = np.sqrt(np.sum(test_residuals ** 2) / (n - k - 1))

mean_ghi = y_test.mean()

rse_percent_of_mean = (rse / mean_ghi) * 100

print(f"\nTest RSE: {rse:.2f}")
print(f"Mean GHI: {mean_ghi:.2f} W/m²")
print(f"RSE as % of Mean GHI: {rse_percent_of_mean:.2f}%")

#Residual Plots
residuals = y_test - y_test_pred
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.scatterplot(x=y_test_pred, y=residuals, ax=axes[0, 0], alpha=0.5)
axes[0, 0].axhline(0, color='red', linestyle='dashed')
axes[0, 0].set_title('Residuals vs. Fitted')
sns.histplot(residuals, kde=True, bins=30, ax=axes[0, 1])
axes[0, 1].set_title('Histogram of Residuals')
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('QQ Plot')
first_feature = X_test.columns[0]
sns.scatterplot(x=X_test[first_feature], y=residuals, ax=axes[1, 1], alpha=0.5)
axes[1, 1].axhline(0, color='red', linestyle='dashed')
axes[1, 1].set_title('Residuals vs. First Feature')
plt.tight_layout()
plt.show()

#Prediction vs. Actual Scatterplot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed', label='Ideal Fit')  # Perfect prediction line
plt.xlabel('Actual GHI (W/m²)')
plt.ylabel('Predicted GHI (W/m²)')
plt.title('Prediction vs. Actual GHI in Fort Stockton, TX')
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,5))
feature_importance[:10].plot(kind='barh', color='c')
plt.xlabel('Importance Score')
plt.title('Top 10 Feature Importances (2021)')
plt.gca().invert_yaxis()
plt.show()

last_timestamp = df['Timestamp'].max()
future_timestamps = pd.date_range(start=last_timestamp, periods=8760, freq='h')
future_df = pd.DataFrame({'Timestamp': future_timestamps})
future_df['Time_Index'] = np.arange(len(df), len(df) + len(future_df))
for k in range(1, K_high + 1):
    future_df[f'sin_{k}_year'] = np.sin(2 * np.pi * k * future_df['Time_Index'] / T_year)
    future_df[f'cos_{k}_year'] = np.cos(2 * np.pi * k * future_df['Time_Index'] / T_year)
    future_df[f'sin_{k}_day'] = np.sin(2 * np.pi * k * future_df['Time_Index'] / T_day)
    future_df[f'cos_{k}_day'] = np.cos(2 * np.pi * k * future_df['Time_Index'] / T_day)
future_df['Month'] = future_df['Timestamp'].dt.month

monthly_meteo = df.groupby('Month')[['Temperature', 'Wind Speed']].mean().reset_index()
future_df['Month'] = future_df['Month'].astype(int)
monthly_meteo['Month'] = monthly_meteo['Month'].astype(int)

future_df = future_df.merge(monthly_meteo, on='Month', how='left')

# Predict GHI
X_future = future_df[X.columns]
future_df['Predicted_GHI'] = rf_model.predict(X_future)

# Save
results_folder = "/Users/andyheintz/Desktop/ISenergy/results/TexGHIResults"
os.makedirs(results_folder, exist_ok=True)
future_predictions_file = os.path.join(results_folder, "GHI_PredictionsTX.csv")
future_df[['Timestamp', 'Predicted_GHI']].to_csv(future_predictions_file, index=False)
print(f"Future predictions saved to: {future_predictions_file}")

# GHI over time
plt.figure(figsize=(10,5))
plt.plot(future_df['Timestamp'], future_df['Predicted_GHI'], color='orange')
plt.xlabel('Date')
plt.ylabel('GHI (W/m²)')
plt.title('Predicted GHI for 2021 in Fort Stockton, TX')
plt.grid(True)
plt.show()

# Monthly Average GHI
future_df['Year'] = future_df['Timestamp'].dt.year
future_df['Month'] = future_df['Timestamp'].dt.to_period('M')
future_df_2021 = future_df[future_df['Year'] == 2021]
monthly_avg_ghi = future_df_2021.groupby('Month')['Predicted_GHI'].mean()
monthly_avg_ghi.plot(kind='bar', color='green', alpha=0.7, figsize=(12,6))
plt.ylabel('Average GHI (W/m²)')
plt.title('Monthly Average Predicted GHI for 2021 in Fort Stockton, TX')
plt.xticks(rotation=45)
plt.show()

# Computing solar energy production

panel_area = 2 #(2m x 1m panel)
panel_efficiency = 0.24 
performance_ratio = 0.8 

# power output equation

future_df_2021['Predicted_Solar_Energy_kWh'] = (
    future_df_2021['Predicted_GHI'] * panel_area * panel_efficiency * performance_ratio
) / 1000  # Convert W to kW

future_df_2021['Predicted_Solar_Energy_kWh'] = future_df_2021['Predicted_Solar_Energy_kWh'].clip(lower=0)

future_df_2021['Month'] = future_df_2021['Timestamp'].dt.to_period('M')

monthly_energy_output_kWh = future_df_2021.groupby('Month')['Predicted_Solar_Energy_kWh'].sum()
monthly_energy_output_MWh = monthly_energy_output_kWh / 1000  # Convert to MWh

# Monthly Energy Output
plt.figure(figsize=(12,6))
monthly_energy_output_MWh.plot(kind='bar', color='red', alpha=0.7)
plt.xlabel("Month")
plt.ylabel("Total Energy Output (MWh)")
plt.title("Total Monthly Predicted Solar Energy Output in 2021 for Fort Stockton, TX")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

total_annual_energy_kWh = future_df_2021['Predicted_Solar_Energy_kWh'].sum()
total_annual_energy_MWh = total_annual_energy_kWh / 1000  # Convert to MWh

print(f"\nTotal Solar Energy Production for 2021: {total_annual_energy_kWh:.2f} kWh ({total_annual_energy_MWh:.2f} MWh)")

solar300k = (total_annual_energy_MWh*300000)
solar600k = (total_annual_energy_MWh*600000)

print(f"\nTotal Solar Energy Production for 300,000 panels in 2021: {solar300k:.2f}")
print(f"\nTotal Solar Energy Production for 600,000 panels in 2021: {solar600k:.2f}")

panel_area_m2 = 2
panel_efficiency = 0.24
irradiance_stc = 1000  # W/m²
panel_capacity_kW = (panel_area_m2 * irradiance_stc * panel_efficiency) / 1000  # kW

total_capacity_MWac = (panel_capacity_kW * 600_000) / 1000  # Convert kW to MW
print(f"\nEstimated MWac Capacity for 600,000 panels: {total_capacity_MWac:.2f} MWac")

rse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"\nTest RSE: {rse:.2f} W/m²")

mean_ghi_full = y_test.mean()
print(f"Mean GHI (Full Test Set): {mean_ghi_full:.2f} W/m²")

rse_percent_full = (rse / mean_ghi_full) * 100
print(f"RSE as % of Mean GHI (Full Test Set): {rse_percent_full:.2f}%")

daytime_mask = y_test > 0
y_test_daytime = y_test[daytime_mask]
y_test_pred_daytime = y_test_pred[daytime_mask]

# daytime only
rse_daytime = np.sqrt(mean_squared_error(y_test_daytime, y_test_pred_daytime))
print(f"\nDaytime Test RSE: {rse_daytime:.2f} W/m²")

mean_ghi_daytime = y_test_daytime.mean()
print(f"Mean GHI (Daytime Only): {mean_ghi_daytime:.2f} W/m²")

rse_percent_daytime = (rse_daytime / mean_ghi_daytime) * 100
print(f"RSE as % of Mean GHI (Daytime Only): {rse_percent_daytime:.2f}%")