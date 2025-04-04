import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

file_path = "/Users/andyheintz/Desktop/ISenergy/data/VaughnWinSpeed100.csv"
df = pd.read_csv(file_path)
df = df.rename(columns={"wind speed at 100m (m/s)": "WindSpeed100m"})
df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce')
df = df.sort_values(by='Timestamp')

numeric_cols = ['WindSpeed100m', 'air temperature at 100m (C)', 'air pressure at 100m (Pa)', 'vertical windspeed at 120m (m/s)']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Time_Index'] = np.arange(len(df))
T_year = 8760  # Annual cycle in hours
T_day = 24     # Daily cycle in hours
K_high = 10    # Fourier order

for k in range(1, K_high + 1):
    df[f'sin_{k}_year'] = np.sin(2 * np.pi * k * df['Time_Index'] / T_year)
    df[f'cos_{k}_year'] = np.cos(2 * np.pi * k * df['Time_Index'] / T_year)
    df[f'sin_{k}_day'] = np.sin(2 * np.pi * k * df['Time_Index'] / T_day)
    df[f'cos_{k}_day'] = np.cos(2 * np.pi * k * df['Time_Index'] / T_day)

X = df[[col for col in df.columns if col.startswith('sin_') or col.startswith('cos_')]].copy()
X['AirTemperature100m'] = df['air temperature at 100m (C)']
X['AirPressure100m'] = df['air pressure at 100m (Pa)']
X['VerticalWindSpeed120m'] = df['vertical windspeed at 120m (m/s)']
y = df['WindSpeed100m']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=25, random_state=42)
rf_model.fit(X_train, y_train)

def wind_to_power(wind_speed):
    """Apply the new logistic equation for power output based on wind speed."""
    numerator = 4301.07782
    exponent = - (0.83286 * wind_speed - 6.17436)
    return numerator / (1 + np.exp(exponent))

last_timestamp = df["Timestamp"].max()
future_timestamps = pd.date_range(start=last_timestamp, periods=8760, freq='H')

future_df = pd.DataFrame({'Timestamp': future_timestamps})
future_df['Time_Index'] = np.arange(len(df), len(df) + len(future_df))

for k in range(1, 11):  
    future_df[f'sin_{k}_year'] = np.sin(2 * np.pi * k * future_df['Time_Index'] / 8760)
    future_df[f'cos_{k}_year'] = np.cos(2 * np.pi * k * future_df['Time_Index'] / 8760)
    future_df[f'sin_{k}_day'] = np.sin(2 * np.pi * k * future_df['Time_Index'] / 24)
    future_df[f'cos_{k}_day'] = np.cos(2 * np.pi * k * future_df['Time_Index'] / 24)

future_df['AirTemperature100m'] = df['air temperature at 100m (C)'].mean()
future_df['AirPressure100m'] = df['air pressure at 100m (Pa)'].mean()
future_df['VerticalWindSpeed120m'] = df['vertical windspeed at 120m (m/s)'].mean()

X_future = future_df[X.columns]

future_df['Predicted_WindSpeed100m'] = rf_model.predict(X_future)

future_df['Predicted_PowerOutput'] = future_df['Predicted_WindSpeed100m'].apply(wind_to_power)

results_folder = "/Users/andyheintz/Desktop/ISenergy/results"
os.makedirs(results_folder, exist_ok=True)
future_predictions_file = os.path.join(results_folder, "future_wind_predictions1.csv")
#future_df[['Timestamp', 'Predicted_WindSpeed100m', 'Predicted_PowerOutput']].to_csv(future_predictions_file, index=False)

print(f"Future predictions saved to: {future_predictions_file}")

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

model_summary = {
    "Train R² Score": r2_score(y_train, rf_model.predict(X_train)),
    "Test R² Score": r2_score(y_test, rf_model.predict(X_test)),
    "Test MSE": mean_squared_error(y_test, rf_model.predict(X_test)),
    "Test RMSE": np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test))),
    "Test MAE": mean_absolute_error(y_test, rf_model.predict(X_test)),
}

n = len(y_test)
k = X_test.shape[1]

print("\nModel Summary:")
for key, value in model_summary.items():
    print(f"{key}: {value}")

rse = np.sqrt(np.sum((y_test - y_test_pred) ** 2) / (n - k - 1))
print(f"Test RSE: {rse:.3f}")

print(f"Wind Speed Range: {df['WindSpeed100m'].min()} to {df['WindSpeed100m'].max()} m/s")
mean_wind_speed = df['WindSpeed100m'].mean()
print(f"Mean Wind Speed: {mean_wind_speed:.3f}")
percent_rse_of_mean = (rse / mean_wind_speed) * 100
print(f"RSE as % of Mean Wind Speed: {percent_rse_of_mean:.3f}%")

future_df['Month'] = future_df['Timestamp'].dt.to_period('M')
monthly_avg_power = future_df.groupby('Month')['Predicted_PowerOutput'].mean()
monthly_avg_power = monthly_avg_power[monthly_avg_power.index >= "2021-01"]

plt.figure(figsize=(10,5))
plt.plot(future_df['Timestamp'], future_df['Predicted_WindSpeed100m'], label="Predicted Wind Speed", color='b', alpha=0.6)
plt.xlabel("Date")
plt.ylabel("Wind Speed (m/s)")
plt.title("Predicted Wind Speeds for 2021 in Vaughn, NM")
plt.legend()
plt.grid(True)
plt.show()

capacity_factor = 0.45
future_df['Predicted_PowerOutput_Adjusted'] = future_df['Predicted_PowerOutput'] * capacity_factor

future_df['Month'] = future_df['Timestamp'].dt.to_period('M')

monthly_total_power_output = future_df.groupby('Month')['Predicted_PowerOutput_Adjusted'].sum()
monthly_total_power_output = monthly_total_power_output[monthly_total_power_output.index >= "2021-01"]
monthly_total_power_output_MWh = monthly_total_power_output / 1000  # Convert kWh to MWh

#predicted energy over the year
total_annual_energy_MWh = monthly_total_power_output_MWh.sum()

hours_per_year = 8760  # 365 * 24
capacity_factor = 0.45  

calculated_capacity_MW = total_annual_energy_MWh / (hours_per_year * capacity_factor)

print(f"Estimated Installed Capacity: {calculated_capacity_MW:.2f} MW")

plt.figure(figsize=(12,5))
monthly_total_power_output_MWh.plot(kind='bar', color='r', alpha=0.7)
plt.xlabel("Month")
plt.ylabel("Total Power Output (MWh)")
plt.title("Total Monthly Predicted Power Output in 2021 for Vaughn, NM")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(18,6))
feature_importance[:10].plot(kind='barh', color='c', alpha=0.7) 
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.title('10 Most Important Features in Random Forest Model for Wind Speeds in Vaughn, NM')
plt.gca().invert_yaxis() 
plt.show()

# Apply Capacity Factor (45%)
capacity_factor = 0.45
total_energy_kWh = future_df.loc[future_df['Timestamp'].dt.year == 2021, 'Predicted_PowerOutput'].sum()
total_energy_kWh_adjusted = total_energy_kWh * capacity_factor

total_energy_MWh = total_energy_kWh / 1000  
total_energy_MWh_adjusted = total_energy_MWh * capacity_factor

turbine_counts = [1, 25, 50, 100]
energy_output = {f"{turbines} Turbine(s) Energy (MWh)": total_energy_MWh_adjusted * turbines for turbines in turbine_counts}

print(f"\nTotal Energy Production for 1 Turbine in 2021 (with 45% CF): {total_energy_kWh_adjusted:.2f} kWh ({total_energy_MWh_adjusted:.2f} MWh)")
for turbines, energy in energy_output.items():
    print(f"{turbines}: {energy:.2f} MWh")

capacity_factor = 0.45
total_energy_kWh = future_df.loc[future_df['Timestamp'].dt.year == 2021, 'Predicted_PowerOutput'].sum()
total_energy_kWh_adjusted = total_energy_kWh * capacity_factor

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, alpha=0.5, color='b')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')  # Ideal line
plt.xlabel('Actual Wind Speed (m/s)')
plt.ylabel('Predicted Wind Speed (m/s)')
plt.title('Prediction vs. Actual Wind Speed in Vaughn, NM')
plt.grid(True)
plt.show()

# diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# residuals vs. fitted
sns.scatterplot(x=y_test_pred, y=test_residuals, ax=axes[0, 0], alpha=0.5)
axes[0, 0].axhline(0, linestyle='dashed', color='red')
axes[0, 0].set_xlabel('Predicted Wind Speed (m/s)')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs. Fitted')

#Histogram of Residuals
sns.histplot(test_residuals, kde=True, bins=30, ax=axes[0, 1])
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_title('Histogram of Residuals')

# QQ Plot
stats.probplot(test_residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('QQ Plot of Residuals')

# residuals vs first predictor
first_feature = X_test.columns[0]  
sns.scatterplot(x=X_test[first_feature], y=test_residuals, ax=axes[1, 1], alpha=0.5)
axes[1, 1].axhline(0, linestyle='dashed', color='red')
axes[1, 1].set_xlabel(f'{first_feature}')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residuals vs. First Predictor Variable')

plt.tight_layout()
plt.show()