import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

def map_weather_to_condition(weather):
    if pd.isna(weather):
        return np.nan
    weather = weather.lower()
    if 'thunderstorms' in weather and ('heavy' in weather or 'moderate' in weather):
        return 9  # Severe positive (warm storms)
    elif 'thunderstorms' in weather:
        return 7  # Thunderstorms
    elif 'moderate' in weather or 'heavy' in weather or 'blowing' in weather:
        return 8  # Moderate/severe
    elif 'freezing' in weather or 'ice' in weather or 'pellets' in weather:
        return -6  # Freezing/ice
    elif 'snow' in weather:
        return -5  # Snow-related
    elif 'rain' in weather:
        return 4  # Rain-related
    elif 'drizzle' in weather:
        return 3  # Drizzle
    elif 'fog' in weather or 'haze' in weather:
        return -1  # Fog/haze
    elif 'cloudy' in weather:
        return 1  # Cloudy
    elif 'clear' in weather:
        return 0  # Clear/neutral
    else:
        return 0  # Default neutral

        
# Load data
df = pd.read_csv('weather_data_labeled.csv')
print("Loaded columns:", df.columns.tolist())  # Debug
df.columns = df.columns.str.strip()  # Strip spaces
df['condition'] = df['Weather'].apply(map_weather_to_condition)

df = df[['temperature', 'humidity', 'pressure', 'condition']]  # Select only numerics
df.rename(columns={'Temp_C': 'temperature', 'Rel Hum_%': 'humidity', 'Press_kPa': 'pressure','Weather':'condition'}, inplace=True)

df['pressure'] = df['pressure'] * 10

imputer = SimpleImputer(strategy='mean')
df[['temperature', 'humidity', 'pressure', 'condition']] = imputer.fit_transform(
    df[['temperature', 'humidity', 'pressure', 'condition']]
)
df = df[(df['temperature'] > -50) & (df['temperature'] < 60)]
df = df[(df['humidity'] >= 0) & (df['humidity'] <= 100)]
df = df[(df['pressure'] > 800) & (df['pressure'] < 1100)]  
X = df[['temperature', 'humidity', 'pressure']]
y = df['condition']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save preprocessed data (optional)
pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

# Save scaler for later use
import joblib
joblib.dump(scaler, 'scaler.pkl')