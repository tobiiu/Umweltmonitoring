import pandas as pd
import numpy as np
import os
import joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)



def create_features(df):
    df = df.copy()
    df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce', utc=True)
    df = df.set_index('createdAt').sort_index()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

def difference_series(df, target):
    df[f"{target}_diff"] = df[target].diff()
    return df.dropna()

def reverse_difference(last_value, predicted_diffs):
    return np.cumsum(np.insert(predicted_diffs, 0, last_value))[1:]

def create_supervised(df, target_diff, horizon):
    X, y = [], []
    for i in range(len(df) - horizon):
        X.append(df.iloc[i].values)
        y.append([df[target_diff].iloc[i + j] for j in range(1, horizon + 1)])
    return np.array(X), np.array(y)

def train_model(df, target='Temperature', horizon=12):
    df = create_features(df)
    df = difference_series(df, target)
    df = df.select_dtypes(include=np.number)

    X, y = create_supervised(df, f"{target}_diff", horizon)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = MultiOutputRegressor(LGBMRegressor(n_estimators=100))
    model.fit(X_train, y_train)

    joblib.dump((model, df.columns.tolist(), df[target].iloc[-1]), f"{MODEL_DIR}/{target}_model.pkl")

def forecast(df, target='Temperature', steps=12):
    model_file = f"{MODEL_DIR}/{target}_model.pkl"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Modell für '{target}' nicht gefunden.")

    model, feature_columns, last_value = joblib.load(model_file)
    df = df.copy()
    df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce', utc=True)
    df = df.dropna(subset=['createdAt'])
    df = create_features(df)
    df = difference_series(df, target)
    df = df.select_dtypes(include=np.number)[feature_columns]

    last_row = df.iloc[-1].values.reshape(1, -1)
    predicted_diffs = model.predict(last_row)[0]
    predicted_values = reverse_difference(last_value, predicted_diffs)

    last_ts = df.index[-1]
    freq = df.index.to_series().diff().median()
    # Vorhersagezeitpunkte
    forecast_times = [last_ts + freq * (i + 1) for i in range(len(predicted_values))]

    # Sicherheitscheck
    if len(forecast_times) != len(predicted_values):
        raise ValueError("Vorhersagezeitpunkte und Werte haben unterschiedliche Längen!")

    # DataFrame erzeugen
    return pd.DataFrame({'timestamp': forecast_times, f'predicted_{target}': predicted_values})
