import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
import holidays
from datetime import timedelta
import matplotlib.pyplot as plt

# === Konfiguration ===
FREQ = '10min'  # 10-Minuten-Frequenz
MODEL_DIR = 'models'
MAX_VALUE = 1e6  # Obergrenze für Feature-Werte
MIN_VALUE = -1e6
os.makedirs(MODEL_DIR, exist_ok=True)

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_prepare_data(csv_path):
    """
    Daten laden und auf 10-Minuten-Intervalle resampeln.
    
    Args:
        csv_path (str): Pfad zur CSV-Datei.
    
    Returns:
        pd.DataFrame: Resampelter und interpolierter DataFrame.
    """
    try:
        df = pd.read_csv(csv_path)
        if 'createdAt' not in df.columns:
            raise ValueError("Spalte 'createdAt' nicht gefunden.")
        
        df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce', utc=True)
        df = df.dropna(subset=['createdAt']).set_index('createdAt').sort_index()
        df.index = df.index.round(FREQ)
        
        df = df.resample(FREQ).mean().interpolate()
        df = df.clip(lower=MIN_VALUE, upper=MAX_VALUE).dropna()
        logging.info(f"Daten geladen von {csv_path}, Zeilen: {len(df)}, Spalten: {list(df.columns)}")
        return df
    except Exception as e:
        logging.error(f"Fehler beim Laden der Daten: {str(e)}")
        raise

def check_stationarity(series, target):
    """
    Prüfen, ob eine Zeitreihe stationär ist (Augmented Dickey-Fuller Test).
    
    Args:
        series (pd.Series): Zeitreihe.
        target (str): Name der Zielvariable.
    
    Returns:
        bool: True, wenn stationär (p-Wert < 0.05), sonst False.
    """
    try:
        result = adfuller(series.dropna())
        p_value = result[1]
        is_stationary = p_value < 0.05
        logging.info(f"Stationaritätsprüfung für {target}: p-Wert={p_value:.4f}, stationär={is_stationary}")
        return is_stationary
    except Exception as e:
        logging.warning(f"Stationaritätsprüfung für {target} fehlgeschlagen: {str(e)}")
        return False

def remove_outliers(df, target, z_thresh=4.0):
    """
    Ausreißer basierend auf Z-Score für die Zielspalte entfernen.
    
    Args:
        df (pd.DataFrame): Eingabe-DataFrame.
        target (str): Name der Zielspalte.
        z_thresh (float): Z-Score-Schwelle.
    
    Returns:
        pd.DataFrame: DataFrame ohne Ausreißer.
    """
    try:
        z_scores = (df[target] - df[target].mean()) / df[target].std()
        df_clean = df[np.abs(z_scores) < z_thresh]
        logging.info(f"Ausreißer für {target} entfernt, ursprüngliche Zeilen: {len(df)}, verbleibende Zeilen: {len(df_clean)}")
        if len(df) - len(df_clean) > 0:
            outliers = df[np.abs(z_scores) >= z_thresh]
            logging.info(f"Entfernte Ausreißer: {len(outliers)} Zeilen, Beispiel: {outliers[target].head().to_dict()}")
        return df_clean
    except Exception as e:
        logging.error(f"Fehler beim Entfernen von Ausreißern für {target}: {str(e)}")
        raise

def detrend_series(series):
    """
    Trend aus einer Zeitreihe entfernen (STL-Dekomposition).
    
    Args:
        series (pd.Series): Eingabe-Zeitreihe.
    
    Returns:
        pd.Series: Detrendete Zeitreihe (Residuen).
    """
    try:
        stl = STL(series, period=144, robust=True)  # 24h * 6 = 144 für 10min-Intervalle
        result = stl.fit()
        return result.resid
    except Exception as e:
        logging.warning(f"STL-Dekomposition fehlgeschlagen: {str(e)}. Differenzierte Reihe wird zurückgegeben.")
        return series.diff().fillna(0)

def create_time_features(df):
    """
    Zeitbasierte Features mit Saisonalität erstellen.
    
    Args:
        df (pd.DataFrame): Eingabe-DataFrame mit DatetimeIndex.
    
    Returns:
        pd.DataFrame: DataFrame mit zusätzlichen Zeit-Features.
    """
    try:
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['trend'] = np.clip((df.index - df.index[0]).total_seconds() / 3600.0, MIN_VALUE, MAX_VALUE)
        
        try:
            de_holidays = holidays.Germany(years=range(df.index.year.min(), df.index.year.max() + 1))
            # Konvertiere Feiertage zu timezone-naiven Datumsobjekten
            holiday_dates = pd.to_datetime(list(de_holidays.keys())).tz_localize(None)
            df['is_holiday'] = df.index.normalize().tz_localize(None).isin(holiday_dates).astype(int)
            logging.info(f"Feiertage erkannt: {df['is_holiday'].sum()} Feiertags-Zeitstempel")
        except Exception as e:
            logging.warning(f"Fehler beim Initialisieren von Feiertagen: {str(e)}. Setze is_holiday auf 0.")
            df['is_holiday'] = 0
        
        return df
    except Exception as e:
        logging.error(f"Fehler beim Erstellen von Zeit-Features: {str(e)}")
        raise

def create_lag_features(df, target, lags=[1, 2, 3, 6, 12, 24, 48, 72, 144]):
    """
    Lag-Features für die Zielvariable erstellen.
    
    Args:
        df (pd.DataFrame): Eingabe-DataFrame.
        target (str): Name der Zielspalte.
        lags (list): Liste der Lag-Perioden (in 10-Minuten-Intervallen).
    
    Returns:
        pd.DataFrame: DataFrame mit zusätzlichen Lag-Features.
    """
    for lag in lags:
        df[f'lag_{lag}'] = df[target].shift(lag)
    return df

def create_aggregate_features(df, target):
    """
    Aggregierte Tages- und Wochen-Features erstellen.
    
    Args:
        df (pd.DataFrame): Eingabe-DataFrame.
        target (str): Name der Zielspalte.
    
    Returns:
        pd.DataFrame: DataFrame mit zusätzlichen aggregierten Features.
    """
    df[f'{target}_day_mean'] = df[target].shift(144).rolling(144).mean()
    df[f'{target}_week_mean'] = df[target].shift(144*7).rolling(144*7).mean()
    return df

def prepare_features(df, target):
    """
    Features für das Modellieren vorbereiten mit Stationarität und robusten Features.
    
    Args:
        df (pd.DataFrame): Eingabe-DataFrame.
        target (str): Name der Zielspalte.
    
    Returns:
        pd.DataFrame: DataFrame mit Features.
    """
    try:
        df = df.copy()
        
        # Stationarität prüfen und detrenden, falls nicht stationär
        if not check_stationarity(df[target], target):
            logging.info(f"STL-Detrending für {target} anwenden")
            df[target] = detrend_series(df[target])
        
        df = create_time_features(df)
        df = create_lag_features(df, target)
        df = create_aggregate_features(df, target)
        
        # Werte clippen und NaN/Inf behandeln
        df = df.clip(lower=MIN_VALUE, upper=MAX_VALUE)
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        logging.info(f"Features erstellt für {target}, Zeilen: {len(df)}, Features: {list(df.columns)}")
        return df
    except Exception as e:
        logging.error(f"Fehler beim Erstellen von Features für {target}: {str(e)}")
        raise

def train_and_select_model(df, target, models=None, max_train_size=10000):
    """
    Bestes Modell trainieren und auswählen mit Kreuzvalidierung.
    
    Args:
        df (pd.DataFrame): Eingabe-DataFrame mit Features.
        target (str): Name der Zielspalte.
        models (dict): Dictionary von Modellnamen und Instanzen.
        max_train_size (int): Maximale Anzahl an Trainingszeilen.
    
    Returns:
        tuple: (best_model, scaler, feature_cols)
    """
    try:
        if models is None:
            models = {
                'ridge': Ridge(),
                'rf': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            }
        
        # Trainingsdatengröße begrenzen
        df = df.iloc[-max_train_size:] if len(df) > max_train_size else df
        
        X = df.drop(columns=[target])
        y = df[target]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        tscv = TimeSeriesSplit(n_splits=5)
        best_model = None
        best_score = float('inf')
        best_model_name = ""
        
        for name, model in models.items():
            scores = -cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_root_mean_squared_error')
            avg_score = scores.mean()
            logging.info(f"Modell {name}: RMSE={avg_score:.3f}")
            if avg_score < best_score:
                best_model = model
                best_score = avg_score
                best_model_name = name
        
        best_model.fit(X_scaled, y)
        joblib.dump((best_model, scaler, X.columns.tolist()), f"{MODEL_DIR}/{target}_model.pkl")
        logging.info(f"Modell '{best_model_name}' gespeichert für {target}, Features: {X.columns.tolist()}")
        if best_model_name == 'rf':
            importances = pd.Series(best_model.feature_importances_, index=X.columns)
            logging.info(f"Feature-Wichtigkeit für {target}: {importances.sort_values(ascending=False).to_dict()}")
        return best_model, scaler, X.columns.tolist()
    except Exception as e:
        logging.error(f"Fehler beim Trainieren des Modells für {target}: {str(e)}")
        raise

def forecast_steps(df, target, steps=24):
    """
    Vorhersage für die angegebene Anzahl an Schritten erstellen.
    
    Args:
        df (pd.DataFrame): Eingabe-DataFrame.
        target (str): Name der Zielspalte.
        steps (int): Anzahl der Vorhersageschritte.
    
    Returns:
        pd.DataFrame: DataFrame mit prognostizierten Werten.
    """
    try:
        model_path = f"{MODEL_DIR}/{target}_model.pkl"
        if not os.path.exists(model_path):
            logging.info(f"Kein Modell für {target} gefunden. Neues Modell wird trainiert.")
            df_clean = remove_outliers(df, target)
            df_feat = prepare_features(df_clean, target)
            train_and_select_model(df_feat, target)
        
        model, scaler, feature_cols = joblib.load(model_path)
        df_feat = prepare_features(df.copy(), target)
        
        predictions = []
        timestamps = []
        current_time = df_feat.index[-1]
        recent_data = df_feat.tail(144*7).copy()  # Letzte Woche für Lags
        
        # Mittelwerte der Nicht-Zielspalten speichern
        non_target_cols = [col for col in df.columns if col != target and col in feature_cols]
        col_means = df[non_target_cols].mean()
        
        for i in range(steps):
            current_time += pd.Timedelta(FREQ)
            
            # Neues DataFrame mit feature_cols erstellen
            new_row_df = pd.DataFrame(index=[current_time], columns=feature_cols)
            
            # Nicht-Zielspalten mit Mittelwerten füllen
            for col in non_target_cols:
                new_row_df[col] = col_means.get(col, 0)
            
            # Zeit-Features aktualisieren
            new_row_df['hour'] = current_time.hour
            new_row_df['month'] = current_time.month
            new_row_df['hour_sin'] = np.sin(2 * np.pi * new_row_df['hour'] / 24)
            new_row_df['hour_cos'] = np.cos(2 * np.pi * new_row_df['hour'] / 24)
            new_row_df['month_sin'] = np.sin(2 * np.pi * new_row_df['month'] / 12)
            new_row_df['month_cos'] = np.cos(2 * np.pi * new_row_df['month'] / 12)
            new_row_df['trend'] = np.clip((current_time - df_feat.index[0]).total_seconds() / 3600.0, MIN_VALUE, MAX_VALUE)
            new_row_df['is_holiday'] = int(current_time.normalize().tz_localize(None) in holidays.Germany(years=current_time.year))
            
            # Lag- und Aggregat-Features aktualisieren
            for lag in [1, 2, 3, 6, 12, 24, 48, 72, 144]:
                lag_idx = len(recent_data) - lag
                if lag_idx >= 0:
                    new_row_df[f'lag_{lag}'] = recent_data.iloc[lag_idx][target]
                else:
                    new_row_df[f'lag_{lag}'] = predictions[-lag] if len(predictions) >= lag else recent_data.iloc[-1][target]
            
            lag_values = [new_row_df[f'lag_{i}'].iloc[0] for i in [1, 2, 3, 6, 12, 24, 48, 72, 144] if f'lag_{i}' in new_row_df]
            new_row_df[f'{target}_day_mean'] = np.mean(lag_values[-144:]) if len(lag_values) >= 144 else recent_data.iloc[-1][target]
            new_row_df[f'{target}_week_mean'] = np.mean(lag_values[-144*7:]) if len(lag_values) >= 144*7 else recent_data.iloc[-1][target]
            
            # Eingabe für Vorhersage vorbereiten
            new_row = new_row_df.iloc[0]
            missing_cols = [col for col in feature_cols if col not in new_row.index]
            if missing_cols:
                logging.error(f"Fehlende Spalten in new_row: {missing_cols}")
                raise KeyError(f"Fehlende Spalten in new_row: {missing_cols}")
            
            input_df = pd.DataFrame([new_row[feature_cols]])
            input_df = input_df.clip(lower=MIN_VALUE, upper=MAX_VALUE)
            if np.any(np.isinf(input_df)) or np.any(np.isnan(input_df)):
                logging.error(f"Ungültige Werte in input_df für {target}: {input_df}")
                raise ValueError("Eingabe enthält NaN oder Unendlich")
            
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            pred = np.clip(pred, MIN_VALUE, MAX_VALUE)
            
            predictions.append(pred)
            timestamps.append(current_time)
            
            # recent_data aktualisieren
            new_row_df[target] = pred
            recent_data = pd.concat([recent_data, new_row_df[[target] + [col for col in feature_cols if col in new_row_df.columns]]])
            recent_data = recent_data.iloc[-144*7:]  # Letzte Woche behalten
            
            logging.debug(f"Vorhersage für Schritt {i+1}: {pred}, input_df: {input_df.iloc[0].to_dict()}")
        
        forecast_df = pd.DataFrame({
            'timestamp': timestamps,
            f'forecast_{target}': predictions
        })
        logging.info(f"Vorhersage erstellt für {target}, Schritte: {steps}, Bereich: {min(predictions):.2f} bis {max(predictions):.2f}")
        return forecast_df
    except Exception as e:
        logging.error(f"Fehler beim Erstellen der Vorhersage für {target}: {str(e)}")
        raise

def plot_forecast(df, forecast_df, target):
    """
    Historische und prognostizierte Daten mit Variabilitätsindikatoren plotten.
    
    Args:
        df (pd.DataFrame): Historischer DataFrame.
        forecast_df (pd.DataFrame): Prognostizierter DataFrame.
        target (str): Name der Zielspalte.
    """
    try:
        plt.figure(figsize=(12, 6))
        # Letzte 1000 Punkte der historischen Daten plotten
        historical = df[target].tail(1000)
        plt.plot(historical.index, historical, label='Historisch', color='blue')
        # Vorhersage plotten
        forecast_values = forecast_df[f'forecast_{target}']
        plt.plot(forecast_df['timestamp'], forecast_values, label='Vorhersage', color='green', linestyle='--')
        # Variabilitätsindikatoren hinzufügen
        if len(forecast_values) > 1:
            std = np.std(forecast_values)
            plt.fill_between(
                forecast_df['timestamp'],
                forecast_values - std,
                forecast_values + std,
                color='green', alpha=0.1, label='Vorhersage ±1 STD'
            )
        plt.title(f'{target} Vorhersage')
        plt.xlabel('Zeit')
        plt.ylabel(target)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        logging.info(f"Vorhersage geplottet für {target}, Vorhersage-Std: {np.std(forecast_values):.2f}")
    except Exception as e:
        logging.error(f"Fehler beim Plotten der Vorhersage für {target}: {str(e)}")
        raise