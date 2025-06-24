import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

class PredictionModel:

    def __init__(self, model=None):
        if model is None:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            self.model = model

        self.features = []
        self.lat_range = (None, None)
        self.lng_range = (None, None)

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            empty_features_df = pd.DataFrame(columns=['hour', 'dayofweek', 'month', 'latitude', 'longitude'])
            return empty_features_df

        df['O_time'] = pd.to_datetime(df['O_time'], errors='coerce')
        df = df.dropna(subset=['O_time']).copy()

        if df.empty:
             empty_features_df = pd.DataFrame(columns=['hour', 'dayofweek', 'month', 'latitude', 'longitude'])
             return empty_features_df

        df['hour'] = df['O_time'].dt.hour
        df['dayofweek'] = df['O_time'].dt.dayofweek
        df['month'] = df['O_time'].dt.month
        df['latitude'] = df['O_lat']
        df['longitude'] = df['O_lng']

        self.features = ['hour', 'dayofweek', 'month', 'latitude', 'longitude']

        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            pass

        return df[self.features]

    def fit(self, historical_data: pd.DataFrame):
        print("Training model...")

        required_cols = ['O_time', 'O_lat', 'O_lng', 'O_FLAG']
        if historical_data.empty or not all(col in historical_data.columns for col in required_cols):
             print(f"Missing required columns: {required_cols}")
             self.model = None
             return

        positive_samples = historical_data[historical_data['O_FLAG'] == 1].copy()
        positive_samples['target'] = 1

        valid_lats = positive_samples['O_lat'].dropna()
        valid_lngs = positive_samples['O_lng'].dropna()

        if valid_lats.empty or valid_lngs.empty:
             print("Invalid location data")
             self.model = None
             return

        self.lat_range = (valid_lats.min(), valid_lats.max())
        self.lng_range = (valid_lngs.min(), valid_lngs.max())

        valid_times = pd.to_datetime(positive_samples['O_time'], errors='coerce').dropna()
        if valid_times.empty:
            print("Invalid time data")
            self.model = None
            return
        self.time_range = (valid_times.min(), valid_times.max())

        num_positive = len(positive_samples)
        num_negative = num_positive

        time_diff_seconds = (self.time_range[1] - self.time_range[0]).total_seconds()
        random_seconds = np.random.rand(num_negative) * time_diff_seconds
        random_times = self.time_range[0] + pd.to_timedelta(random_seconds, unit='s')

        random_lats = np.random.uniform(self.lat_range[0], self.lat_range[1], num_negative)
        random_lngs = np.random.uniform(self.lng_range[0], self.lng_range[1], num_negative)

        negative_samples = pd.DataFrame({
            'O_time': random_times,
            'O_lat': random_lats,
            'O_lng': random_lngs,
            'O_FLAG': 0,
            'target': 0
        })

        training_data = pd.concat([
            positive_samples[['O_time', 'O_lat', 'O_lng', 'target']],
            negative_samples[['O_time', 'O_lat', 'O_lng', 'target']]
        ], ignore_index=True)

        X = self._create_features(training_data)
        y = training_data['target']

        original_indices = X.index
        X = X.dropna()
        y = y.loc[X.index]

        if X.empty:
            print("Empty features after processing")
            self.model = None
            return

        print(f"Training samples: {len(X)}")
        print(f"Features: {len(self.features)}")

        try:
            self.model.fit(X, y)
            print("Model trained successfully")
        except Exception as e:
            print(f"Training failed: {e}")
            self.model = None

    def predict_daily(self, date: datetime, lat: float, lng: float) -> dict:
        """
        Predict order probabilities for all hours of a given day at a specific location.

        Args:
            date: The date to predict for (time part will be ignored)
            lat: Latitude of the location
            lng: Longitude of the location

        Returns:
            Dictionary with hours (0-23) as keys and predicted probabilities as values
        """
        if self.model is None:
            print("Model not trained")
            return {hour: np.nan for hour in range(24)}

        if not self.features:
            print("Features not defined")
            return {hour: np.nan for hour in range(24)}

        predictions = {}
        base_date = date.replace(hour=0, minute=0, second=0, microsecond=0)

        for hour in range(24):
            prediction_time = base_date + timedelta(hours=hour)

            prediction_data = pd.DataFrame({
                'O_time': [prediction_time],
                'O_lat': [lat],
                'O_lng': [lng]
            })

            try:
                X_predict = self._create_features(prediction_data)

                if list(X_predict.columns) != self.features:
                    print("Feature mismatch")
                    predictions[hour] = np.nan
                    continue

                probability = self.model.predict_proba(X_predict).squeeze()[1]
                predictions[hour] = float(probability)
            except Exception as e:
                print(f"Prediction error for hour {hour}: {e}")
                predictions[hour] = np.nan

        return predictions