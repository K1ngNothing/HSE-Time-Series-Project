import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from config import SEASON_LEN, SEED

LAGS = [1, 2, 3, 4]
ROLLS = [4, 8]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag/rolling/calendar/fourier features."""
    res = df.copy()

    # Lags + diff
    groupby_series = res.groupby("id")
    res["id"] = res["id"].astype(str)
    res["time_index"] = groupby_series.cumcount()
    for lag in LAGS:
        res[f"lag_{lag}"] = groupby_series["y"].shift(lag)
    res["diff_1"] = groupby_series["y"].diff(1)

    # Rolling features
    for window_size in ROLLS:
        res[f"roll_mean_{window_size}"] = groupby_series["y"].transform(
            lambda series_values: series_values.shift(1).rolling(window_size).mean()
        )
        res[f"roll_std_{window_size}"] = groupby_series["y"].transform(
            lambda series_values: series_values.shift(1).rolling(window_size).std()
        )

    res["fourier4_sin_1"] = np.sin(2 * np.pi * res["time_index"] / SEASON_LEN)
    res["fourier4_cos_1"] = np.cos(2 * np.pi * res["time_index"] / SEASON_LEN)

    # Calendar feature
    res["year"] = res["ds"].dt.year

    return res


def make_ml_dataset(df: pd.DataFrame):
    """
    Prepares dataset for ML training.
    Also returns categorical columns to use in CatBoost.
    """
    feat = add_features(df)
    feature_cols = [
        "id",
        *[f"lag_{l}" for l in LAGS],
        "diff_1",
        *[f"roll_mean_{w}" for w in ROLLS],
        *[f"roll_std_{w}" for w in ROLLS],
        "fourier4_sin_1",
        "fourier4_cos_1",
        "year",
    ]
    feat = feat.dropna(subset=feature_cols + ["y"])
    feature_frame = feat[feature_cols]
    target = feat["y"].values
    categorical_feature_names = ["id"]
    return feature_frame, target, feature_cols, categorical_feature_names


def fit_model(features: pd.DataFrame, target, categorical_feature_names):
    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function="RMSE",
        random_seed=SEED,
        verbose=False,
    )
    categorical_feature_indices = [
        features.columns.get_loc(name) for name in categorical_feature_names
    ]
    model.fit(features, target, cat_features=categorical_feature_indices)
    return model


def forecast_recursive(
    model,
    series_train: pd.DataFrame,
    test_ds,
    feature_cols: list[str],
) -> list[float]:
    """Makes recursive forecast for fitted model"""
    y_history = series_train["y"].values.tolist()

    predictions = []
    series_id = str(series_train["id"].iloc[0])
    for ds in test_ds:
        # 1. Make features for prediction
        feats = {}
        feats["id"] = series_id

        # Lags + diff
        for lag in LAGS:
            feats[f"lag_{lag}"] = y_history[-lag]
        feats["diff_1"] = y_history[-1] - y_history[-2]

        # Rolling features
        for w in ROLLS:
            window = y_history[-w:]
            feats[f"roll_mean_{w}"] = float(np.mean(window))
            feats[f"roll_std_{w}"] = float(np.std(window))

        # Fourier features with period = 4
        time_index = len(y_history)
        feats["fourier4_sin_1"] = np.sin(2 * np.pi * time_index / 4)
        feats["fourier4_cos_1"] = np.cos(2 * np.pi * time_index / 4)

        # Calendar feature
        feats["year"] = pd.Timestamp(ds).year

        # 2. Cast to DataFrame and make prediction
        feature_frame = pd.DataFrame([feats], columns=feature_cols)
        y_hat = float(model.predict(feature_frame))

        predictions.append(y_hat)
        y_history.append(y_hat)

    return predictions
