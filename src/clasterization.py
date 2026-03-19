import numpy as np
import plotly.graph_objects as go
from tslearn.clustering import TimeSeriesKMeans

from config import SEED


def build_ts_matrix(train_long_scaled, id_col: str = "id", value_col: str = "y"):
    """Builds a matrix to use in TimeSeriesKMeans"""
    series_ids = []
    series_values = []
    lengths = train_long_scaled.groupby(id_col)[value_col].size()
    max_len = int(lengths.max())
    for sid, df_series in train_long_scaled.groupby(id_col):
        y = df_series[value_col].values
        if len(y) < max_len:
            pad_val = y[-1]
            y = np.concatenate([y, np.full(max_len - len(y), pad_val)])
        series_ids.append(sid)
        series_values.append(y)

    X_series = np.array(series_values)[:, :, None]  # (n_series, max_len, 1)
    return series_ids, X_series, max_len


def plot_elbow_curve(
    X_series: np.ndarray,
    k_values: list[int],
) -> None:
    """
    Goes through k_values, runs TimeSeriesKMeans for k clusters an elbow curve.
    Used to find the optimal number of clusters.
    """
    inertias = []
    for k in k_values:
        kmeans = TimeSeriesKMeans(
            n_clusters=k,
            metric="dtw",
            random_state=SEED,
            verbose=False,
        )
        _ = kmeans.fit_predict(X_series)
        inertias.append(kmeans.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(k_values), y=inertias, mode="lines+markers", name="DTW inertia"))
    fig.update_layout(title="Elbow plot for K", xaxis_title="K", yaxis_title="DTW inertia")
    fig.show()
