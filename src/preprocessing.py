import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_inversable(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, callable]:
    """
    Normalize each series with StandardScaler (fit on train).
    Returns scaled dfs + function to inverse transformation.
    """
    scalers: dict[str, StandardScaler] = {}
    result_train = train_df.copy()
    result_test = test_df.copy()

    for id, train_df_slice in train_df.groupby("id"):
        scaler = StandardScaler()

        # Fit scaler
        y_train = train_df_slice[["y"]].values
        scaler.fit(y_train)
        scalers[id] = scaler

        # Scale data
        mask_train = result_train["id"] == id
        mask_test = result_test["id"] == id
        y_test = result_test.loc[mask_test, ["y"]].values
        result_train.loc[mask_train, "y"] = scaler.transform(y_train).ravel()
        result_test.loc[mask_test, "y"] = scaler.transform(y_test).ravel()

    def inverse_scale_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        result = df.copy()
        for id, scaler in scalers.items():
            mask = result["id"] == id
            if mask.any():
                result.loc[mask, cols] = scaler.inverse_transform(result.loc[mask, cols])
        return result

    return result_train, result_test, inverse_scale_df
