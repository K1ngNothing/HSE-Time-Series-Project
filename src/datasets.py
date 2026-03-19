import pandas as pd
from numpy.random import Generator
from typing import TYPE_CHECKING

from config import DATASET_FOLDER, HORIZON, SEED

if TYPE_CHECKING:
    from numpy.random import Generator


def load_m4_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads M4-Quarterly dataset

    Returns:
        df_train: train dataset
        df_test: test dataset
        df_info: table with meta information
    """

    df_train = pd.read_csv(DATASET_FOLDER / "Quarterly-train.csv")
    df_test = pd.read_csv(DATASET_FOLDER / "Quarterly-test.csv")
    df_info = pd.read_csv(DATASET_FOLDER / "M4-info.csv")
    return df_train, df_test, df_info


def select_aligned_series(
        n_series: int,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_info: pd.DataFrame,
        rng: Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select n_series random series from aligned group (same start & end) and return new dataframes.

    Args:
        n_series: number of series to sample.
        df_train: df_train in WIDE format (like in M4 dataset)
        df_test: df_test in WIDE format (like in M4 dataset)
        df_info: info dataframe in M4 format
        rng: external random number generator

    Returns:
        (df_train_aligned, df_test_aligned, df_info_aligned): datasets with aligned series
    """

    # train length per series
    train_len = df_train.drop(columns=["V1"]).notna().sum(axis=1)
    train_len.index = df_train.index

    # parse start date
    start_map = df_info.set_index("M4id")["StartingDate"]
    start_map = pd.to_datetime(start_map, dayfirst=True, errors="coerce")

    # build meta per id
    meta = df_train[["V1"]].copy()
    meta["train_len"] = train_len.values
    meta["start"] = meta["V1"].map(start_map)
    meta["end"] = meta["start"].dt.to_period("Q") + (meta["train_len"] + HORIZON - 1)

    # choose a random group with enough series (same start & end)
    groups = meta.groupby(["start", "end"]).size()
    eligible_groups = groups[groups >= n_series]
    assert not eligible_groups.empty, "No aligned groups found with enough series"
    chosen_start, chosen_end = eligible_groups.sample(n=1, random_state=SEED).index[0]
    chosen_group_ids = meta[(meta["start"] == chosen_start) & (meta["end"] == chosen_end)]["V1"]

    chosen_series = rng.choice(chosen_group_ids.values, size=n_series, replace=False)
    df_train_aligned = df_train[df_train["V1"].isin(chosen_series)].copy()
    df_test_aligned = df_test[df_test["V1"].isin(chosen_series)].copy()
    df_info_aligned = df_info[df_info["M4id"].isin(chosen_series)].copy()

    print(f"Chosen aligned series group size: {len(chosen_group_ids)}")
    print(f"Start and end dates: {chosen_start}, {chosen_end}")

    return df_train_aligned, df_test_aligned, df_info_aligned


def _wide_to_long(
        df_wide: pd.DataFrame,
        df_info: pd.DataFrame,
        offset_extra: pd.Series | None = None) -> pd.DataFrame:
    """
    Transforms dataframe from wide M4 format to standard long.

    Args:
        df_wide: df to transform
        df_info: df with meta information
        offset_extra: (optional) additional offset added to all timestamps

    Returns:
        df_long: transformed long dataframe
    """
    id_col = "V1"
    value_cols = [c for c in df_wide.columns if c != id_col]
    df_long = (
        df_wide.melt(id_vars=id_col, value_vars=value_cols, var_name="ds", value_name="y")
          .dropna(subset=["y"])
    )

    # Retrieve starting dates
    start_map = df_info.set_index("M4id")["StartingDate"]
    start_map = pd.to_datetime(start_map, dayfirst=True)

    # Base offset from column index
    offset = (df_long["ds"].str[1:].astype(int) - 2).to_numpy()
    if offset_extra is not None:
        offset = offset + df_long[id_col].map(offset_extra).to_numpy()

    # Add timestamps
    start_period = pd.PeriodIndex(df_long[id_col].map(start_map), freq="Q")
    periods = pd.PeriodIndex(start_period, freq="Q") + offset
    df_long["ds"] = periods.to_timestamp(how="end")

    return df_long.rename(columns={id_col: "id"})


def wide_to_long_train(
        df_train: pd.DataFrame,
        df_info: pd.DataFrame) -> pd.DataFrame:
    """Transform df_train from wide to long format"""
    return _wide_to_long(df_train, df_info)


def wide_to_long_test(
        df_test: pd.DataFrame,
        df_train_long: pd.DataFrame,
        df_info: pd.DataFrame) -> pd.DataFrame:
    """
    Transform df_test from wide to long format.
    Accounts for a fact that test timestamps should go after train
    """
    train_len = df_train_long.groupby("id")["y"].size()
    df_test_long = _wide_to_long(df_test, df_info, offset_extra=train_len)

    # Test timestamp alignment for tain and test dataframes
    train_end = df_train_long.groupby("id")["ds"].max()
    test_start = df_test_long.groupby("id")["ds"].min()
    expected_start = train_end.dt.to_period("Q") + 1
    actual_start = test_start.dt.to_period("Q")
    assert (actual_start == expected_start).all(), "Train/test s are not aligned"

    return df_test_long
