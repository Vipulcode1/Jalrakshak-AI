import pandas as pd
import numpy as np
from pathlib import Path


# =====================================
# PROJECT DIRECTORIES
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"


# Create folders automatically
for folder in [
    DATA_DIR,
    PROCESSED_DIR,
    MODEL_DIR,
    REPORT_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


# =====================================
# READ CSV FILE
# =====================================

def read_csv_file(filename):

    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return pd.read_csv(
        path,
        low_memory=False
    )


# =====================================
# NORMALIZE COLUMN NAMES
# =====================================

def normalize_columns(df):

    df = df.copy()

    df.columns = [
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for col in df.columns
    ]

    return df


# =====================================
# FIND COLUMN
# =====================================

def find_column(df, keywords):

    for col in df.columns:

        for keyword in keywords:

            if keyword in col:
                return col

    return None


# =====================================
# FIND DATE COLUMN
# =====================================

def find_date_column(df):

    return find_column(
        df,
        [
            "data_acquisition_time",
            "timestamp",
            "datetime",
            "date"
        ]
    )


# =====================================
# FIND STATION COLUMN
# =====================================

def find_station_column(df):

    return find_column(
        df,
        [
            "station",
            "site_name",
            "location"
        ]
    )


# =====================================
# FIND WATER LEVEL COLUMN
# =====================================

def find_level_column(df):

    return find_column(
        df,
        [
            "river_water_level_telemetry_hourly",
            "water_level",
            "waterlevel",
            "level_telemetry"
        ]
    )


# =====================================
# FIND RAINFALL COLUMN
# =====================================

def find_rainfall_column(df):

    return find_column(
        df,
        [
            "rainfall",
            "rain_mm",
            "precipitation"
        ]
    )


# =====================================
# FIND DISCHARGE COLUMN
# =====================================

def find_discharge_column(df):

    return find_column(
        df,
        [
            "discharge",
            "flow"
        ]
    )


# =====================================
# PARSE DATE
# =====================================

def parse_dates(series):

    return pd.to_datetime(
        series,
        errors="coerce",
        dayfirst=True
    )


# =====================================
# PREPARE STATION DATA
# =====================================

def prepare_station_data(
    df,
    station_name,
    value_keywords
):

    df = normalize_columns(df)

    station_col = find_station_column(df)

    date_col = find_date_column(df)

    value_col = find_column(
        df,
        value_keywords
    )

    if station_col is None:
        raise ValueError(
            "Station column not found. "
            f"Available columns: {list(df.columns)}"
        )

    if date_col is None:
        raise ValueError(
            "Date column not found. "
            f"Available columns: {list(df.columns)}"
        )

    if value_col is None:
        raise ValueError(
            "Value column not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Parse dates
    df[date_col] = parse_dates(
        df[date_col]
    )

    # Convert values to numeric
    df[value_col] = pd.to_numeric(
        df[value_col],
        errors="coerce"
    )

    # Filter station
    df = df[
        df[station_col]
        .astype(str)
        .str.contains(
            station_name,
            case=False,
            na=False
        )
    ].copy()

    # Remove invalid records
    df = df.dropna(
        subset=[
            date_col,
            value_col
        ]
    )

    # Sort by date
    df = df.sort_values(
        date_col
    )

    if df.empty:
        raise ValueError(
            f"No valid records found for station: {station_name}"
        )

    return df, date_col, value_col


# =====================================
# MODEL EVALUATION METRICS
# =====================================

def calculate_metrics(
    actual,
    predicted
):

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


print("JALRAKSHAK AI common module loaded successfully!")