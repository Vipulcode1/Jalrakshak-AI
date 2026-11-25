import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ==================================================
# 2. LOAD DATASETS
# ==================================================

print("\nLoading processed datasets...")

river = pd.read_csv(
    os.path.join(PROCESSED_DIR, "river_features.csv")
)

reservoir = pd.read_csv(
    os.path.join(PROCESSED_DIR, "reservoir_features.csv")
)

rainfall = pd.read_csv(
    os.path.join(PROCESSED_DIR, "rainfall_features.csv")
)


# ==================================================
# 3. FIND REQUIRED COLUMNS
# ==================================================

def find_column(df, possible_names):

    for col in possible_names:
        if col in df.columns:
            return col

    raise ValueError(
        f"Required column not found: {possible_names}"
    )


river_date = find_column(
    river, ["date", "Date", "datetime"]
)

reservoir_date = find_column(
    reservoir, ["date", "Date", "datetime"]
)

rain_date = find_column(
    rainfall, ["date", "Date", "datetime"]
)

river_level_col = find_column(
    river, ["river_level", "water_level", "level"]
)

reservoir_level_col = find_column(
    reservoir, ["reservoir_level", "water_level", "level"]
)

rain_col = find_column(
    rainfall, ["rainfall", "rainfall_mm", "rainfall_1d"]
)


# ==================================================
# 4. DATA PREPARATION
# ==================================================

river = river[[river_date, river_level_col]].copy()

reservoir = reservoir[
    [reservoir_date, reservoir_level_col]
].copy()

rainfall = rainfall[[rain_date, rain_col]].copy()


for df in [river, reservoir, rainfall]:

    df["date"] = pd.to_datetime(
        df.iloc[:, 0], errors="coerce"
    )

    df.dropna(subset=["date"], inplace=True)

    df["date"] = df["date"].dt.normalize()


river = river[["date", river_level_col]]
reservoir = reservoir[["date", reservoir_level_col]]
rainfall = rainfall[["date", rain_col]]

river.columns = ["date", "river_level"]

reservoir.columns = ["date", "reservoir_level"]

rainfall.columns = ["date", "rainfall"]


# ==================================================
# 5. HANDLE DUPLICATE DATES
# ==================================================

river = (
    river.groupby("date")["river_level"]
    .mean()
    .reset_index()
)

reservoir = (
    reservoir.groupby("date")["reservoir_level"]
    .mean()
    .reset_index()
)

rainfall = (
    rainfall.groupby("date")["rainfall"]
    .mean()
    .reset_index()
)


# ==================================================
# 6. MERGE DATASETS
# ==================================================

data = pd.merge(
    river,
    reservoir,
    on="date",
    how="inner"
)

data = pd.merge(
    data,
    rainfall,
    on="date",
    how="left"
)

data = (
    data.sort_values("date")
    .set_index("date")
)

data = data.asfreq("D")


# ==================================================
# 7. MISSING VALUE HANDLING
# ==================================================

data["rainfall_missing"] = (
    data["rainfall"].isna().astype(int)
)

rainfall_median = data["rainfall"].median()

if pd.isna(rainfall_median):
    rainfall_median = 0

data["rainfall"] = data["rainfall"].fillna(
    rainfall_median
)


# ==================================================
# 8. FEATURE ENGINEERING
# ==================================================

data["river_change"] = (
    data["river_level"].diff()
)

data["reservoir_change"] = (
    data["reservoir_level"].diff()
)

data["rainfall_3d"] = (
    data["rainfall"].rolling(3).sum()
)

data["rainfall_7d"] = (
    data["rainfall"].rolling(7).sum()
)


FEATURES = [
    "river_level",
    "reservoir_level",
    "river_change",
    "reservoir_change",
    "rainfall",
    "rainfall_3d",
    "rainfall_7d",
    "rainfall_missing"
]

print("\nFeature columns:")
print(FEATURES)


# ==================================================
# 9. CREATE FUTURE TARGETS
# ==================================================

def create_future_targets(df, horizon):

    result = df.copy()

    result["future_river"] = (
        result["river_level"].shift(-horizon)
    )

    result["future_reservoir"] = (
        result["reservoir_level"].shift(-horizon)
    )

    result["future_rainfall"] = (
        result["rainfall"]
        .shift(-1)
        .rolling(horizon)
        .sum()
        .shift(-(horizon - 1))
    )

    return result


# ==================================================
# 10. PERCENTILE SCORE FUNCTION
# ==================================================

def percentile_score(values, reference):

    reference = np.sort(
        np.asarray(reference, dtype=float)
    )

    values = np.asarray(values, dtype=float)

    return (
        np.searchsorted(
            reference,
            values,
            side="right"
        ) / len(reference)
    ) * 100


# ==================================================
# 11. CREATE RISK LABELS
# ==================================================

def create_risk_labels(train, test):

    # References are calculated only from training data

    river_ref = np.sort(
        train["future_river"].dropna().values
    )

    reservoir_ref = np.sort(
        train["future_reservoir"].dropna().values
    )

    rainfall_ref = np.sort(
        train["future_rainfall"].dropna().values
    )

    if min(
        len(river_ref),
        len(reservoir_ref),
        len(rainfall_ref)
    ) == 0:

        raise ValueError(
            "Insufficient training reference data."
        )

    def calculate_score(df):

        river_rank = percentile_score(
            df["future_river"],
            river_ref
        )

        reservoir_rank = percentile_score(
            df["future_reservoir"],
            reservoir_ref
        )

        rainfall_rank = percentile_score(
            df["future_rainfall"],
            rainfall_ref
        )

        score = (
            river_rank * 0.40
            + reservoir_rank * 0.35
            + rainfall_rank * 0.25
        )

        return score

    train = train.copy()
    test = test.copy()

    train["risk_score"] = calculate_score(train)

    test["risk_score"] = calculate_score(test)

    bins = [-1, 39, 69, 100]

    labels = [
        "GREEN",
        "YELLOW",
        "RED"
    ]

    train["risk_class"] = pd.cut(
        train["risk_score"],
        bins=bins,
        labels=labels
    )

    test["risk_class"] = pd.cut(
        test["risk_score"],
        bins=bins,
        labels=labels
    )

    references = {

        "river": river_ref,

        "reservoir": reservoir_ref,

        "rainfall": rainfall_ref

    }

    return train, test, references


# ==================================================
# 12. MODEL CONFIGURATION
# ==================================================

MODEL_CONFIG = {

    1: RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    3: GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    7: RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

}


# ==================================================
# 13. TRAINING PROCESS
# ==================================================

all_metrics = []

print("\n====================================")
print(" FINAL FLOOD MODEL TRAINING")
print("====================================")


for horizon in [1, 3, 7]:

    print(f"\nTraining {horizon}-Day Model...")

    df = create_future_targets(
        data,
        horizon
    )

    df = df.dropna(
        subset=FEATURES + [
            "future_river",
            "future_reservoir",
            "future_rainfall"
        ]
    ).copy()

    # Chronological 80:20 split

    split = int(len(df) * 0.80)

    train = df.iloc[:split].copy()

    test = df.iloc[split:].copy()

    if len(train) == 0 or len(test) == 0:

        raise ValueError(
            f"Not enough samples for {horizon}-day model."
        )

    # Create labels and retrieve references

    train, test, references = create_risk_labels(
        train,
        test
    )

    X_train = train[FEATURES]

    X_test = test[FEATURES]

    y_train = train["risk_class"]

    y_test = test["risk_class"]

    # Remove undefined labels

    train_mask = y_train.notna()

    test_mask = y_test.notna()

    X_train = X_train.loc[train_mask]

    y_train = y_train.loc[train_mask]

    X_test = X_test.loc[test_mask]

    y_test = y_test.loc[test_mask]

    if X_train.empty or X_test.empty:

        raise ValueError(
            f"Empty training or testing data for {horizon}-day model."
        )

    # Train model

    model = MODEL_CONFIG[horizon]

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)


    # ==================================================
    # 14. MODEL EVALUATION
    # ==================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    red_precision = precision_score(
        y_test,
        predictions,
        labels=["RED"],
        average="macro",
        zero_division=0
    )

    red_recall = recall_score(
        y_test,
        predictions,
        labels=["RED"],
        average="macro",
        zero_division=0
    )

    red_f1 = f1_score(
        y_test,
        predictions,
        labels=["RED"],
        average="macro",
        zero_division=0
    )


    print("\nAccuracy:", round(accuracy, 4))

    print("Macro F1:", round(macro_f1, 4))

    print("RED Precision:", round(red_precision, 4))

    print("RED Recall:", round(red_recall, 4))

    print("RED F1:", round(red_f1, 4))


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[
                "GREEN",
                "YELLOW",
                "RED"
            ],
            zero_division=0
        )
    )


    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[
                "GREEN",
                "YELLOW",
                "RED"
            ]
        )
    )


    # ==================================================
    # 15. SAVE MODEL
    # ==================================================

    model_path = os.path.join(
        MODEL_DIR,
        f"flood_final_{horizon}day.pkl"
    )

    bundle = {

        "model": model,

        "features": FEATURES,

        "horizon": horizon,

        "classes": [
            "GREEN",
            "YELLOW",
            "RED"
        ],

        "algorithm": type(model).__name__,

        "label_method": (
            "Experimental percentile-based proxy risk labels"
        ),

        "train_end_date": str(train.index.max()),

        "test_start_date": str(test.index.min()),

        "rainfall_median": rainfall_median,

        "training_reference": references

    }

    joblib.dump(
        bundle,
        model_path
    )

    print("\nModel saved:", model_path)


    # ==================================================
    # 16. SAVE PREDICTIONS
    # ==================================================

    prediction_df = pd.DataFrame({

        "date": X_test.index,

        "actual_risk": y_test.values,

        "predicted_risk": predictions,

        "risk_score": test.loc[
            test_mask,
            "risk_score"
        ].values

    })

    prediction_path = os.path.join(
        REPORT_DIR,
        f"flood_final_{horizon}day_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    print("Predictions saved:", prediction_path)


    # ==================================================
    # 17. STORE METRICS
    # ==================================================

    all_metrics.append({

        "Horizon": f"{horizon}-Day",

        "Algorithm": type(model).__name__,

        "Accuracy": accuracy,

        "Macro_F1": macro_f1,

        "RED_Precision": red_precision,

        "RED_Recall": red_recall,

        "RED_F1": red_f1,

        "Train_Samples": len(X_train),

        "Test_Samples": len(X_test)

    })


# ==================================================
# 18. SAVE FINAL METRICS
# ==================================================

metrics_df = pd.DataFrame(
    all_metrics
)

metrics_path = os.path.join(
    REPORT_DIR,
    "flood_final_model_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ==================================================
# 19. FINAL SUMMARY
# ==================================================

print("\n====================================")
print(" FINAL TRAINING COMPLETED")
print("====================================")

print("\nFINAL MODEL PERFORMANCE:\n")

print(
    metrics_df.to_string(index=False)
)

print("\nModels saved in:")
print(MODEL_DIR)

print("\nMetrics saved in:")
print(metrics_path)

print("\nAll prediction reports saved in:")
print(REPORT_DIR)

print("\n====================================")
print(" JALRAKSHAK AI TRAINING FINISHED")
print("====================================")