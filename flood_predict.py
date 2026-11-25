import os
import joblib
import numpy as np
import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")


# ==========================================
# 2. LOAD TRAINED MODELS
# ==========================================

def load_models():

    models = {}

    for horizon in [1, 3, 7]:

        path = os.path.join(
            MODEL_DIR,
            f"flood_final_{horizon}day.pkl"
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        models[horizon] = joblib.load(path)

    return models


# ==========================================
# 3. USER INPUT
# ==========================================

def get_positive_input(message):

    while True:

        try:
            value = float(input(message))

            if not np.isfinite(value) or value < 0:
                print("Please enter a valid non-negative number.")
                continue

            return value

        except ValueError:
            print("Invalid input. Enter a number.")


def get_rainfall_input():

    rainfall_values = []

    print("\nEnter rainfall for the last 7 days (in mm).")
    print("Enter values from oldest day to today.")

    for day in range(1, 8):

        value = get_positive_input(
            f"Day {day} rainfall: "
        )

        rainfall_values.append(value)

    return rainfall_values


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

def create_input_features():

    print("\n========== CURRENT WATER DATA ==========")

    previous_river = get_positive_input(
        "Previous river level (m): "
    )

    current_river = get_positive_input(
        "Current river level (m): "
    )

    previous_reservoir = get_positive_input(
        "Previous reservoir level (m): "
    )

    current_reservoir = get_positive_input(
        "Current reservoir level (m): "
    )

    rainfall_values = get_rainfall_input()

    rainfall_today = rainfall_values[-1]

    rainfall_3d = sum(rainfall_values[-3:])

    rainfall_7d = sum(rainfall_values)

    river_change = current_river - previous_river

    reservoir_change = (
        current_reservoir - previous_reservoir
    )

    features = {

        "river_level": current_river,

        "reservoir_level": current_reservoir,

        "river_change": river_change,

        "reservoir_change": reservoir_change,

        "rainfall": rainfall_today,

        "rainfall_3d": rainfall_3d,

        "rainfall_7d": rainfall_7d,

        "rainfall_missing": 0

    }

    return pd.DataFrame([features])


# ==========================================
# 5. PREDICTION
# ==========================================

def predict_flood_risk(models, input_data):

    print("\n====================================")
    print(" JALRAKSHAK AI - FLOOD RISK RESULTS")
    print("====================================")

    for horizon in [1, 3, 7]:

        bundle = models[horizon]

        model = bundle["model"]

        features = bundle["features"]

        X = input_data[features]

        prediction = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        classes = model.classes_

        print(f"\n{horizon}-DAY PREDICTION")
        print("----------------------------")

        print("Predicted Risk:", prediction)

        print("\nClass Probabilities:")

        for class_name, probability in zip(
            classes, probabilities
        ):

            print(
                f"{class_name}: {probability * 100:.2f}%"
            )


# ==========================================
# 6. MAIN FUNCTION
# ==========================================

def main():

    print("\n====================================")
    print("       JALRAKSHAK AI")
    print("    FLOOD RISK PREDICTION")
    print("====================================")

    try:

        print("\nLoading trained models...")

        models = load_models()

        print("All three models loaded successfully!")

        input_data = create_input_features()

        print("\nGenerated Features:")
        print(input_data.to_string(index=False))

        predict_flood_risk(
            models,
            input_data
        )

        print("\n====================================")
        print("       PREDICTION COMPLETED")
        print("====================================")

        print(
            "\nNOTE: Experimental predictions only. "
            "Not an official flood warning."
        )

    except Exception as error:

        print("\nERROR:", error)


if __name__ == "__main__":
    main()