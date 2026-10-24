
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------- FILES ----------
DATASET = Path("drone_communication_dataset.csv")
MODEL_PATH = Path("best_knn_model.joblib")
OUTPUT_PATH = Path("sensor_data.csv")

# Features used by the KNN model
FEATURES = [
    "altitude",
    "speed_trajectory",
    "signal_strength",
    "round_trip_time",
    "network_traffic_volume",
]

RANDOM_SEED = 42
NUMBER_OF_SAMPLES = 20


def main():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Cannot find {DATASET}. Put it in this folder."
        )

    df = pd.read_csv(DATASET)

    required = FEATURES + ["label_normal"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    # 1 means anomaly; 0 means normal.
    df = df.dropna(subset=required).copy()
    df["is_anomaly"] = 1 - df["label_normal"].astype(int)

    # Ensure the selected features are numeric.
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES)
    X = df[FEATURES]
    y = df["is_anomaly"]

    # ------------- TRAIN OR LOAD KNN -------------
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"Loaded saved model: {MODEL_PATH}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=y,
        )

        model = Pipeline([
            ("scaler", RobustScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=41,
                weights="distance",
                p=2,
            )),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print("KNN evaluation against dataset labels")
        print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
        print(classification_report(
            y_test,
            predictions,
            zero_division=0,
        ))

        joblib.dump(model, MODEL_PATH)
        print(f"Saved model: {MODEL_PATH}")

    # ------------- SELECT TEST SAMPLES -------------
    # Sampling from the dataset gives the KNN realistic feature combinations.
    sample_count = min(NUMBER_OF_SAMPLES, len(df))
    samples = df.sample(
        n=sample_count,
        random_state=RANDOM_SEED,
    ).copy()

    samples = samples.reset_index(drop=True)
    samples["sample_id"] = range(len(samples))
    samples["knn_prediction"] = model.predict(samples[FEATURES])
    samples["knn_anomaly_probability"] = (
        model.predict_proba(samples[FEATURES])[:, 1]
    )

    # Keep original ML features for the Python model.
    # Map altitude/speed into the current Verilog input ranges for simulation.
    # These are scaled test values, NOT original physical measurements.
    def scale_to_range(series, low, high):
        min_value = series.min()
        max_value = series.max()

        if max_value == min_value:
            return pd.Series(
                [(low + high) / 2] * len(series),
                index=series.index,
            )

        return low + (series - min_value) * (
            (high - low) / (max_value - min_value)
        )

    samples["verilog_altitude"] = (
        scale_to_range(samples["altitude"], 10, 120).round().astype(int)
    )
    samples["verilog_speed"] = (
        scale_to_range(samples["speed_trajectory"], 0, 30)
        .round().astype(int)
    )

    # Battery and GPS are not features in this KNN model.
    # These are simulated inputs for testing the Verilog logic.
    rng = np.random.default_rng(RANDOM_SEED)
    samples["battery"] = rng.integers(40, 101, size=sample_count)
    samples["gps_valid"] = 1

    # Add a few deliberate Verilog fault cases.
    # Adjust these values if your Verilog parameters differ.
    if sample_count >= 4:
        samples.loc[0, "verilog_altitude"] = 150  # altitude-range fault
        samples.loc[1, "verilog_speed"] = 40      # speed-range fault
        samples.loc[2, "battery"] = 10            # low-battery fault
        samples.loc[3, "gps_valid"] = 0           # GPS fault

    # Export the original ML features, KNN prediction, and Verilog inputs.
    columns = [
    "sample_id",
    *FEATURES,
    "is_anomaly",
    "knn_prediction",
    "knn_anomaly_probability",
    "verilog_altitude",
    "verilog_speed",
    "battery",
    "gps_valid",
]

    samples[columns].to_csv(OUTPUT_PATH, index=False)

    print(f"\nCreated {OUTPUT_PATH} with {sample_count} test cases.")
    print(samples[[
        "knn_prediction",
        "verilog_altitude",
        "verilog_speed",
        "battery",
        "gps_valid",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()



