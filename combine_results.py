
import pandas as pd

sensors = pd.read_csv("sensor_data.csv")
verilog = pd.read_csv("verilog_results.csv")

# Keep only the four test cases with verified Verilog outputs.
sensors = sensors[sensors["sample_id"].isin([0, 1, 2, 3])]

results = sensors.merge(
    verilog,
    on="sample_id",
    how="inner",
    validate="one_to_one",
)

results["knn_prediction"] = results["knn_prediction"].astype(int)
results["fault_flag"] = results["fault_flag"].astype(int)


results["both_flagged"] = (
    (results["knn_prediction"] == 1) &
    (results["fault_flag"] == 1)
).astype(int)

results["knn_only"] = (
    (results["knn_prediction"] == 1) &
    (results["fault_flag"] == 0)
).astype(int)

results["verilog_only"] = (
    (results["knn_prediction"] == 0) &
    (results["fault_flag"] == 1)
).astype(int)

columns = [
    "sample_id",
    "knn_prediction",
    "knn_anomaly_probability",
    "verilog_altitude",
    "verilog_speed",
    "battery",
    "gps_valid",
    "fault_code",
    "fault_flag",
    "recorder_frozen",
    "both_flagged",
    "knn_only",
    "verilog_only",
]

results[columns].to_csv("integration_results.csv", index=False)

print("\nINTEGRATED DRONEGUARD RESULTS")
print(results[columns].to_string(index=False))

print("\nSummary")
print("Matched samples:", len(results))
print("Both flagged:", results["both_flagged"].sum())
print("KNN only:", results["knn_only"].sum())
print("Verilog only:", results["verilog_only"].sum())

print("\nSaved integration_results.csv")
