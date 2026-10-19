import os
import numpy as np
import pandas as pd

FEATURES = [
    "sensor_temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "rotational_speed_rpm",
    "operating_hours",
    "torque_nm",
]

def generate_sensor_telemetry(n_samples: int = 7000, drift: bool = False, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_failure = int(n_samples * (0.04 if drift else 0.02))
    n_normal = n_samples - n_failure
    shift = 1.3 if drift else 1.0

    normal = pd.DataFrame({
        "sensor_temperature_c": rng.normal(68.5, 4.2, n_normal).clip(50, 95),
        "vibration_mm_s": rng.normal(1.2, 0.3, n_normal).clip(0.1, 4.0),
        "pressure_bar": rng.normal(30.0, 1.5, n_normal).clip(20, 45),
        "rotational_speed_rpm": rng.normal(1500, 35, n_normal).clip(1200, 1800),
        "operating_hours": rng.integers(10, 1500, n_normal),
        "torque_nm": rng.normal(45.0, 5.0, n_normal).clip(20, 70),
        "failure_imminent": 0,
    })

    failure = pd.DataFrame({
        "sensor_temperature_c": rng.normal(88.0, 6.5, n_failure).clip(75, 120) * shift,
        "vibration_mm_s": rng.normal(4.8, 1.2, n_failure).clip(2.0, 10.0) * shift,
        "pressure_bar": rng.normal(24.0, 3.2, n_failure).clip(10, 38),
        "rotational_speed_rpm": rng.normal(1380, 80, n_failure).clip(1000, 1700),
        "operating_hours": rng.integers(1200, 5000, n_failure),
        "torque_nm": rng.normal(62.0, 8.5, n_failure).clip(40, 95),
        "failure_imminent": 1,
    })

    df = pd.concat([normal, failure], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    ref_df = generate_sensor_telemetry(n_samples=7000, drift=False, seed=42)
    drift_df = generate_sensor_telemetry(n_samples=2000, drift=True, seed=99)

    ref_df.to_csv("data/raw/sensor_reference.csv", index=False)
    drift_df.to_csv("data/raw/sensor_drifted.csv", index=False)
    print(f"[OK] PredictiveOps reference dataset ({len(ref_df)} rows) & drifted dataset ({len(drift_df)} rows) generated.")