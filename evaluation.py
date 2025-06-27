import sys

import wandb
from wandb import Api
import pandas as pd
import numpy as np

def analyze_final_runs(
    entity: str,
    project: str,
    group: str,
    y_metric_key: str
):
    api = Api()

    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    if not runs:
        print(f"No runs found in project '{project}' with group '{group}'")
        return

    run_means = []
    last_val = []
    for run in runs:
        print(f"Processing run: {run.name} ({run.id})")
        try:
            history = run.history(samples=10000)
            if y_metric_key not in history.columns:
                print(f"Metric '{y_metric_key}' not found in run {run.name}")
                continue

            series = history[y_metric_key].dropna()
            if series.empty:
                print(f"No valid data for metric '{y_metric_key}' in run {run.name}")
                continue

            last_5_percent_index = int(len(series) * 0.95)
            last_5_percent_values = pd.to_numeric(
            series.iloc[last_5_percent_index:].reset_index(drop=True),
            errors='coerce'
            ).dropna().reset_index(drop=True)
            last_val.append(last_5_percent_values)

            mean_val = last_5_percent_values.mean()
            run_means.append(mean_val)

            print(f"→ Last 5% mean: {mean_val:.2f}")

        except Exception as e:
            print(f"Failed to process run {run.name}: {e}")

    if not run_means:
        print("❌ No valid data across all runs.")
        return

    overall_mean = np.mean(run_means)

    # Compute the mean of stds over time across runs:
    # 1. Find minimum length
    min_length = min(len(lst) for lst in last_val)
    if min_length == 0:
        print("One of the runs has no last 5% data.")
        return

    # 2. Truncate all to this length
    truncated = [lst.iloc[:min_length].values for lst in last_val]
    # Shape: (num_runs, min_length)
    data_matrix = np.stack(truncated)
    #assert data_matrix.shape[1] == min_length
    # 3. Compute std at each timestep (over runs)
    #np.set_printoptions(threshold=sys.maxsize)
    #print(data_matrix)
    stds_over_time = np.std(data_matrix, axis=0)

    # 4. Mean of those stds
    overall_std = np.mean(stds_over_time)

    print("\n Final Aggregated Statistics (over last 5% of each run):")
    print(f"Mean of means: {overall_mean:.2f}")
    print(f"Mean of stds over time: {overall_std:.2f}")


# Run it
analyze_final_runs(
    entity="akcremer11a",
    project="Final",
    group="CartPole_PPO",
    y_metric_key="on-policy cumulative reward"
)
