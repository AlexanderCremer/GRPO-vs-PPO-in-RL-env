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
            last_5_percent_values = series.iloc[last_5_percent_index:]

            mean_val = last_5_percent_values.mean()
            run_means.append(mean_val)

            print(f"→ Last 5% mean: {mean_val:.2f}")

        except Exception as e:
            print(f"Failed to process run {run.name}: {e}")

    if not run_means:
        print("❌ No valid data across all runs.")
        return

    overall_mean = np.mean(run_means)
    overall_std = np.std(run_means)

    print("\n Final Aggregated Statistics (over last 5% of each run):")
    print(f"Mean of means: {overall_mean:.2f}")
    print(f"Std across run means: {overall_std:.2f}")


# Run it
analyze_final_runs(
    entity="akcremer11a",
    project="Final",
    group="Acrobot_G10",
    y_metric_key="on-policy cumulative reward"
)
