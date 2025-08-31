import wandb
from wandb import Api
import pandas as pd
import os

def export_and_update_runs_relative_time(
    entity: str,
    project_source: str,
    group: str,
    project_final: str,
    y_metric_key_source: str,
    y_metric_key_csv: str,
    output_dir: str = "wandb_rewards_csvs"
):
    os.makedirs(output_dir, exist_ok=True)
    api = Api()

    #Get source runs filtered by group
    source_runs = api.runs(f"{entity}/{project_source}", filters={"group": group})
    if not source_runs:
        print(f"No runs found in project '{project_source}' with group '{group}'")
        return

    csv_paths = {}

    #Export CSVs using relative time as x-axis
    for run in source_runs:
        try:
            print(f"\nProcessing run: {run.name} ({run.id})")

            rows = []
            start_time = None

            for row in run.scan_history():
                if y_metric_key_source in row and "_timestamp" in row:
                    reward_val = row[y_metric_key_source]
                    #Drop entries where reward_val is None, NaN, or empty string
                    if reward_val is not None and reward_val != '' and not pd.isna(reward_val):
                        if start_time is None:
                            start_time = row["_timestamp"]
                        rows.append({
                            "Time (Wall)": row["_timestamp"] - start_time,
                            y_metric_key_csv: reward_val
                        })

            if not rows:
                print(f"No usable data found for run {run.name}")
                continue

            df = pd.DataFrame(rows)
            out_path = os.path.join(output_dir, f"{run.id}_{run.name}_relative_time.csv")
            df.to_csv(out_path, index=False)
            csv_paths[run.id] = out_path
            print(f"Saved: {out_path}")

        except Exception as e:
            print(f"Failed to process run {run.name}: {e}")

    if not csv_paths:
        print("No valid CSVs were created. Exiting.")
        return

    #Find smallest max time across runs for truncation
    min_last_time = None
    for path in csv_paths.values():
        df = pd.read_csv(path)
        # Drop empty or missing reward rows before checking max time
        df = df.dropna(subset=[y_metric_key_csv])
        df = df[df[y_metric_key_csv] != '']
        last_time = df["Time (Wall)"].max()
        if min_last_time is None or last_time < min_last_time:
            min_last_time = last_time

    print(f"\nLowest last time across all runs: {min_last_time:.2f} seconds")

    #Truncate all CSVs by min_last_time
    for path in csv_paths.values():
        df = pd.read_csv(path)
        df = df.dropna(subset=[y_metric_key_csv])
        df = df[df[y_metric_key_csv] != '']
        df_truncated = df[df["Time (Wall)"] <= min_last_time]
        df_truncated.to_csv(path, index=False)
        print(f"Truncated and saved: {path}")

    #Update or create final runs
    final_runs = {run.name: run for run in api.runs(f"{entity}/{project_final}")}

    for source_run in source_runs:
        if source_run.id not in csv_paths:
            print(f"Skipping source run {source_run.id} - no matching CSV")
            continue

        csv_path = csv_paths[source_run.id]
        run_name = f"{source_run.id}_{source_run.name}_reward"

        if run_name in final_runs:
            print(f"Updating existing run: {run_name}")
            run = wandb.init(
                project=project_final,
                entity=entity,
                name=run_name,
                id=final_runs[run_name].id,
                resume="must",
                reinit=True
            )
        else:
            print(f"No matching run in '{project_final}' for: {run_name}, creating a new run.")
            run = wandb.init(
                project=project_final,
                entity=entity,
                name=run_name,
                reinit=True
            )

        df = pd.read_csv(csv_path)
        #Drop rows with missing or empty reward before logging
        df = df.dropna(subset=[y_metric_key_csv])
        df = df[df[y_metric_key_csv] != '']

        for _, row in df.iterrows():
            wandb.log({
                "Time (Wall)": float(row["Time (Wall)"]),
                y_metric_key_csv: float(row[y_metric_key_csv])
            })

        run.finish()
        print(f"Finished run '{run_name}'")


export_and_update_runs_relative_time(
     entity="akcremer11a",
     project_source="PPO_gymnasium",
     group="CartPole_PPO",
     project_final="PPO_paper",
     y_metric_key_source="evaluation/mean_reward_vs_time",
     y_metric_key_csv="mean_reward_vs_time",
     output_dir="wandb_rewards_csvs"
)


export_and_update_runs_relative_time(
     entity="akcremer11a",
     project_source="PPO_gymnasium",
     group="Acrobot_PPO",
     project_final="PPO_paper",
     y_metric_key_source="evaluation/mean_reward_vs_time",
     y_metric_key_csv="mean_reward_vs_time",
     output_dir="wandb_rewards_csvs"
)
