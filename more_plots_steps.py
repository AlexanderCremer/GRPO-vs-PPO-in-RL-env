import wandb
from wandb import Api
import pandas as pd
import os

def export_and_update_runs(
    entity: str,
    project_source: str,
    group: str,
    project_final: str,
    y_metric_key: str,
    y_metric_key_csv: str,
    output_dir: str = "wandb_rewards_csvs"
):
    os.makedirs(output_dir, exist_ok=True)
    api = Api()

    # Get source runs filtered by group
    source_runs = api.runs(f"{entity}/{project_source}", filters={"group": group})
    if not source_runs:
        print(f"No runs found in project '{project_source}' with group '{group}'")
        return

    csv_paths = {}

    # Step 1: Export CSVs using global_step as x-axis
    for run in source_runs:
        try:
            print(f"\nProcessing run: {run.name} ({run.id})")

            rows = []
            for row in run.scan_history():
                if y_metric_key in row and "global_step" in row:
                    rows.append({
                        "global_step": row["global_step"],
                        y_metric_key_csv: row[y_metric_key]  # Save with friendly column name
                    })

            if not rows:
                print(f"No usable data found for run {run.name}")
                continue

            df = pd.DataFrame(rows)
            out_path = os.path.join(output_dir, f"{run.id}_{run.name}_global_step.csv")
            df.to_csv(out_path, index=False)
            csv_paths[run.id] = out_path
            print(f"Saved: {out_path}")

        except Exception as e:
            print(f"Failed to process run {run.name}: {e}")

    if not csv_paths:
        print("❌ No valid CSVs were created. Exiting.")
        return

    # Step 2: Find the smallest max global_step across runs for truncation
    min_last_step = None
    for path in csv_paths.values():
        df = pd.read_csv(path)
        last_step = df["global_step"].max()
        if min_last_step is None or last_step < min_last_step:
            min_last_step = last_step

    print(f"\nLowest last global_step across all runs: {min_last_step}")

    # Step 3: Truncate all CSVs by min_last_step
    for path in csv_paths.values():
        df = pd.read_csv(path)
        df_truncated = df[df["global_step"] <= min_last_step]
        df_truncated.to_csv(path, index=False)
        print(f"Truncated and saved: {path}")

    # Step 4: Update final runs
    final_runs = {run.name: run for run in api.runs(f"{entity}/{project_final}")}

    for source_run in source_runs:
        if source_run.id not in csv_paths:
            print(f"Skipping source run {source_run.id} - no matching CSV")
            continue

        csv_path = csv_paths[source_run.id]
        run_name = f"{source_run.id}_{source_run.name}_reward"

        if run_name not in final_runs:
            print(f"No matching run in '{project_final}' for: {run_name}")
            continue

        print(f"Updating run: {run_name}")

        run = wandb.init(
            project=project_final,
            entity=entity,
            name=run_name,
            id=final_runs[run_name].id,
            resume="must",
            reinit=True
        )

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            wandb.log({
                "total steps": int(row["global_step"]),
                y_metric_key_csv: float(row[y_metric_key_csv])  # Use friendly CSV column here
            })

        run.finish()
        print(f"✅ Updated run '{run_name}'")

# Example usage:
export_and_update_runs(
    entity="akcremer11a",
    project_source="GRPO",
    group="CartPole_G2",
    project_final="Final",
    y_metric_key="reward/mean_reward",
    y_metric_key_csv="on-policy cumulative reward",
    output_dir="wandb_rewards_csvs"
)
