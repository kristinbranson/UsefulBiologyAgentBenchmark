#!/usr/bin/env python3
"""
Script to collect metrics from all fly benchmark jobs.
Fetches data from the internal benchmark server API.
"""

import requests
import json
from collections import defaultdict

BASE_URL = "http://h07u12.int.janelia.org:8080/api"

# Default metrics for each job type when missing
DEFAULT_METRICS = {
    "fly-body-tracking": {
        "HOTA": 0.0,
        "HOTA_baseline": 0.0,
        "DetA": 0.0,
        "AssA": 0.0,
        "DetRe": 0.0,
        "DetPr": 0.0,
        "AssRe": 0.0,
        "AssPr": 0.0,
        "LocA": 0.0,
    },
    "fly-movement-classifier": {
        "model_py_exists": False,
        "model_pkl_exists": False,
        "valid_interface": False,
        "feature_names": [],
        "n_features": 0,
        "predict_valid": False,
        "predict_unique_values": [],
        "balanced_accuracy": 0.0,
        "n_test_frames": 0,
        "n_walk_frames": 0,
        "n_none_frames": 0,
        "recall_walk": 0.0,
        "recall_none": 0.0,
        "baseline_balanced_accuracy": 0.0,
        "baseline_recall_walk": 0.0,
        "baseline_recall_none": 0.0,
    },
    "fly-registration": {
        "ntargets_pred": 0,
        "ntargets_gt": 0,
        "x_mm_max_error": 0.0,
        "x_mm_mean_max_error": 0.0,
        "y_mm_max_error": 0.0,
        "y_mm_mean_max_error": 0.0,
        "translation_tolerance_mm": 0.0,
        "radius_max_error_px": 0.0,
        "radius_mean_error_px": 0.0,
        "radius_tolerance_px": 0.0,
    },
    "fly-swing-stance": {
        "balanced_accuracy": 0.0,
        "recall_stance": 0.0,
        "recall_swing": 0.0,
        "stance_total": 0,
        "swing_total": 0,
        "stance_correct": 0,
        "swing_correct": 0,
    },
}

def get_all_jobs():
    """Fetch all jobs from the API."""
    response = requests.get(f"{BASE_URL}/jobs")
    response.raise_for_status()
    return response.json()["items"]

def get_trials(job_name):
    """Fetch all trials for a specific job."""
    response = requests.get(f"{BASE_URL}/jobs/{job_name}/trials")
    response.raise_for_status()
    return response.json()["items"]

def get_metrics(job_name, trial_name, job_type):
    """Fetch metrics.json for a specific trial. Returns defaults if missing."""
    try:
        response = requests.get(f"{BASE_URL}/jobs/{job_name}/trials/{trial_name}/files/verifier/metrics.json")
        response.raise_for_status()
        metrics = response.json()
        metrics["metrics_missing"] = False
        return metrics
    except Exception as e:
        print(f"  Warning: Could not fetch metrics for {trial_name}, using zeros")
        defaults = DEFAULT_METRICS.get(job_type, {}).copy()
        defaults["metrics_missing"] = True
        return defaults

def collect_metrics_for_job_type(all_jobs, job_type_prefix, trial_filter=None):
    """Collect metrics for jobs matching a prefix."""
    matching_jobs = [j for j in all_jobs if j["name"].startswith(job_type_prefix) and "RETRY" not in j["name"]]
    print(f"\nFound {len(matching_jobs)} {job_type_prefix} jobs:")
    for j in matching_jobs:
        print(f"  - {j['name']}")

    all_metrics = []

    for job in matching_jobs:
        job_name = job["name"]
        print(f"\nProcessing job: {job_name}")

        trials = get_trials(job_name)

        # Apply trial filter if provided
        if trial_filter:
            trials = [t for t in trials if trial_filter(t["name"])]

        print(f"  Found {len(trials)} trials")

        for trial in trials:
            trial_name = trial["name"]
            task_name = trial["task_name"]
            agent = trial["agent_name"]
            model = trial["model_name"]
            provider = trial.get("model_provider", "")

            print(f"  Fetching metrics for: {trial_name}")
            metrics = get_metrics(job_name, trial_name, job_type_prefix)

            entry = {
                "job_name": job_name,
                "trial_name": trial_name,
                "task_name": task_name,
                "agent": agent,
                "model": model,
                "provider": provider,
                "reward": trial.get("reward", None),
                **metrics
            }
            all_metrics.append(entry)

    return all_metrics

def print_summary(metrics, job_type, metric_columns):
    """Print summary for a job type with specified metric columns."""
    print(f"\n\n=== {job_type.upper()} SUMMARY ===")

    # Group by job name
    job_data = defaultdict(lambda: {"metrics": defaultdict(list), "missing": 0})

    for entry in metrics:
        job_name = entry["job_name"]
        for col in metric_columns:
            job_data[job_name]["metrics"][col].append(entry.get(col, 0.0) or 0.0)
        if entry.get("metrics_missing", False):
            job_data[job_name]["missing"] += 1

    # Print header
    header = f"{'Job Name':<50}"
    for col in metric_columns:
        header += f" {col:>14}"
    header += f" {'Count':>8} {'Missing':>8}"
    print(header)
    print("-" * len(header))

    # Print data
    for job_name in sorted(job_data.keys()):
        data = job_data[job_name]
        row = f"{job_name:<50}"
        for col in metric_columns:
            values = data["metrics"][col]
            avg = sum(values) / len(values) if values else 0.0
            row += f" {avg:>14.4f}"
        row += f" {len(data['metrics'][metric_columns[0]]):>8}"
        row += f" {data['missing']:>8}"
        print(row)

def main():
    print("Fetching all jobs...")
    all_jobs = get_all_jobs()

    # Collect fly-body-tracking metrics (only vnc2 trials)
    body_tracking_metrics = collect_metrics_for_job_type(
        all_jobs, "fly-body-tracking",
        trial_filter=lambda name: name.startswith("fly-body-tracking-vnc2")
    )
    with open("fly_body_tracking_metrics.json", "w") as f:
        json.dump(body_tracking_metrics, f, indent=2)
    print(f"\nSaved {len(body_tracking_metrics)} entries to fly_body_tracking_metrics.json")

    # Collect fly-movement-classifier metrics
    classifier_metrics = collect_metrics_for_job_type(all_jobs, "fly-movement-classifier")
    with open("fly_movement_classifier_metrics.json", "w") as f:
        json.dump(classifier_metrics, f, indent=2)
    print(f"\nSaved {len(classifier_metrics)} entries to fly_movement_classifier_metrics.json")

    # Collect fly-registration metrics
    registration_metrics = collect_metrics_for_job_type(
        all_jobs, "fly-registration",
        trial_filter=lambda name: name.startswith("fly-registration-vnc2")
    )
    with open("fly_registration_metrics.json", "w") as f:
        json.dump(registration_metrics, f, indent=2)
    print(f"\nSaved {len(registration_metrics)} entries to fly_registration_metrics.json")

    # Collect fly-swing-stance metrics
    swing_stance_metrics = collect_metrics_for_job_type(
        all_jobs, "fly-swing-stance",
        trial_filter=lambda name: name.startswith("fly-swing-stance-vnc2")
    )
    with open("fly_swing_stance_metrics.json", "w") as f:
        json.dump(swing_stance_metrics, f, indent=2)
    print(f"\nSaved {len(swing_stance_metrics)} entries to fly_swing_stance_metrics.json")

    # Print summaries
    print_summary(body_tracking_metrics, "fly-body-tracking", ["HOTA", "HOTA_baseline", "reward"])
    print_summary(registration_metrics, "fly-registration", ["reward"])
    print_summary(classifier_metrics, "fly-movement-classifier", ["balanced_accuracy", "baseline_balanced_accuracy", "reward"])
    print_summary(swing_stance_metrics, "fly-swing-stance", ["balanced_accuracy", "reward"])

if __name__ == "__main__":
    main()
