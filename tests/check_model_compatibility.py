"""
Script to automate model compatibility and reproducibility checks for Annif.

- Downloads previous models and metrics from Hugging Face Hub (if available)
- Evaluates them with the current Annif version
- Trains and evaluates new models
- Compares metrics and reports differences (1% threshold)
- Uses projects defined in projects.cfg.dist
- Gracefully handles missing models/metrics
"""

import argparse
import json
import os
import subprocess

HUB_REPO = "juhoinkinen/Annif-models-compat"
PROJECTS_CFG = "tests/projects-compatibility.cfg"
CORPORA_DIR = "tests/corpora/archaeology/fulltext/"
PREV_RESULTS_DIR = "metrics"
CURR_RESULTS_DIR = "new_metrics"
THRESHOLD = 0.01  # Allowable relative difference in metrics


def setup_dirs():
    os.makedirs(CURR_RESULTS_DIR, exist_ok=True)
    os.makedirs(PREV_RESULTS_DIR, exist_ok=True)


def get_project_ids(cfg_path):
    ids = []

    with open(cfg_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if line.startswith("[") and line.endswith("]"):
                    ids.append(line[1:-1])
    return ids


def run_cmd(cmd, check=True):

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        if check:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def download_model(project_id):

    cmd = ["annif", "download", project_id, HUB_REPO, "-f", "--trust-repo"]
    try:
        run_cmd(cmd, check=False)
    except Exception as e:
        print(f"Download failed for {project_id}: {e}")


def download_metrics():

    cmd = ["hf", "download", HUB_REPO, "--include", "metrics/*", "--local-dir", "./"]
    try:
        run_cmd(cmd, check=False)
    except Exception as e:
        print(f"Download failed: {e}")


def upload_models():

    cmd = ["annif", "upload", "*", HUB_REPO, "-p", PROJECTS_CFG]
    try:
        run_cmd(cmd, check=False)
    except Exception as e:
        print(f"Upload failed: {e}")


def upload_metrics():

    cmd = ["hf", "upload", HUB_REPO, CURR_RESULTS_DIR, "metrics/"]
    try:
        run_cmd(cmd, check=False)
    except Exception as e:
        print(f"Upload failed: {e}")


def eval_model(project_id, result_file):
    # Assume test data is defined in the project config
    cmd = [
        "annif",
        "eval",
        project_id,
        CORPORA_DIR,
        "-p",
        PROJECTS_CFG,
        "-M",
        result_file,
    ]
    run_cmd(cmd)


def train_model(project_id):
    cmd = ["annif", "train", project_id, CORPORA_DIR, "-p", PROJECTS_CFG]
    run_cmd(cmd)


def load_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        return json.load(f)


def compare_metrics(metrics1, metrics2):
    diffs = {}
    for key in metrics1:
        if (
            key in metrics2
            and isinstance(metrics1[key], (int, float))
            and isinstance(metrics2[key], (int, float))
        ):
            v1, v2 = metrics1[key], metrics2[key]
            if v1 == 0:
                continue
            rel_diff = abs(v1 - v2) / abs(v1)
            if rel_diff > THRESHOLD:
                diffs[key] = (v1, v2, rel_diff)
    return diffs


def main():
    parser = argparse.ArgumentParser(
        description="Annif model compatibility and reproducibility check."
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Enable CI mode for GitHub Actions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Relative metric difference threshold",
    )
    args = parser.parse_args()

    setup_dirs()
    project_ids = get_project_ids(PROJECTS_CFG)
    download_metrics()
    significant_diffs = []
    for project_id in project_ids:
        print(f"\n=== Checking project {project_id} ===")
        # 1. Download previous model and metrics
        download_model(project_id)
        prev_metrics_path = os.path.join(PREV_RESULTS_DIR, f"{project_id}.json")
        prev_metrics = load_metrics(prev_metrics_path)
        if prev_metrics is None:
            print(
                f"No previous metrics for {project_id}, "
                "skipping compatibility check."
            )
        else:
            # 2. Evaluate previous model with current Annif
            curr_metrics_path = os.path.join(CURR_RESULTS_DIR, f"{project_id}.json")
            try:
                eval_model(project_id, curr_metrics_path)
                curr_metrics = load_metrics(curr_metrics_path)
                if curr_metrics:
                    diffs = compare_metrics(prev_metrics, curr_metrics)
                    if diffs:
                        msg = (
                            f"Metric differences for {project_id} "
                            f"(> {THRESHOLD * 100:.1f}%): {diffs}"
                        )
                        print(msg)
                        if args.ci:
                            print(
                                f"::error file={project_id}::"
                                f"Backward compatibility check failed: {msg}"
                            )
                        significant_diffs.append((project_id, "compatibility", diffs))
                    else:
                        print(f"No significant metric differences for {project_id}.")
            except Exception as e:
                print(f"Evaluation failed for {project_id}: {e}")
        # 3. Train and evaluate new model
        try:
            train_model(project_id)
            new_metrics_path = os.path.join(CURR_RESULTS_DIR, f"{project_id}.json")
            eval_model(project_id, new_metrics_path)
            new_metrics = load_metrics(new_metrics_path)
            if prev_metrics and new_metrics:
                diffs = compare_metrics(prev_metrics, new_metrics)
                if diffs:
                    msg = (
                        f"Reproducibility metric differences for {project_id} "
                        f"(> {THRESHOLD * 100:.1f}%): {diffs}"
                    )
                    print(msg)
                    if args.ci:
                        print(
                            f"::error file={project_id}::"
                            f"Reproducibility check failed: {msg}"
                        )
                    significant_diffs.append((project_id, "reproducibility", diffs))
                else:
                    print(
                        f"No significant reproducibility metric differences for "
                        f"{project_id}."
                    )
        except Exception as e:
            print(f"Training/evaluation failed for {project_id}: {e}")
    # Upload new models and metrics
    upload_models()
    upload_metrics()

    if args.ci and significant_diffs:
        print("\n::error::Significant metric differences found. Failing CI.")
        exit(1)


if __name__ == "__main__":
    main()
