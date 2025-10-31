import json
import os
import subprocess

import click

HUB_REPO = "juhoinkinen/Annif-models-compat"
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


def run_cmd(cmd, check=True, silent=False):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not silent:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        if check:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def download_models():
    cmd = ["annif", "download", "*", HUB_REPO, "-f", "--trust-repo"]
    try:
        run_cmd(cmd, check=False)
    except Exception as e:
        print(f"Download failed: {e}")


def download_metrics():
    cmd = ["hf", "download", HUB_REPO, "--include", "metrics/*", "--local-dir", "./"]
    try:
        run_cmd(cmd, check=False, silent=True)
    except Exception as e:
        print(f"Download failed: {e}")


def upload_models():
    cmd = ["annif", "upload", "*", HUB_REPO]
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
    cmd = [
        "annif",
        "eval",
        project_id,
        CORPORA_DIR,
        "--metrics-file",
        result_file,
    ]
    run_cmd(cmd)


def train_model(project_id):
    cmd = ["annif", "train", project_id, CORPORA_DIR]
    run_cmd(cmd)


def load_metrics(metrics_path):
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def compare_metrics(metrics1, metrics2, threshold):
    diffs = {}
    for key in metrics1:
        if (
            key in metrics2
            and isinstance(metrics1[key], (int, float))
            and isinstance(metrics2[key], (int, float))
        ):
            v1, v2 = metrics1[key], metrics2[key]
            if v1 == 0:
                if v2 != 0:
                    rel_diff = 1.0
            else:
                rel_diff = abs(v1 - v2) / abs(v1)
            if rel_diff > threshold:
                diffs[key] = (v1, v2, rel_diff)
    return diffs


def check_project_metrics(
    ci, threshold, significant_diffs, project_id, prev_metrics, check_type
):
    curr_metrics_path = os.path.join(CURR_RESULTS_DIR, f"{project_id}.json")
    try:
        eval_model(project_id, curr_metrics_path)
        curr_metrics = load_metrics(curr_metrics_path)
        diffs = compare_metrics(prev_metrics, curr_metrics, threshold)
        if diffs:
            msg = (
                f"❌ {check_type.capitalize()} differences for {project_id} "
                f"(> {threshold * 100:.1f}%): {diffs}"
            )
            if ci:
                print(
                    f"::error file={project_id}::{check_type.capitalize()} "
                    f"check failed: {msg}"
                )
            else:
                print(msg + "\n")
            significant_diffs.append((project_id, check_type, diffs))
        else:
            print(f"✅ No significant {check_type} differences for {project_id}.\n")
    except Exception as e:
        print(f"❗ Evaluation failed for {project_id}: {e}\n")


def check(check_type, ci, train=False):
    projects_cfg_name = f"tests/projects-{check_type}.cfg"
    project_ids = get_project_ids(projects_cfg_name)
    os.environ["ANNIF_PROJECTS"] = projects_cfg_name
    significant_diffs = []
    for project_id in project_ids:
        print(f"=== Checking {check_type} of project {project_id} ===")
        prev_metrics_path = os.path.join(PREV_RESULTS_DIR, f"{project_id}.json")
        prev_metrics = load_metrics(prev_metrics_path)
        if train:
            train_model(project_id)
        if prev_metrics is not None:
            check_project_metrics(
                ci, THRESHOLD, significant_diffs, project_id, prev_metrics, check_type
            )
        else:
            print(f"❔ No previous metrics for {project_id}, skipping check.\n")
    if ci and significant_diffs:
        print("::error::Significant metric differences found. Failing CI.")
        exit(1)


@click.group()
def cli():
    pass


@cli.command("compatibility")
@click.option("--ci", is_flag=True, help="Enable CI mode for GitHub Actions")
def run_compatibility_checks(ci):
    check("compatibility", ci)


@cli.command("consistency")
@click.option("--ci", is_flag=True, help="Enable CI mode for GitHub Actions")
@click.option(
    "--upload", is_flag=True, help="Upload new models and metrics to Hugging Face Hub"
)
def run_consistency_checks(ci, upload):
    check("consistency", ci, train=True)
    if upload:
        print("\nUploading new models and metrics to Hugging Face Hub.")
        upload_models()
        upload_metrics()


if __name__ == "__main__":
    setup_dirs()
    download_models()  # Downloads also the vocabularies
    download_metrics()
    cli()
