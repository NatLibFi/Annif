import json
import os
import subprocess
import tempfile

import click

HUB_REPO = "juhoinkinen/Annif-models-compat"
CORPORA_DIR = "tests/corpora/archaeology/fulltext/"
THRESHOLD_COMPATIBILITY = 0.01
THRESHOLD_CONSISTENCY = 0.03


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
    cmd = [
        "hf",
        "download",
        HUB_REPO,
        "--include",
        "metrics/*",
        "--local-dir",
        PREV_RESULTS_DIR,
    ]
    try:
        run_cmd(cmd, check=False, silent=True)
    except Exception as e:
        print(f"Download failed: {e}")
    # Move downloaded metrics to PREV_RESULTS_DIR/metrics
    downloaded_metrics_dir = os.path.join(PREV_RESULTS_DIR, "metrics")
    for filename in os.listdir(downloaded_metrics_dir):
        src_path = os.path.join(downloaded_metrics_dir, filename)
        dest_path = os.path.join(PREV_RESULTS_DIR, filename)
        os.rename(src_path, dest_path)
    os.rmdir(downloaded_metrics_dir)


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
        "--metric",
        "F1@5",
        "--metric",
        "NDCG",
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


def check_projects(check_type, ci, train=False):
    projects_cfg_name = f"tests/projects-{check_type}.cfg"
    threshold = (
        THRESHOLD_COMPATIBILITY
        if check_type == "compatibility"
        else THRESHOLD_CONSISTENCY
    )
    project_ids = get_project_ids(projects_cfg_name)
    os.environ["ANNIF_PROJECTS"] = projects_cfg_name
    significant_diffs = []
    for project_id in project_ids:
        print(f"=== Checking {check_type} of project {project_id} ===")
        prev_metrics_path = os.path.join(PREV_RESULTS_DIR, f"{project_id}.json")
        prev_metrics = load_metrics(prev_metrics_path)
        if prev_metrics is not None:
            if train:
                train_model(project_id)
            check_project_metrics(
                ci, threshold, significant_diffs, project_id, prev_metrics, check_type
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
    download_models()
    download_metrics()
    check_projects("compatibility", ci)


@cli.command("consistency")
@click.option("--ci", is_flag=True, help="Enable CI mode for GitHub Actions")
def run_consistency_checks(ci):
    download_models()
    download_metrics()
    check_projects("consistency", ci, train=True)


@cli.command("upload")
def run_upload():
    print("Training new models and evaluating metrics.")
    projects_cfg_name = "tests/projects-consistency.cfg"
    project_ids = get_project_ids(projects_cfg_name)
    os.environ["ANNIF_PROJECTS"] = projects_cfg_name

    for project_id in project_ids:
        print(f"=== Project {project_id} ===")
        train_model(project_id)
        curr_metrics_path = os.path.join(CURR_RESULTS_DIR, f"{project_id}.json")
        eval_model(project_id, curr_metrics_path)

    print("Uploading new models and metrics to Hugging Face Hub.")
    upload_models()
    upload_metrics()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        CURR_RESULTS_DIR = os.path.join(temp_dir, "curr_metrics")
        PREV_RESULTS_DIR = os.path.join(temp_dir, "prev_metrics")
        os.makedirs(CURR_RESULTS_DIR, exist_ok=True)
        os.makedirs(PREV_RESULTS_DIR, exist_ok=True)
        os.environ["ANNIF_DATADIR"] = os.path.join(temp_dir, "data")
        cli()
