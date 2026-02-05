import json
import os
import subprocess
import sys
import tempfile
from glob import glob

import click

CORPORA_DIR = "tests/corpora/archaeology/fulltext/"
THRESHOLD_EVAL_REPRO = 0.01
THRESHOLD_TRAIN_REPRO = 0.03

# Explicit list of metrics to compare to avoid silent regressions
METRICS = ["F1@5", "NDCG"]


def get_project_ids(projects_cfg_dir):
    fpaths = glob(os.path.join(projects_cfg_dir, "*.cfg"))
    ids = []
    for fpath in sorted(fpaths):
        with open(fpath) as f:
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


def download_models(hf_repo):
    cmd = ["annif", "download", "*", hf_repo, "-f", "--trust-repo"]
    run_cmd(cmd, check=True)


def download_metrics(hf_repo):
    cmd = [
        "hf",
        "download",
        hf_repo,
        "--include",
        "metrics/*",
        "--local-dir",
        PREV_METRICS_DIR,
    ]
    try:
        run_cmd(cmd, check=False, silent=True)
    except Exception as e:
        print(f"Download failed: {e}")
        raise
    # Move downloaded metrics to PREV_RESULTS_DIR/metrics
    downloaded_metrics_dir = os.path.join(PREV_METRICS_DIR, "metrics")
    for filename in os.listdir(downloaded_metrics_dir):
        src_path = os.path.join(downloaded_metrics_dir, filename)
        dest_path = os.path.join(PREV_METRICS_DIR, filename)
        os.rename(src_path, dest_path)
    os.rmdir(downloaded_metrics_dir)


def upload_models(hf_repo):
    cmd = ["annif", "upload", "*", hf_repo]
    try:
        run_cmd(cmd, check=False)
    except Exception as e:
        print(f"Upload failed: {e}")


def upload_metrics(hf_repo):
    cmd = ["hf", "upload", hf_repo, CURR_METRICS_DIR, "metrics/"]
    try:
        run_cmd(cmd, check=False)
    except Exception as e:
        print(f"Upload failed: {e}")


def eval_model(project_id, metrics_path):
    cmd = [
        "annif",
        "eval",
        project_id,
        CORPORA_DIR,
        "--metrics-file",
        metrics_path,
    ]
    for metric in METRICS:
        cmd.extend(["--metric", metric])
    run_cmd(cmd)

    # Wrap raw metrics with metadata for reproducibility
    raw = load_metrics(metrics_path)
    if raw is not None:
        save_metrics(metrics_path, raw)


def train_model(project_id):
    cmd = ["annif", "train", project_id, CORPORA_DIR]
    run_cmd(cmd)


def load_vocab():
    cmd = [
        "annif",
        "load-vocab",
        "yso",
        "tests/corpora/archaeology/yso-archaeology.ttl",
        "--force",
    ]
    run_cmd(cmd, check=False)


def get_env_metadata():
    annif_version = "unknown"
    try:
        res = subprocess.run(
            ["annif", "--version"], capture_output=True, text=True, check=False
        )
        if res.stdout:
            annif_version = res.stdout.strip()
    except Exception:
        pass

    return {
        "annif_version": annif_version,
        "python_version": sys.version.split()[0],
    }


def load_metrics(metrics_path):
    try:
        with open(metrics_path) as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        return None


def save_metrics(metrics_path, metrics):
    payload = {
        "metrics": metrics,
        "meta": get_env_metadata(),
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def compare_metrics(metrics1, metrics2, threshold):
    diffs = {}

    m1 = metrics1["metrics"]
    m2 = metrics2["metrics"]

    for metric in METRICS:
        if metric not in m1 or metric not in m2:
            raise KeyError(
                f"Required metric '{metric}' missing "
                f"(prev={metric in m1}, curr={metric in m2})"
            )

        v1, v2 = m1[metric], m2[metric]
        if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
            raise TypeError(
                f"Metric '{metric}' must be numeric (got {type(v1)}, {type(v2)})")

        if v1 == 0:
            rel_diff = 0.0 if v2 == 0 else 1.0
        else:
            rel_diff = abs(v1 - v2) / abs(v1)

        if rel_diff > threshold:
            diffs[metric] = (v1, v2, rel_diff)

    return diffs


def check_project_metrics(
    ci, threshold, significant_diffs, project_id, prev_metrics, check_type
):
    curr_metrics_path = os.path.join(CURR_METRICS_DIR, f"{project_id}.json")
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
    if check_type == "eval_repro":
        projects_cfg_dir = f"tests/projects-all"
        threshold = THRESHOLD_EVAL_REPRO
    else:
        projects_cfg_dir = f"tests/projects-trainable"
        threshold = THRESHOLD_TRAIN_REPRO

    project_ids = get_project_ids(projects_cfg_dir)
    os.environ["ANNIF_PROJECTS"] = projects_cfg_dir
    significant_diffs = []
    for project_id in sorted(project_ids):
        print(f"=== Checking {check_type} of project {project_id} ===")
        prev_metrics_path = os.path.join(PREV_METRICS_DIR, f"{project_id}.json")
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
        sys.exit(1)


@click.group()
def cli():
    pass


@cli.command("eval_repro")
@click.option("--ci", is_flag=True, help="Enable CI mode for GitHub Actions")
@click.option("--hf_repo", required=True, envvar="HF_REPO")
def run_eval_repro_checks(ci, hf_repo):
    download_models(hf_repo)
    download_metrics(hf_repo)
    check_projects("eval_repro", ci)


@cli.command("train_repro")
@click.option("--ci", is_flag=True, help="Enable CI mode for GitHub Actions")
@click.option("--hf_repo", required=True, envvar="HF_REPO")
def run_train_repro_checks(ci, hf_repo):
    download_models(hf_repo)
    download_metrics(hf_repo)
    check_projects("train_repro", ci, train=True)


@cli.command("upload")
@click.option("--hf_repo", required=True, envvar="HF_REPO")
def run_upload(hf_repo):
    # 1) Train and evaluate trainable projects
    projects_cfg_name = "tests/projects-trainable"
    trainable_project_ids = get_project_ids(projects_cfg_name)
    os.environ["ANNIF_PROJECTS"] = projects_cfg_name

    load_vocab()
    for project_id in trainable_project_ids:
        print(f"=== Training project {project_id} ===")
        train_model(project_id)

    # 2) Evaluate all projects
    projects_cfg_name = "tests/projects-all"
    os.environ["ANNIF_PROJECTS"] = projects_cfg_name
    all_project_ids = get_project_ids(projects_cfg_name)

    for project_id in all_project_ids:
        print(f"=== Evaluating project {project_id} ===")
        curr_metrics_path = os.path.join(CURR_METRICS_DIR, f"{project_id}.json")
        eval_model(project_id, curr_metrics_path)

    print("Uploading new models and metrics to Hugging Face Hub.")
    upload_models(hf_repo)
    upload_metrics(hf_repo)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        CURR_METRICS_DIR = os.path.join(temp_dir, "curr_metrics")
        PREV_METRICS_DIR = os.path.join(temp_dir, "prev_metrics")
        os.makedirs(CURR_METRICS_DIR)
        os.makedirs(PREV_METRICS_DIR)
        os.environ["ANNIF_DATADIR"] = os.path.join(temp_dir, "data")
        cli()
