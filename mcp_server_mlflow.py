import os
import json
import re
import time
import subprocess
import boto3
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from typing import Optional
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

load_dotenv()

mcp = FastMCP("mlflow")

_tunnel_procs: list = []

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_cmd(args: list, timeout: int = 300) -> dict:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"returncode": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"returncode": -1, "stdout": "", "stderr": str(e)}


def _bi_ssh(experiment_name: str, vm_name: str, command: str, timeout: int = 300) -> dict:
    """Run a command on the BI VM via 'slices bi ssh'."""
    return _run_cmd(
        ["slices", "bi", "ssh", "--experiment", experiment_name, vm_name, "--", command],
        timeout=timeout,
    )


def _parse_jump_host(experiment_name: str, vm_name: str) -> dict:
    """
    Parse the jump-host and VM IP from 'slices bi ssh' output.
    slices bi ssh prints the SSH command on the first line before executing.
    Example: "ssh -J proxy@bastion2.slices-be.eu ubuntu@10.10.221.190 echo ok"
    Returns {"jump_host": "proxy@bastion2.slices-be.eu", "vm_ip": "10.10.221.190", "vm_user": "ubuntu"}
    """
    r = _run_cmd(
        ["slices", "bi", "ssh", "--experiment", experiment_name, vm_name, "echo", "ok"],
        timeout=30,
    )
    stdout = r["stdout"]

    # First line contains the SSH command echo
    first_line = stdout.splitlines()[0] if stdout else ""

    jump_match = re.search(r'-J\s+(\S+)', first_line)
    user_ip_match = re.search(r'(\w+)@((?:\d{1,3}\.){3}\d{1,3})', first_line)

    if not jump_match or not user_ip_match:
        return {"error": f"Could not parse SSH command from: {first_line!r}"}

    return {
        "jump_host": jump_match.group(1),
        "vm_user":   user_ip_match.group(1),
        "vm_ip":     user_ip_match.group(2),
    }


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def bi_deploy_mlops_stack(
    experiment_name: str,
    vm_name: str = "mlops-server",
) -> str:
    """
    Install Docker and deploy MLflow + MinIO + PostgreSQL on the BI VM.
    Call this once after bi_create_mlops_vm. Takes ~3 minutes.

    Services inside the VM:
      MLflow   → :5000
      MinIO    → :9000 / :9001
      Postgres → :5432
    """
    docker_compose = (
        "services:\\n"
        "  s3_dmi:\\n"
        "    image: minio/minio\\n"
        "    ports: ['9000:9000', '9001:9001']\\n"
        "    environment:\\n"
        "      - MINIO_ROOT_USER=admin\\n"
        "      - MINIO_ROOT_PASSWORD=password123\\n"
        "    command: server /data --console-address ':9001'\\n"
        "    volumes: [minio_data:/data]\\n"
        "  db_mrs:\\n"
        "    image: postgres:13\\n"
        "    environment:\\n"
        "      - POSTGRES_USER=mlflow_user\\n"
        "      - POSTGRES_PASSWORD=mlflow_password\\n"
        "      - POSTGRES_DB=mlflow_db\\n"
        "    ports: ['5432:5432']\\n"
        "    volumes: [pg_data:/var/lib/postgresql/data]\\n"
        "  mlflow:\\n"
        "    image: python:3.9-slim\\n"
        "    ports: ['5000:5000']\\n"
        "    environment:\\n"
        "      - AWS_ACCESS_KEY_ID=admin\\n"
        "      - AWS_SECRET_ACCESS_KEY=password123\\n"
        "      - MLFLOW_S3_ENDPOINT_URL=http://s3_dmi:9000\\n"
        "    depends_on: [db_mrs]\\n"
        "    command: >\\n"
        "      bash -c \\\"pip install mlflow psycopg2-binary boto3 &&\\n"
        "      mlflow server --host 0.0.0.0 --port 5000\\n"
        "      --backend-store-uri postgresql://mlflow_user:mlflow_password@db_mrs:5432/mlflow_db\\n"
        "      --default-artifact-root s3://mlflow-artifacts/\\\"\\n"
        "volumes:\\n"
        "  minio_data:\\n"
        "  pg_data:\\n"
    )

    steps = [
        ("write docker-compose.yml",
         f"printf '{docker_compose}' > /home/ubuntu/docker-compose.yml", 60),
        ("install Docker",
         "sudo apt-get update -qq && sudo apt-get install -y -qq docker.io docker-compose", 300),
        ("start stack",
         "cd /home/ubuntu && sudo docker-compose up -d", 120),
        ("wait for MLflow",
         "sleep 60 && curl -sf http://localhost:5000/health && echo OK || echo 'not ready yet'", 90),
    ]

    for step_name, cmd, timeout in steps:
        r = _bi_ssh(experiment_name, vm_name, cmd, timeout=timeout)
        if r["returncode"] != 0:
            return json.dumps({
                "success":     False,
                "failed_step": step_name,
                "error":       r["stderr"],
                "stdout":      r["stdout"][:300],
            }, indent=2)

    return json.dumps({
        "success":   True,
        "message":   "MLOps stack deployed.",
        "next_step": "Call bi_open_tunnels to expose MLflow and MinIO locally.",
    }, indent=2)


@mcp.tool()
def bi_open_tunnels(
    experiment_name: str,
    vm_name: str = "mlops-server",
) -> str:
    """
    Open SSH tunnels to the BI VM so MLflow (:5000) and MinIO (:9000) are
    reachable at localhost. Uses the jump-host parsed from 'slices bi ssh'.
    Must be called before upload_csv_to_minio and train_generic_model.
    """
    global _tunnel_procs

    # Terminate any existing tunnels
    for p in _tunnel_procs:
        try:
            p.terminate()
        except Exception:
            pass
    _tunnel_procs = []

    # Parse jump-host info from slices bi ssh
    conn = _parse_jump_host(experiment_name, vm_name)
    if "error" in conn:
        return json.dumps({"success": False, "error": conn["error"]}, indent=2)

    jump_host = conn["jump_host"]
    vm_ip     = conn["vm_ip"]
    vm_user   = conn["vm_user"]

    # Open a single SSH tunnel for both ports
    tunnel_cmd = [
        "ssh", "-N",
        "-J",  jump_host,
        "-L",  f"5000:localhost:5000",
        "-L",  f"9000:localhost:9000",
        "-o",  "StrictHostKeyChecking=no",
        "-o",  "ServerAliveInterval=30",
        "-o",  "ExitOnForwardFailure=yes",
        f"{vm_user}@{vm_ip}",
    ]
    p = subprocess.Popen(tunnel_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _tunnel_procs.append(p)

    # Wait for tunnel to establish
    time.sleep(6)

    # Verify connectivity
    check = _run_cmd(["curl", "-sf", "--max-time", "5", "http://localhost:5000/health"])
    mlflow_ok = check["returncode"] == 0

    return json.dumps({
        "success":         True,
        "jump_host":       jump_host,
        "vm_ip":           vm_ip,
        "tunnels": [
            {"service": "MLflow", "local_url": "http://localhost:5000"},
            {"service": "MinIO",  "local_url": "http://localhost:9000"},
        ],
        "mlflow_reachable": mlflow_ok,
        "tracking_ip":      "localhost",
        "hint": "Pass tracking_ip='localhost' to upload_csv_to_minio and train_generic_model.",
    }, indent=2)


@mcp.tool()
def upload_csv_to_minio(
    local_csv_path: str,
    tracking_ip: str = "localhost",
    bucket: str = "datasets",
    public_ip: str = None,
) -> str:
    """
    Upload a local CSV to MinIO (reachable via tunnel from bi_open_tunnels).
    Returns dataset_url to pass directly to train_generic_model.

    Parameters:
      local_csv_path : local path e.g. /tmp/anomaly_log.csv
      tracking_ip    : 'localhost' when tunnel is active (default) — used for the
                        actual upload connection, which must go through the tunnel.
      bucket         : MinIO bucket (created automatically if missing)
      public_ip      : VM's real IP (from bi_open_tunnels' vm_ip). Used ONLY to build
                        dataset_public_url, a display/reference link — NOT dataset_url.
                        dataset_url must stay tracking_ip-based: train_generic_model
                        reconnects to that exact host to actually download the file,
                        and public_ip is only reachable via tunnel/bastion, not directly.
    """
    endpoint = f"http://{tracking_ip}:9000"
    s3 = boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id="admin",
        aws_secret_access_key="password123",
    )

    try:
        s3.create_bucket(Bucket=bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("BucketAlreadyExists", "BucketAlreadyOwnedByYou"):
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    filename = os.path.basename(local_csv_path)
    try:
        s3.upload_file(local_csv_path, bucket, filename)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Upload failed: {e}"}, indent=2)

    # dataset_url MUST stay tunnel-based — train_generic_model parses its host
    # back out and reconnects to it directly to download the file.
    dataset_url = f"{endpoint}/{bucket}/{filename}"
    # dataset_public_url is display-only (e.g. for a DO's downloadUrl) — never
    # fed back into another tool that opens a real connection to it.
    dataset_public_url = f"http://{public_ip}:9000/{bucket}/{filename}" if public_ip else dataset_url
    return json.dumps({
        "success":            True,
        "bucket":             bucket,
        "filename":           filename,
        "dataset_url":        dataset_url,
        "dataset_public_url": dataset_public_url,
        "hint":        f"Pass dataset_url='{dataset_url}' and target_column=<col> to train_generic_model.",
    }, indent=2)


@mcp.tool()
def train_generic_model(
    dataset_url: str,
    target_column: str,
    tracking_ip: str = "localhost",
    experiment_name: str = "AnomalyDetection",
    public_ip: str = None,
) -> str:
    """
    Train 4 models (RandomForest, GradientBoosting, SVM, LogisticRegression)
    on a CSV from MinIO. Logs all runs to MLflow via autolog (same as the notebook).
    Selects the best model by training_score, registers it as Production.
    Saves a results JSON and uploads it to MinIO.

    Parameters:
      dataset_url     : URL from upload_csv_to_minio
      target_column   : label column name in the CSV (e.g. 'status')
      tracking_ip     : 'localhost' when tunnel active (default) — used for the actual
                        MLflow/MinIO connections, which must go through the tunnel.
      experiment_name : MLflow experiment name (default: AnomalyDetection)
      public_ip       : VM's real IP (from bi_open_tunnels' vm_ip), used only to build
                        the *returned* results_minio_url so it stays valid beyond your
                        local tunnel (e.g. for publish_digital_object's downloadUrl).
                        Falls back to tracking_ip if not given.
    """
    # ── MLflow + MinIO env ────────────────────────────────────────────────────
    os.environ["AWS_ACCESS_KEY_ID"]      = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"]  = "password123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{tracking_ip}:9000"

    mlflow.set_tracking_uri(f"http://{tracking_ip}:5000")
    mlflow.set_experiment(experiment_name)

    # Ensure artifact bucket exists
    s3 = boto3.client(
        "s3", endpoint_url=f"http://{tracking_ip}:9000",
        aws_access_key_id="admin", aws_secret_access_key="password123",
    )
    for bucket in ("mlflow-artifacts", "results"):
        try:
            s3.create_bucket(Bucket=bucket)
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("BucketAlreadyExists", "BucketAlreadyOwnedByYou"):
                return json.dumps({"success": False, "error": f"Cannot create bucket: {e}"}, indent=2)

    # ── Load dataset from MinIO via boto3 (pd.read_csv can't auth) ──────────
    try:
        from urllib.parse import urlparse
        parsed   = urlparse(dataset_url)
        # e.g. http://localhost:9000/datasets/anomaly_log.csv
        # Usa l'endpoint estratto da dataset_url (non tracking_ip) per il
        # download: i due potrebbero differire se l'LLM ha passato tracking_ip
        # diverso nei due task, causando HeadObject 404 perché il client S3
        # è connesso all'endpoint sbagliato rispetto a dove il file è stato caricato.
        minio_endpoint = f"{parsed.scheme}://{parsed.netloc}"
        s3_dl = boto3.client(
            "s3", endpoint_url=minio_endpoint,
            aws_access_key_id="admin", aws_secret_access_key="password123",
        )
        bucket   = parsed.path.lstrip("/").split("/")[0]
        key      = "/".join(parsed.path.lstrip("/").split("/")[1:])
        local_tmp = f"/tmp/_mlflow_dataset_{key.replace('/', '_')}"
        s3_dl.download_file(bucket, key, local_tmp)
        df = pd.read_csv(local_tmp)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Cannot load dataset: {e}"}, indent=2)

    # The Post-5G anomaly dataset's schema is NOT deterministic across runs (the
    # simulator has produced ['anomaly','timestamp','type'] in one run and
    # ['timestamp','target','action','latency','status'] in another for the
    # same intent), so a target_column name supplied up front — whether typed
    # by the user or picked by the intent layer — is inherently a guess. Rather
    # than failing outright on a mismatch, try a short list of common label
    # column names before giving up, and report which one was actually used.
    resolved_target_column = target_column
    requested_target_column = target_column
    if resolved_target_column not in df.columns:
        FALLBACK_TARGET_CANDIDATES = [
            "status", "anomaly", "target", "label", "class", "is_anomaly", "anomaly_type",
        ]
        match = next(
            (c for c in FALLBACK_TARGET_CANDIDATES if c in df.columns and c != target_column),
            None,
        )
        if match is None:
            return json.dumps({
                "success": False,
                "error": (
                    f"Column '{target_column}' not found, and none of the common label "
                    f"column names {FALLBACK_TARGET_CANDIDATES} were found either. "
                    f"Available: {list(df.columns)}"
                ),
            }, indent=2)
        resolved_target_column = match
    target_column = resolved_target_column

    # ── Feature engineering ───────────────────────────────────────────────────
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["ts_delta"]        = df["timestamp"].diff().fillna(0)
        df["ts_rolling_rate"] = (
            df["timestamp"].rolling(10, min_periods=1).count() /
            df["timestamp"].rolling(10, min_periods=1)
            .apply(lambda x: max(x.iloc[-1] - x.iloc[0], 1e-9), raw=False)
        )

    X = df.drop(columns=[target_column]).select_dtypes(include="number")
    X = X.loc[:, X.nunique() > 1]
    y = df[target_column]

    if X.empty:
        return json.dumps({"success": False,
                           "error": "No usable numeric features after dropping constants."}, indent=2)

    if y.dtype == object:
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(y), name=target_column)

    class_counts  = y.value_counts().to_dict()
    is_imbalanced = max(class_counts.values()) / max(min(class_counts.values()), 1) > 3
    cw = "balanced" if is_imbalanced else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train — same as notebook ──────────────────────────────────────────────
    modelli = {
        "RandomForest":       RandomForestClassifier(n_estimators=300, max_depth=3,
                                                      class_weight=cw, random_state=42),
        "GradientBoosting":   GradientBoostingClassifier(n_estimators=300, random_state=42),
        "SVM":                SVC(kernel="rbf", probability=True,
                                  class_weight=cw, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500,
                                                  class_weight=cw, random_state=42),
    }

    run_ids      = {}
    train_scores = {}
    test_scores  = {}
    f1_scores    = {}
    reports      = {}

    # autolog logs training_score automatically (same as notebook)
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=False, silent=True)

    for nome, modello in modelli.items():
        with mlflow.start_run(run_name=nome) as run:
            modello.fit(X_train, y_train)
            y_pred = modello.predict(X_test)

            train_score = modello.score(X_train, y_train)
            test_score  = modello.score(X_test,  y_test)
            f1          = f1_score(y_test, y_pred, average="macro", zero_division=0)

            # Log additional metrics on top of autolog
            mlflow.log_metric("test_accuracy", test_score)
            mlflow.log_metric("f1_macro",      f1)

            run_ids[nome]      = run.info.run_id
            train_scores[nome] = train_score
            test_scores[nome]  = test_score
            f1_scores[nome]    = f1
            reports[nome]      = classification_report(y_test, y_pred, zero_division=0)

            print(f"{nome} — train: {train_score:.3f}  test: {test_score:.3f}  f1: {f1:.3f}")

    # ── Best model by training_score (same as notebook) ───────────────────────
    best_name     = max(train_scores, key=train_scores.get)
    best_run_id   = run_ids[best_name]

    model_uri        = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(model_uri, f"{experiment_name}_BestModel")

    client = MlflowClient(tracking_uri=f"http://{tracking_ip}:5000")
    client.transition_model_version_stage(
        name=f"{experiment_name}_BestModel",
        version=registered_model.version,
        stage="Production",
    )

    # ── Results table (same as notebook) ─────────────────────────────────────
    all_scores = [
        {
            "model":          k,
            "training_score": round(train_scores[k], 4),
            "test_accuracy":  round(test_scores[k],  4),
            "f1_macro":       round(f1_scores[k],    4),
        }
        for k in sorted(train_scores, key=train_scores.get, reverse=True)
    ]

    result = {
        "success": True,
        "experiment_name": experiment_name,
        "dataset": {
            "rows":          int(df.shape[0]),
            "features_used": list(X.columns),
            "target":        target_column,
            "class_counts":  {str(k): int(v) for k, v in class_counts.items()},
            "imbalanced":    is_imbalanced,
        },
        "best_model": {
            "name":             best_name,
            "training_score":   round(train_scores[best_name], 4),
            "test_accuracy":    round(test_scores[best_name],  4),
            "f1_macro":         round(f1_scores[best_name],    4),
            "run_id":           best_run_id,
            "version":          registered_model.version,
            "stage":            "Production",
            "classification_report": reports[best_name],
        },
        "all_models": all_scores,
        "mlflow_ui":  f"http://{tracking_ip}:5000",
    }

    if target_column != requested_target_column:
        result["dataset"]["target_column_note"] = (
            f"Requested target_column '{requested_target_column}' was not present in this "
            f"run's dataset; used fallback column '{target_column}' instead."
        )

    # ── Save + upload results to MinIO ────────────────────────────────────────
    results_path = "/tmp/training_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    try:
        s3.upload_file(results_path, "results", "training_results.json")
        url_host = public_ip or tracking_ip
        result["results_minio_url"] = f"http://{url_host}:9000/results/training_results.json"
    except Exception as e:
        result["results_upload_warning"] = str(e)

    return json.dumps(result, indent=2)

@mcp.tool()
def download_artifact_from_minio(
    bucket: str = "results",
    filename: str = "training_results.json",
    tracking_ip: str = "localhost",
    local_dest: str = "/tmp/training_results.json",
) -> str:
    """
    Download a file from MinIO (reachable via tunnel from bi_open_tunnels) to a
    LOCAL path on this machine. Call this AFTER train_generic_model and BEFORE
    publish_digital_object, to get a real local file you can attach as artifact_path.

    Defaults match what train_generic_model uploads: bucket="results",
    filename="training_results.json". Returns local_path to pass directly
    to publish_digital_object's artifact_path.
    """
    endpoint = f"http://{tracking_ip}:9000"
    s3 = boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id="admin",
        aws_secret_access_key="password123",
    )
    try:
        s3.download_file(bucket, filename, local_dest)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Download failed: {e}"}, indent=2)

    return json.dumps({
        "success":    True,
        "bucket":     bucket,
        "filename":   filename,
        "local_path": local_dest,
        "hint":       f"Pass artifact_path='{local_dest}' to publish_digital_object.",
    }, indent=2)

@mcp.tool()
def bi_close_tunnels() -> str:
    """Terminate all SSH tunnels opened by bi_open_tunnels."""
    global _tunnel_procs
    closed = sum(1 for p in _tunnel_procs if not p.terminate())
    _tunnel_procs = []
    return json.dumps({"success": True, "tunnels_closed": closed}, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")