# AI-Agent-for-SLICES-Infrastructure

![SLICES Infrastructure](https://img.shields.io/badge/Infrastructure-SLICES-blue)
![Architecture](https://img.shields.io/badge/Architecture-MCP_Protocol-orange)
![MLOps](https://img.shields.io/badge/Lifecycle-MLOps-success)

## 📖 About the Project

This project implements an autonomous **AI Agent** designed to execute a complete, end-to-end **MLOps cycle** on the SLICES Research Infrastructure. By acting as a "co-experimenter," the agent interprets scientific intents expressed in natural language (e.g., *"Run a Post5G xApp, generate traffic, train ML models, and publish the results"*) and orchestrates the entire workflow without manual intervention.

To ensure reliability, determinism, and to prevent LLM hallucinations, the system strictly separates the **planning phase** (handled by the LLM) from the **execution phase** (handled by a deterministic workflow engine).

### 🎯 Main Objectives
1. **Full Automation:** Eliminate manual intervention by orchestrating everything from node provisioning to metadata publication.
2. **Task Flexibility:** Allow the agent to perform smaller, isolated tasks (e.g., fetching an IP address, booking calendar slots, or searching for objects in the Model Registry System).
3. **Accessibility:** Simplify the developer experience and lower the barrier to entry for researchers using the SLICES infrastructure for the first time.

---

## 🏗️ Architecture & Core Philosophy

The agent is built around a modular architecture based on the **Model Context Protocol (MCP)** introduced by Anthropic. Rather than using a monolithic LLM script, the system uses distinct MCP Servers, ensuring:
* **Separation of Concerns:** Each MCP server encapsulates its own domain logic, credentials, and APIs. If the BI VM goes down, the SLICES session remains unaffected.
* **Replaceability:** Servers expose clean tool interfaces. Swapping MLflow for another tracker only requires updating one file.
* **Reusability:** MCP servers can be used standalone. A user can search a dataset without running the entire MLOps pipeline.

---

## ⚙️ Core Engine & Workflow Components

The system relies on a two-step process: **Planning** and **Execution**.

1. **`run_pipeline.py`**: The main orchestrator. It receives the user's natural language intent and coordinates the flow, but *does not* interface directly with MCP servers.
2. **`intent_layer.py`**: Uses an LLM to transform the natural language intent into an executable plan. It connects to MCP servers only to read their input schemas.
3. **`execution_spec.py`**: Validates the plan (checks for tool existence, unique IDs, correct dependencies, and missing parameters). It organizes tasks into parallel execution "waves".
4. **`workflow_executor.py`**: The execution engine. It takes the validated plan and calls the MCP tools in a deterministic, sequential manner. It stops at the first failure.
5. **`mrs_lifecycle.py`**: Responsible for recording *every* step on the Model Registry System (MRS) using PATCH operations to maintain a single source of truth for the experiment.

---

## 🔌 MCP Servers & Tools

The infrastructure relies on four specialized MCP Servers to handle different stages of the MLOps lifecycle.

### 1. SLICES Server (`mcp_server_slices.py`)
Manages the Post5G infrastructure setup and traffic generation.
* **`get_slices_session`**: Authenticates to the SLICES and Duckburg portals.
* **`slices_create_experiment`**: Creates (or reuses) the experiment space.
* **`post5g_get_prefix` & `configure_post5g_experiment`**: Retrieves IPs and configures the environment (nodes, MCC, MNC, etc.).
* **`book_pos_calendar`**: Reserves the required time slots on the infrastructure calendar.
* **`post5g_launch_experiment`**: Bootstraps the setup on remote nodes.
* **`trigger_5g_anomaly`**: Generates synthetic anomalous network traffic (e.g., DDoS) and downloads the resulting `.csv` logs.

### 2. Basic Infrastructure Server (`mcp_server_bi.py`)
Manages Virtual Machines for data processing and training.
* **`bi_create_mlops_vm`**: Deploys a VM (e.g., Ubuntu 22.04, `m1.small`) on the SLICES BI.
* **`bi_transfer_file_from_post5g`**: Moves the generated datasets/logs from the local machine to the remote BI VM.

### 3. MLflow Server (`mcp_server_mlflow.py`)
Handles the MLOps stack, model training, and experiment tracking.
* **`bi_deploy_mlops_stack`**: Deploys Docker containers for MLflow, MinIO, and PostgreSQL on the BI VM.
* **`bi_open_tunnels`**: Opens SSH tunnels to expose MLflow (port 5000) and MinIO (port 9000) locally.
* **`upload_csv_to_minio`**: Uploads the training data to the MinIO object storage.
* **`train_generic_model`**: Trains the dataset using multiple algorithms (**Random Forest, GradientBoosting, SVM, Logistic Regression**). Uses MLflow's `autolog()` to automatically record metrics (precision, recall, F1, accuracy) and saves the results.

### 4. MRS Server (`mcp_server_mrs.py`)
Interfaces with the Model Registry System for data sharing and provenance.
* **`publish_digital_object`**: Publishes the final dataset, models, and metadata to the MRS catalog.
* **`search_digital_object`**: Allows querying the MRS catalog by ID or description.

---

## 📊 Results & Performance

* **Reliability:** By separating the LLM planning phase from the deterministic execution phase, the system drastically reduces LLM hallucinations and token usage.
* **Speed:** The `execution_spec.py` groups independent tasks into "waves", allowing parallel execution and significantly cutting down the overall runtime (which previously took 10-15 minutes sequentially).
* **End-to-End Tracking:** Every training run is successfully logged in MLflow, datasets are stored safely in MinIO, and the final digital objects are seamlessly published to the SLICES MRS with complete metadata.

---

## 🚀 Getting Started

### Prerequisites
* Create an account and a project on SLICES Portal
* Create a .env file like this in the root directory
```bash
SLICES_USER=
SLICES_PASS=
SSH_KEY_PATH=
LLM_MODEL=
LLM_API_KEY=
LLM_BASE_URL=
MRS_CLI_CLIENT_ID=
MRS_CLI_CLIENT_SECRET=
MRS_CLI_USERNAME=
MRS_CLI_PASSWORD=
```

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install the libraries from requirements.txt file
pip install -r requirements.txt

# Install SLICES CLI
pip install slices-cli --extra-index-url=https://doc.slices-ri.eu/pypi/

# Install SLICES CLI DM library
pip install slices-cli-dm --index-url https://gitlab.inria.fr/api/v4/projects/65212/packages/pypi/simple --extra-index-url=https://doc.slices-ri.eu/pypi/
```

### Usage
```bash
# Add run commands here
```

---
**Authors & Credits:**
* Prof. Serge Fdida - serge.fdida@lip6.fr
* Flavio Olivieri - flavio.olivieri@lip6.fr
* Sorbonne Université - Sciences LIP6 Laboratory
