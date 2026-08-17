from __future__ import annotations

import json
import os
import re
import sys
from contextlib import AsyncExitStack
from typing import Any
from datetime import datetime, UTC

import httpx
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from execution_spec import (
    ExecutionSpecification,
    Task,
    MCP_SERVERS,
    ResourceKind,
)

load_dotenv()

MAX_GENERATION_ATTEMPTS = 5   # retries for a single phase's task-list generation
MAX_PLANNING_ATTEMPTS = 3     # retries for the coarse phase plan
MAX_MERGE_RETRIES = 2         # retries for fixing a single offending phase after merge

SERVERS = {
    "slices": "mcp_server_slices.py",
    "bi":     "mcp_server_bi.py",
    "mlflow": "mcp_server_mlflow.py",
    "mrs":    "mcp_server_mrs.py",
}

_LIFECYCLE_ONLY_TOOLS = {"record_lifecycle_event", "get_lifecycle_events"}
_REF_PATTERN = re.compile(r"^\$\{([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\}$")


class IntentLayerError(Exception):
    """Raised when the intent layer cannot produce a valid ExecutionSpecification."""


# ══════════════════════════════════════════════════════════════════════════
# TOOL CATALOG — per-tool one-line descriptions used to build the catalog
# shown to the LLM. Kept separate from execution logic so it stays easy to
# extend when a new MCP server is registered.
# ══════════════════════════════════════════════════════════════════════════

TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_slices_session": "Login (first step). Produces: session, mod_auth_openidc_session.",
    "slices_create_experiment": "Create experiment. Produces: experiment_name, experiment_token.",
    "post5g_get_prefix": "Get network info. Needs: experiment_name. Produces: nrf_lb_ip, multus_network, lb.",
    "get_available_nodes": "List nodes.",
    "configure_post5g_experiment": "Config. Needs: session_cookie, mod_auth_openidc_session, experiment_name, pos_deployment_node, nrf_lb_ip, multus_network.",
    "book_pos_calendar": "Reserve node. Needs: mod_auth_openidc_session, node, start_time, end_time.",
    "post5g_get_experiment": "Check exp status. Needs: experiment_name.",
    "post5g_launch_experiment": "Launch exp. Needs: experiment_name.",
    "trigger_5g_anomaly": "Generate CSV anomaly. Needs: lb_ip. Produces: log_file_local.",
    "bi_list_infra": "List BI infra.",
    "bi_create_mlops_vm": "Create VM. Needs: experiment_name, vm_name.",
    "bi_wait_vm_ready": "Wait for VM SSH. Needs: experiment_name, vm_name.",
    "bi_transfer_file_from_post5g": "Transfer CSV to VM. Needs: experiment_name, local_file, vm_name. Produces: destination, source, success, message.",
    "bi_deploy_mlops_stack": "Install MLOps on VM. Needs: experiment_name, vm_name.",
    "bi_open_tunnels": "Open tunnels. Needs: experiment_name, vm_name. Produces: tracking_ip, vm_ip.",
    "upload_csv_to_minio": "Upload to MinIO. Needs: local_csv_path, tracking_ip. Produces: dataset_url.",
    "train_generic_model": "Train/Register model. Needs: dataset_url, target_column, experiment_name, tracking_ip, vm_name. Produces: best_model_uri, results_minio_url.",
    "publish_digital_object": "Publish to MRS. Needs: do_type, artifact_path, metadata.",
    "download_artifact_from_minio": "Download a file from MinIO to a LOCAL path. Needs: tracking_ip. Produces: local_path.",
}


def _build_tool_catalog(schemas: dict[str, dict], servers: list[str]) -> str:
    """
    Builds the tool catalog text shown to the LLM, restricted to the servers
    relevant to the phase currently being generated. Parameter names are
    read from the live MCP schema (list_tools) rather than hand-maintained,
    so a schema change on the server side is reflected automatically.
    """
    lines = []
    for server in servers:
        tools = MCP_SERVERS.get(server, set())
        if not tools:
            continue
        lines.append(f"\n## Server: {server}")
        for tool in sorted(tools):
            if tool in _LIFECYCLE_ONLY_TOOLS:
                continue
            desc = TOOL_DESCRIPTIONS.get(tool, "Action tool.")
            params_str = ""
            if schemas and tool in schemas:
                props = schemas[tool].get("properties", {}).keys()
                if props:
                    params_str = f" [Params: {', '.join(props)}]"
            lines.append(f"- {tool}: {desc}{params_str}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# PHASE-SPECIFIC GENERATION RULES
# Same content that used to live inside one giant SYSTEM_PROMPT_TEMPLATE,
# just split so each phase only ever sees its own rules. Task numbers below
# ("task 1", "task 2", ...) refer to positions WITHIN this phase — they get
# namespaced automatically after generation, see _rename_task_ids().
# ══════════════════════════════════════════════════════════════════════════

PHASE1_RULES = """\
1. get_slices_session: No dependencies. Produces fields named "session" and "mod_auth_openidc_session".
   IMPORTANT: the output field is named "session", NOT "session_cookie". "session_cookie" is only
   the *parameter name* that configure_post5g_experiment expects. When wiring that param you MUST
   write `"session_cookie": "${TaskN.session}"`, referencing this task's "session" field, not a
   nonexistent "session_cookie" field.
2. slices_create_experiment: No dependencies. Produces experiment_name, experiment_token. Extract
   project_name and experiment_name from the intent. The `duration` param MUST be a string in the
   SLICES CLI duration format, e.g. "1h", "30m", "6h" — NEVER a raw integer number of seconds. If the
   user doesn't specify a duration, default to "2h".
3. post5g_get_prefix: Uses `experiment_name: "${TaskN.experiment_token}"` (task 2's output).
   MUST depend on task 2. Produces nrf_lb_ip, multus_network, lb.
4. get_available_nodes: No dependencies. Lists nodes.
5. configure_post5g_experiment: Uses `session_cookie`, `mod_auth_openidc_session` (task 1's outputs),
   `experiment_name` (task 2's output), `nrf_lb_ip`, `multus_network` (task 3's outputs).
   MUST depend on tasks 1, 2, and 3. Extract the exact node name (e.g., "standard-2-1") from the
   user's intent for `pos_deployment_node`.
6. book_pos_calendar: Uses `mod_auth_openidc_session` (task 1's output). Extract start/end time from
   the intent. `node` MUST be the exact node name (e.g., "standard-2-1"). `start_time` and `end_time`
   MUST be bare time-of-day strings only, e.g. "10:00" / "11:00" — NEVER a full ISO 8601 datetime. The
   date is supplied separately via `start_date`/`end_date` (format "YYYY-MM-DD"; defaults to today if
   omitted — for "today" in the intent, you may omit start_date/end_date entirely). MUST depend on
   task 1. Do NOT include an `owner` param unless the user's intent explicitly names a specific
   person/username to book on behalf of. The tool defaults `owner` to the authenticated SLICES_USER
   automatically — inventing a placeholder value like "user@example.com" will likely cause the booking
   to be rejected.
7. post5g_get_experiment: Uses `experiment_name: "${TaskN.experiment_token}"` (task 2's output).
   MUST depend on tasks 2, 5, and 6 (task 2 because you reference its output; tasks 5 and 6 because
   you cannot get an experiment before it's configured and booked).
8. post5g_launch_experiment: Uses `experiment_name: "${TaskN.experiment_token}"` (task 2's output).
   MUST depend on tasks 2 and 7 (task 2 because you reference its output; task 7 because you must
   check the experiment before launching it).
9. trigger_5g_anomaly: Uses `lb_ip: "${TaskN.lb}"` (task 3's output). MUST depend on tasks 3 and 8
   (task 3 because you reference its output; task 8 because the experiment must be launched first).
   Declare "log_file_local" in its declared_outputs — a later phase will need it.

REMINDER: every task above that says "Uses <field>: <ref>" MUST list the referenced task's number in
its own `depends_on` — this holds even where a task also has other, ordering-only dependencies listed
alongside it. Missing just one of these is enough to fail validation.
"""

PHASE2_RULES = """\
1. bi_list_infra: No dependencies, no special params.
2. bi_create_mlops_vm: Uses `experiment_name` (the upstream task's output — see "TASKS AVAILABLE FROM
   EARLIER PHASES" above). MUST depend on that upstream task. Extract vm_name from the intent. If
   site_id/image/flavor/duration are not specified in the intent, OMIT them from params entirely (do
   not pass "") so the tool's defaults apply.
3. bi_wait_vm_ready: Uses `experiment_name` (upstream task's output — see "TASKS AVAILABLE FROM
   EARLIER PHASES" above) and `vm_name`. MUST depend on task 2 (bi_create_mlops_vm) AND on the
   upstream task that provides `experiment_name` — depending on task 2 alone is NOT enough even
   though task 2 itself also depends on that upstream task: dependency validation checks each task's
   OWN depends_on list directly, transitive dependencies never count.
4. bi_transfer_file_from_post5g: Uses `experiment_name` (upstream task), `local_file` (the
   "log_file_local" output of the upstream task that generated the anomaly dataset — see "TASKS
   AVAILABLE FROM EARLIER PHASES" above), and `vm_name`. MUST depend on task 3 (bi_wait_vm_ready, so
   the VM is reachable) and on the upstream task producing "log_file_local". Produces destination (the
   remote path on the VM), source, success, message.
5. bi_deploy_mlops_stack: Uses `experiment_name`, `vm_name`. MUST depend on task 3 (bi_wait_vm_ready).
6. bi_open_tunnels: Uses `experiment_name`, `vm_name`. MUST depend on task 5 (bi_deploy_mlops_stack —
   tunnels only make sense once the stack is running). Produces tracking_ip, vm_ip (the VM's real IP —
   use this as public_ip below, NOT tracking_ip/localhost).
7. upload_csv_to_minio: Uses `tracking_ip` (task 6's output), `local_csv_path` (same upstream
   "log_file_local" output used in task 4), and `public_ip: "${TaskN.vm_ip}"` (task 6's output, so
   dataset_url stays valid beyond your local tunnel). MUST depend on task 6 and on the upstream task
   producing "log_file_local". Produces dataset_url.
8. train_generic_model: Uses `dataset_url: "${TaskN.dataset_url}"` (task 7's output — NOT
   dataset_public_url: train_generic_model reconnects directly to that host to download the file, and
   dataset_public_url is only reachable via the tunnel/bastion, not from wherever this tool runs),
   `tracking_ip` (task 6's output), `target_column` (extract from the intent), `experiment_name`
   (upstream task), and `public_ip: "${TaskN.vm_ip}"` (task 6's output — safe here, since
   results_minio_url is only ever used as a display string, never re-fetched by another tool). MUST
   depend on tasks 6 and 7, and on the upstream experiment task. Produces best_model_uri,
   results_minio_url (S3/MinIO link to the uploaded training_results.json). Declare
   "results_minio_url" in declared_outputs — a later phase may need it.
9. download_artifact_from_minio: Only include this task if the overall intent implies the trained
   artifact should later be published to MRS. Uses `tracking_ip: "${TaskN.tracking_ip}"` (task 6's
   output — same tunnel as upload_csv_to_minio / train_generic_model). bucket/filename can be left at
   defaults ("results"/"training_results.json") unless the intent says otherwise. MUST depend on
   tasks 6 and 8 (task 6 because you reference its output; task 8 because the model must be trained
   and its results uploaded first). Declare "local_path" in declared_outputs.

REMINDER: every task above that says "Uses <field>: <ref>" MUST list the referenced task's number in
its own `depends_on` — this holds even where a task also has other, ordering-only dependencies listed
alongside it. Missing just one of these is enough to fail validation.
"""

PHASE3_RULES = """\
1. publish_digital_object: do_type="dataset". You MUST fully populate `metadata` with descriptive
   fields (e.g., "identifier", "name", "description"). Do NOT leave it empty "{}".
   You MUST also set `metadata.downloadUrl` to the S3/MinIO link of the published artifact — use the
   upstream "results_minio_url" output (see "TASKS AVAILABLE FROM EARLIER PHASES" above) unless the
   intent points to a different artifact, in which case use that artifact's own MinIO URL output
   instead. downloadUrl is NOT auto-filled — omitting it leaves the digital object with an empty
   download link. MUST depend on whichever upstream task produces the URL you reference.
   By default, ALWAYS set `artifact_path` to the upstream "local_path" output (produced by
   download_artifact_from_minio) so the file is actually attached inside MRS, not just referenced via
   downloadUrl. Without artifact_path, the portal's own download button will 404 (NoSuchKey) even
   though downloadUrl still works as an external reference. Only omit artifact_path if the intent
   explicitly asks for a metadata-only record.
   Do NOT set `artifact_path` directly to an MLflow run URI like "runs:/<id>/model" or to any MinIO/HTTP
   URL — those are NOT local filesystem paths and will fail with "Artifact path not found". MUST depend
   on the upstream task producing "local_path" since artifact_path is set by default.
"""


class PhaseDefinition(BaseModel):
    servers: list[str]
    provides: list[ResourceKind] = []
    requires: list[ResourceKind] = []
    summary: str   # one-line description, shown to the coarse planner only
    rules: str     # detailed generation rules, shown only during fine generation of this phase


# ══════════════════════════════════════════════════════════════════════════
# PHASE CATALOG — single source of truth for how servers group into
# workflow phases. Registering a new MCP server means adding (or extending)
# an entry here; the coarse planner, the tool-catalog filter, and the
# resource-based phase ordering all pick it up automatically. Nothing else
# in this file needs to change.
# ══════════════════════════════════════════════════════════════════════════

PHASE_CATALOG: dict[str, PhaseDefinition] = {
    "post5g_setup": PhaseDefinition(
        servers=["slices"],
        provides=[ResourceKind.POST5G_EXPERIMENT, ResourceKind.DATASET],
        requires=[],
        summary="Create, configure, book and launch a Post-5G experiment, then generate anomaly traffic.",
        rules=PHASE1_RULES,
    ),
    "bi_training": PhaseDefinition(
        servers=["bi", "mlflow"],
        provides=[ResourceKind.BI_VM, ResourceKind.MLFLOW_TRACKING],
        requires=[ResourceKind.DATASET],
        summary="Create a BI VM, deploy the MLOps stack, and train a model on the generated dataset.",
        rules=PHASE2_RULES,
    ),
    "mrs_publish": PhaseDefinition(
        servers=["mrs"],
        provides=[],
        requires=[ResourceKind.MLFLOW_TRACKING],
        summary="Download the trained artifact and publish it as a Digital Object on MRS.",
        rules=PHASE3_RULES,
    ),
}


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════
# LEVEL 1 — COARSE PLANNING
# Decide WHICH phases are needed. Output space is tiny (a handful of phase
# names out of PHASE_CATALOG), which keeps hallucination risk low even as
# the number of registered phases grows over time.
# ══════════════════════════════════════════════════════════════════════════

COARSE_PLANNER_PROMPT = """\
You are the planning layer of an MLOps orchestration system.
Your ONLY task is to decide WHICH workflow phases are needed to satisfy the
user's intent. You do NOT decide task order, parameters, or dependencies —
that is handled separately and deterministically.

Current context time: {current_time}

== AVAILABLE PHASES ==
{phase_catalog}

== RULES ==
- Respond with a JSON object: {{"phases": ["phase_name", ...]}}
- Only use phase names that exist in the list above, spelled exactly as shown.
- Include a phase only if the intent actually requires the outcome it produces.
- Order does not matter here — execution order is derived automatically from
  each phase's resource dependencies.
- Output ONLY the JSON object, nothing else.

Intent: {intent}
"""


class _PhasePlan(BaseModel):
    phases: list[str]


def _order_phases(selected: list[str]) -> list[str]:
    """
    Deterministic topological ordering of the selected phases, derived from
    each phase's `provides`/`requires` resource kinds — the same idea as
    ExecutionSpecification.topological_order(), one level up. The LLM only
    chooses WHICH phases are needed; ordering between them is never left to
    generation.
    """
    remaining = set(selected)
    available: set[ResourceKind] = set()
    ordered: list[str] = []

    while remaining:
        ready = sorted(
            name for name in remaining
            if set(PHASE_CATALOG[name].requires) <= available
        )
        if not ready:
            missing = {
                name: [r.value for r in PHASE_CATALOG[name].requires if r not in available]
                for name in remaining
            }
            raise IntentLayerError(
                f"Selected phases have unsatisfied resource dependencies: {missing}. "
                f"Include the phase(s) that provide the missing resources."
            )
        ordered.extend(ready)
        for name in ready:
            available.update(PHASE_CATALOG[name].provides)
            remaining.discard(name)

    return ordered


async def _plan_phases(
    intent: str,
    llm_client: AsyncOpenAI,
    max_attempts: int = MAX_PLANNING_ATTEMPTS,
) -> list[str]:
    catalog_str = "\n".join(f"- {name}: {p.summary}" for name, p in PHASE_CATALOG.items())
    prompt = COARSE_PLANNER_PROMPT.format(
        phase_catalog=catalog_str,
        intent=intent,
        current_time=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = await llm_client.chat.completions.create(
                model=os.environ["LLM_MODEL"], messages=messages,
            )
        except openai.RateLimitError:
            import asyncio
            await asyncio.sleep(6)
            continue
        except openai.OpenAIError as e:
            last_error = f"LLM error during planning: {e}"
            messages.append({"role": "user", "content": last_error})
            continue

        raw_text = response.choices[0].message.content or ""
        cleaned = _strip_markdown_fences(raw_text)
        try:
            plan = _PhasePlan(**json.loads(cleaned))
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = f'Invalid plan ({e}). Respond ONLY with {{"phases": [...]}}.'
            print(f"[intent_layer] Planning attempt {attempt}/{max_attempts}: {last_error}")
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": last_error})
            continue

        unknown = [p for p in plan.phases if p not in PHASE_CATALOG]
        if unknown or not plan.phases:
            last_error = (
                f"Invalid phase selection {plan.phases}. Unknown names: {unknown}. "
                f"Valid phase names are exactly: {list(PHASE_CATALOG.keys())}. "
                f"At least one phase is required."
            )
            print(f"[intent_layer] Planning attempt {attempt}/{max_attempts}: {last_error}")
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": last_error})
            continue

        try:
            ordered = _order_phases(plan.phases)
        except IntentLayerError as e:
            last_error = str(e)
            print(f"[intent_layer] Planning attempt {attempt}/{max_attempts}: {last_error}")
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": last_error})
            continue

        print(f"[intent_layer] Coarse plan (attempt {attempt}/{max_attempts}): {ordered}")
        return ordered

    raise IntentLayerError(
        f"Unable to obtain a valid phase plan after {max_attempts} attempts. Last error: {last_error}"
    )


# ══════════════════════════════════════════════════════════════════════════
# LEVEL 2 — FINE GENERATION (one phase at a time)
# ══════════════════════════════════════════════════════════════════════════

PHASE_SYSTEM_PROMPT_TEMPLATE = """\
You are the intent layer of an MLOps system for the SLICES-RI research infrastructure.
You are generating ONLY the tasks for ONE phase of a larger workflow: "{phase_name}".
Other phases are generated separately and merged afterwards — do NOT invent tasks that
belong to a different phase, and do NOT try to cover parts of the intent unrelated to
this phase.

Respond ONLY with a valid JSON object of the form {{"tasks": [...]}}.

Current context time: {current_time}

== AVAILABLE TOOLS FOR THIS PHASE ==
{tool_catalog}

== TASKS AVAILABLE FROM EARLIER PHASES ==
{upstream_context}

== RULES SPECIFIC TO THIS PHASE ==
{phase_rules}

== GENERAL RULES FOR EVERY TASK ==
- Each task object MUST have exactly these keys: "id" (string), "tool" (string),
  "depends_on" (array of strings), "params" (object), and optionally "declared_outputs"
  (array of strings).
- Do NOT use "task_id" (use "id") and do NOT use "name" (use "tool").
- Task ids only need to be unique WITHIN this phase — they are namespaced automatically
  after generation, so plain names like "Task1", "Task2" are fine.
- The "params" keys MUST exactly match the names listed in the tool catalog.
- For OPTIONAL parameters not mentioned in the user's intent, OMIT the key entirely from
  "params" — do NOT include it as an empty string "" or as null.
- Reference syntax: "${{id.field_name}}". You may reference either a task id you defined
  in this phase, or one of the exact upstream task ids listed above.
- Dependency rule: if Task Y uses "${{TaskX.field}}", Task X MUST be in Task Y's
  `depends_on` list — this applies to upstream ids too: if you reference an upstream id,
  add it to `depends_on`.
- If a task's output might be needed by a later phase, list those exact field names in
  that task's `declared_outputs` array — this is what makes them referenceable downstream.
- SELF-CHECK before answering: for every "${{id.field}}" you wrote anywhere in `params`, go back and
  confirm that exact `id` is present in that same task's `depends_on` list. Do this even when a
  phase-specific rule above only mentioned other, ordering-only dependencies for that task — reference
  dependencies are ALWAYS required in addition to those, never instead of them.
- Output ONLY the JSON object.
"""


class _PhaseTasks(BaseModel):
    tasks: list[Task]


def _rename_task_ids(tasks: list[Task], prefix: str) -> list[Task]:
    """
    Namespaces every task id generated for one phase with a phase prefix, so
    ids from independently generated phases never collide once merged. Only
    LOCAL ids (ids the phase itself generated) get rewritten — references to
    upstream tasks from earlier phases are left untouched, because the model
    was given their final, already-namespaced id up front (see
    `_upstream_context`).
    """
    id_map = {t.id: f"{prefix}__{t.id}" for t in tasks}
    renamed: list[Task] = []
    for t in tasks:
        new_depends_on = [id_map.get(d, d) for d in t.depends_on]
        new_params: dict[str, Any] = {}
        for k, v in t.params.items():
            if isinstance(v, str):
                m = _REF_PATTERN.match(v)
                if m and m.group(1) in id_map:
                    v = f"${{{id_map[m.group(1)]}.{m.group(2)}}}"
            new_params[k] = v
        renamed.append(t.model_copy(update={
            "id": id_map[t.id],
            "depends_on": new_depends_on,
            "params": new_params,
        }))
    return renamed


def _upstream_context(merged_so_far: list[Task]) -> str:
    """
    Lists every task produced by earlier phases, with its exact (already
    namespaced) id and declared outputs, so the current phase can reference
    any of them correctly instead of guessing an id it never saw.
    """
    if not merged_so_far:
        return "No upstream tasks — this is the first generated phase."
    lines = ["Reference these by their exact id shown here:"]
    for t in merged_so_far:
        outputs = t.declared_outputs or ["(no declared outputs)"]
        lines.append(f'- "{t.id}" (tool: {t.tool}) -> outputs: {outputs}')
    return "\n".join(lines)


async def _generate_phase_spec(
    phase_name: str,
    intent: str,
    schemas: dict[str, dict],
    merged_so_far: list[Task],
    llm_client: AsyncOpenAI,
    max_attempts: int = MAX_GENERATION_ATTEMPTS,
    extra_context: str | None = None,
) -> list[Task]:
    """
    Fine generation step: produces the concrete task list for ONE phase
    only. The system prompt only contains the rules and tools relevant to
    this phase, which is what actually reduces hallucination, compared to
    asking for the entire multi-phase DAG in a single generation.
    """
    phase = PHASE_CATALOG[phase_name]
    system_prompt = PHASE_SYSTEM_PROMPT_TEMPLATE.format(
        phase_name=phase_name,
        tool_catalog=_build_tool_catalog(schemas, phase.servers),
        upstream_context=_upstream_context(merged_so_far),
        phase_rules=phase.rules,
        current_time=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    if extra_context:
        system_prompt += f"\n== CORRECTION NEEDED (previous attempt was merged and failed) ==\n{extra_context}\n"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Full intent (for context): {intent}"},
    ]
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = await llm_client.chat.completions.create(
                model=os.environ["LLM_MODEL"], messages=messages,
            )
        except openai.RateLimitError:
            import asyncio
            await asyncio.sleep(6)
            continue
        except openai.OpenAIError as e:
            last_error = f"LLM error: {e}"
            messages.append({"role": "user", "content": last_error})
            continue

        raw_text = response.choices[0].message.content or ""
        cleaned = _strip_markdown_fences(raw_text)
        try:
            phase_tasks = _PhaseTasks(**json.loads(cleaned))
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = f'Invalid phase output ({e}). Respond ONLY with {{"tasks": [...]}}.'
            print(f"[intent_layer] Phase '{phase_name}' attempt {attempt}/{max_attempts}: {last_error}")
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": last_error})
            continue

        print(f"[intent_layer] Phase '{phase_name}': valid task list obtained on attempt {attempt}/{max_attempts}.")
        return _rename_task_ids(phase_tasks.tasks, prefix=phase_name)

    raise IntentLayerError(
        f"Unable to generate a valid task list for phase '{phase_name}' after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


_OFFENDING_TASK_PATTERN = re.compile(r"[Tt]ask '([a-zA-Z0-9_]+)'")


def _guess_offending_phase(error_text: str, phase_names: list[str]) -> str:
    """
    Identifies which phase produced the invalid task. Every validator in
    execution_spec.py names the OFFENDING task first in its error message
    (e.g. "Task 'X' refers to output of [...]", "Task 'X' depends on
    non-existent IDs [...]"), even though other (correct) task ids from
    other phases are frequently mentioned later in the same message (e.g.
    the upstream id it failed to depend on). So we take the FIRST "Task
    '<id>'" match only — never search the whole message for a phase-name
    substring, since that also matches ids mentioned later for unrelated
    reasons and misattributes the failure to the wrong (innocent) phase.
    """
    match = _OFFENDING_TASK_PATTERN.search(error_text)
    if match:
        task_id = match.group(1)
        prefix = task_id.split("__", 1)[0]
        if prefix in phase_names:
            return prefix
    return phase_names[-1]


# ══════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════

async def generate_execution_spec(
    intent: str,
    workflow_id: str,
) -> ExecutionSpecification:
    """
    Two-level generation pipeline:
      1. Coarse planning: decide WHICH phases (PHASE_CATALOG entries) are
         needed, then order them deterministically from resource
         provides/requires.
      2. Fine generation: generate the concrete task list for each phase
         separately, with a system prompt scoped to that phase only.
      3. Merge all phases into a single ExecutionSpecification and run the
         full set of structural/semantic validators on the merged graph.
         If merging fails validation, only the offending phase (and any
         phase generated after it) is regenerated, with the validation
         error fed back as extra context.

    This keeps each individual LLM generation small and focused, and scales
    to new MCP servers by extending PHASE_CATALOG instead of growing a
    single monolithic prompt.
    """
    llm_client = AsyncOpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        http_client=httpx.AsyncClient(verify=False),
    )

    schemas: dict[str, dict] = {}
    mrs_session: ClientSession | None = None
    _stack = AsyncExitStack()
    try:
        await _stack.__aenter__()
        for server_name, script in SERVERS.items():
            try:
                params = StdioServerParameters(
                    command=sys.executable, args=[script], env=os.environ.copy(),
                )
                read, write = await _stack.enter_async_context(stdio_client(params))
                session = await _stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                for tool in (await session.list_tools()).tools:
                    schemas[tool.name] = tool.inputSchema or {}
                if server_name == "mrs":
                    mrs_session = session
            except Exception as e:
                print(f"[intent_layer] Connect to '{server_name}' failed: {e}")

        if mrs_session:
            try:
                await mrs_session.call_tool("record_lifecycle_event", {
                    "event_type": "intent_received", "workflow_id": workflow_id,
                    "payload": {"intent": intent},
                })
            except Exception as e:
                print(f"[intent_layer] intent_received not recorded: {e}")

        try:
            phase_names = await _plan_phases(intent, llm_client)

            phase_tasks: dict[str, list[Task]] = {}
            correction: dict[str, str] = {}

            for merge_attempt in range(1, MAX_MERGE_RETRIES + 2):
                merged: list[Task] = []
                for phase_name in phase_names:
                    if phase_name not in phase_tasks:
                        phase_tasks[phase_name] = await _generate_phase_spec(
                            phase_name, intent, schemas, merged, llm_client,
                            extra_context=correction.get(phase_name),
                        )
                    merged.extend(phase_tasks[phase_name])

                try:
                    spec = ExecutionSpecification(
                        workflow_id=workflow_id, intent=intent, tasks=merged,
                    )
                    print(
                        f"[intent_layer] Merged spec valid on merge attempt {merge_attempt}: "
                        f"{len(merged)} tasks across phases {phase_names}."
                    )
                    return spec
                except ValidationError as e:
                    offending = _guess_offending_phase(str(e), phase_names)
                    print(
                        f"[intent_layer] Merge attempt {merge_attempt} failed validation "
                        f"(regenerating phase '{offending}'): {e}"
                    )
                    if merge_attempt > MAX_MERGE_RETRIES:
                        raise IntentLayerError(
                            f"Unable to merge a valid spec after {MAX_MERGE_RETRIES + 1} attempts. "
                            f"Last error: {e}"
                        ) from e
                    # Regenerate the offending phase (with the error as context) and every
                    # phase generated after it, since their upstream task manifest is stale.
                    correction[offending] = str(e)
                    for name in phase_names[phase_names.index(offending):]:
                        phase_tasks.pop(name, None)
                        if name != offending:
                            correction.pop(name, None)

            raise IntentLayerError("Unreachable: merge retry loop exhausted without returning.")

        except IntentLayerError as e:
            if mrs_session:
                try:
                    await mrs_session.call_tool("record_lifecycle_event", {
                        "event_type": "intent_failed", "workflow_id": workflow_id,
                        "payload": {"intent": intent, "error": str(e)},
                    })
                except Exception:
                    pass
            raise
    finally:
        await _stack.__aexit__(None, None, None)


async def main():
    import sys as _sys

    intent = input("Enter the intent: ") if _sys.stdin.isatty() else _sys.stdin.read()
    workflow_id = f"wf-{abs(hash(intent)) % 100000}"

    spec = await generate_execution_spec(intent, workflow_id)

    print("\n-- Spec generated --")
    print(spec.model_dump_json(indent=2))

    print("\n-- Execution waves --")
    for i, wave in enumerate(spec.topological_order()):
        print(f"  wave {i}: {wave}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
