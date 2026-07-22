from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any
from datetime import datetime, UTC

import httpx
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from execution_spec import ExecutionSpecification, MCP_SERVERS, ALL_TOOLS

load_dotenv()

MAX_GENERATION_ATTEMPTS = 5

SERVERS = {
    "slices": "mcp_server_slices.py",
    "bi":     "mcp_server_bi.py",
    "mlflow": "mcp_server_mlflow.py",
    "mrs":    "mcp_server_mrs.py",
}

_LIFECYCLE_ONLY_TOOLS = {"record_lifecycle_event", "get_lifecycle_events"}


class IntentLayerError(Exception):
    pass


async def _connect_for_intent_layer(stack: AsyncExitStack) -> tuple[dict[str, dict], ClientSession]:
    """
    It connects to all real MCP servers (the same mechanism as 
    WorkflowExecutor._connect_servers in workflow_executor.py) for: 
    1) reading the real schema (inputSchema) of each tool via list_tools(), 
    so the system prompt reflects the actual parameter names instead of 
    relying only on manually written descriptions that could (and in 
    practice were already) be misaligned — e.g., post5g_get_prefix wants 
    "experiment_name" not "experiment_token", configure_post5g_experiment 
    wants "session_cookie" not "session" and it doe doesn't have a parameter at all 
    "lb"; 
    2) return the ClientSession to 'mrs', kept open throughout 
    generation duration, to record intent_received/intent_failed. 
    Returns (schemas, mrs_session).
    """
    schemas: dict[str, dict] = {}
    mrs_session: ClientSession | None = None

    for server_name, script in SERVERS.items():
        params = StdioServerParameters(
            command=sys.executable,
            args=[script],
            env=os.environ.copy(),
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools_response = await session.list_tools()
        for tool in tools_response.tools:
            schemas[tool.name] = tool.inputSchema or {}

        if server_name == "mrs":
            mrs_session = session

    assert mrs_session is not None, "mcp_server_mrs.py does not have a valid session."
    return schemas, mrs_session


async def _record_intent_event(
    mrs_session: ClientSession,
    event_type: str,
    workflow_id: str,
    intent: str,
    extra_payload: dict | None = None,
) -> None:
    """
    Records a lifecycle event at the intent level (intent received at the start, intent_failed if generation runs out of attempts).
    """
    try:
        result = await mrs_session.call_tool("record_lifecycle_event", {
            "event_type":  event_type,
            "workflow_id": workflow_id,
            "payload":     {"intent": intent, **(extra_payload or {})},
        })
        text = result.content[0].text if result.content else "{}"
        data = json.loads(text) if text else {}
        if not data.get("success"):
            print(f"[intent_layer] Recording event '{event_type}' failed: {data.get('error')}")
    except Exception as e:
        print(f"[intent_layer] Error recording event '{event_type}': {e}")


def _build_tool_catalog(schemas: dict[str, dict] | None = None) -> str:
    descriptions = {
        "get_slices_session": "Login (first step). Produces: session, mod_auth_openidc_session.",
        "slices_create_experiment": "Create experiment. Produces: experiment_name, experiment_token.",
        "post5g_get_prefix": "Get network info. Needs: experiment_name.",
        "get_available_nodes": "List nodes.",
        "configure_post5g_experiment": "Config. Needs: session_cookie, mod_auth_openidc_session, experiment_name, pos_deployment_node, nrf_lb_ip, multus_network.",
        "book_pos_calendar": "Reserve node. Needs: mod_auth_openidc_session, node, start_time, end_time.",
        "post5g_get_experiment": "Check exp status. Needs: experiment_name.",
        "post5g_launch_experiment": "Launch exp. Needs: experiment_name.",
        "trigger_5g_anomaly": "Generate CSV anomaly. Needs: lb_ip.",
        "bi_list_infra": "List BI infra.",
        "bi_create_mlops_vm": "Create VM. Needs: experiment_name, vm_name.",
        "bi_wait_vm_ready": "Wait for VM SSH. Needs: experiment_name, vm_name.",
        "bi_transfer_file_from_post5g": "Transfer CSV to VM. Needs: experiment_name, local_file, vm_name.",
        "bi_deploy_mlops_stack": "Install MLOps on VM. Needs: experiment_name, vm_name.",
        "bi_open_tunnels": "Open tunnels. Needs: experiment_name, vm_name. Produces: tracking_ip.",
        "upload_csv_to_minio": "Upload to MinIO. Needs: local_csv_path, tracking_ip.",
        "train_generic_model": "Train/Register model. Needs: dataset_url, target_column, experiment_name, tracking_ip, vm_name.",
        "publish_digital_object": "Publish to MRS. Needs: do_type, artifact_path, metadata.",
        "download_artifact_from_minio": "Download a file from MinIO to a LOCAL path. Needs: tracking_ip (from bi_open_tunnels). "
                                        "Produces: local_path. Call before publish_digital_object if you want to attach an artifact.",
    }

    lines = []
    for server, tools in MCP_SERVERS.items():
        lines.append(f"\n## Server: {server}")
        for tool in sorted(tools):
            if tool in _LIFECYCLE_ONLY_TOOLS: continue
            
            desc = descriptions.get(tool, "Action tool.")

            params_str = ""
            if schemas and tool in schemas:
                props = schemas[tool].get("properties", {}).keys()
                if props:
                    params_str = f" [Params: {', '.join(props)}]"
            
            lines.append(f"- {tool}: {desc}{params_str}")
            
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """\
You are the intent layer of an MLOps system for the SLICES-RI research infrastructure.
 
Your ONLY task is to translate the user's scientific intent into a structured ExecutionSpecification, compliant with the provided JSON schema.
You do NOT execute anything. Respond ONLY with a valid JSON object.
 
Current context time: {current_time}
 
══ AVAILABLE TOOLS ══
{tool_catalog}
 
══ MANDATORY WORKFLOW DOMAIN RULES ══
Read the user intent carefully. You MUST generate tasks for EVERY step requested. Do not stop halfway. If the user asks for a full pipeline (5G + BI VM + Training + MRS), your JSON MUST contain ALL 3 phases below, generating up to 17 tasks.
 
PHASE 1: FULL POST-5G EXPERIMENT SETUP
  1. get_slices_session: Produces fields named "session" and "mod_auth_openidc_session".
     IMPORTANT: the output field is named "session", NOT "session_cookie". "session_cookie" is only
     the *parameter name* that configure_post5g_experiment expects. When wiring that param you MUST
     write `"session_cookie": "${{Task1_ID.session}}"` — referencing Task1's "session" field, not a
     nonexistent "session_cookie" field.
  2. slices_create_experiment: Produces experiment_name, experiment_token. (Extract project_name and experiment_name from intent). 
     The `duration` param MUST be a string in the SLICES CLI duration format, e.g. "1h", "30m", "6h" — NEVER a raw integer number of seconds. If the user doesn't specify a duration, default to "2h".
  3. post5g_get_prefix: Uses `experiment_name: "${{Task2_ID.experiment_token}}"`. Produces nrf_lb_ip, multus_network, lb.
  4. get_available_nodes: Lists nodes.
  5. configure_post5g_experiment: Uses outputs from tasks 1, 2, 3 — specifically `session_cookie: ${{Task1_ID.session}}`, `mod_auth_openidc_session: ${{Task1_ID.mod_auth_openidc_session}}`,
     experiment_name, nrf_lb_ip, multus_network. Extract the exact node name (e.g., "standard-2-1") from the user's intent for `pos_deployment_node`.
  6. book_pos_calendar: Extract start/end time from intent. `node` MUST be the exact node name (e.g., "standard-2-1"). `start_time` and `end_time` MUST be bare time-of-day strings only, e.g. "10:00" / "11:00" — NEVER a full
     ISO 8601 datetime. The date is supplied separately via `start_date`/`end_date` (format "YYYY-MM-DD"; defaults to today if omitted — for "today" in the intent, you may omit start_date/end_date entirely). MUST depend on Task 1.
     Do NOT include an `owner` param unless the user's intent explicitly names a specific person/username to book on behalf of. The tool defaults `owner` to the authenticated SLICES_USER automatically — inventing a placeholder value like "user@example.com" will likely cause the booking to be rejected by the server.
  7. post5g_get_experiment: MUST use `experiment_name: "${{Task2_ID.experiment_token}}"`. MUST depend on Task 5 and Task 6 (you cannot get an experiment before configuring and booking).
  8. post5g_launch_experiment: MUST use `experiment_name: "${{Task2_ID.experiment_token}}"`. MUST depend on Task 7.
  9. trigger_5g_anomaly: Uses `lb_ip: "${{Task3_ID.lb}}"`. MUST depend on Task 8.
 
PHASE 2: MLOPS & BI VM DEPLOYMENT
  10. bi_list_infra
  11. bi_create_mlops_vm: Extract vm_name from intent. If site_id/image/flavor/duration are not specified in the
      intent, OMIT them from params entirely (do not pass "") so the tool's defaults apply.
  12. bi_wait_vm_ready
  13. bi_transfer_file_from_post5g: Uses `local_file: "${{Task9_ID.log_file_local}}"`. Produces destination (the remote path on the VM), source, success, message.
  14. bi_deploy_mlops_stack
  15. bi_open_tunnels: Produces tracking_ip, vm_ip (the VM's real IP — use this as public_ip below, NOT tracking_ip/localhost).
  16. upload_csv_to_minio: Uses `tracking_ip`, `local_csv_path: "${{Task9_ID.log_file_local}}"`, and
      `public_ip: "${{Task15_ID.vm_ip}}"` (so dataset_url stays valid beyond your local tunnel). Produces dataset_url.
   17. train_generic_model: Uses `dataset_url: "${{Task16_ID.dataset_url}}"` (NOT dataset_public_url —
      train_generic_model reconnects directly to that host to download the file, and dataset_public_url
      is only reachable via the tunnel/bastion, not from wherever this tool runs), `tracking_ip`,
      `target_column` (extract from intent), `experiment_name`, and `public_ip: "${{Task15_ID.vm_ip}}"`
      (safe here — results_minio_url is only ever used as a display string, never re-fetched by another tool).
      Produces best_model_uri, results_minio_url (S3/MinIO link to the uploaded training_results.json).
  17b. download_artifact_from_minio: Only if the user wants the trained model artifact published to MRS in PHASE 3.
       Uses `tracking_ip: "${{Task15_ID.tracking_ip}}"` (same tunnel as upload_csv_to_minio/train_generic_model).
       bucket/filename can be left at defaults ("results"/"training_results.json") unless the intent says otherwise.
       MUST depend on the train_generic_model task. Produces local_path.
 
PHASE 3: PUBLISHING TO MRS
  18. publish_digital_object: do_type="dataset". You MUST fully populate `metadata` with descriptive fields
      (e.g., "identifier", "name", "description"). Do NOT leave it empty "{{}}".
      You MUST also set `metadata.downloadUrl` to the S3/MinIO link of the published artifact — use
      `"${{Task17_ID.results_minio_url}}"` (the output of train_generic_model) unless the intent points to a
      different artifact, in which case use that task's own MinIO URL output instead. downloadUrl is NOT
      auto-filled — omitting it leaves the digital object with an empty download link. MUST depend on
      whichever task produces the URL you reference.
      By default, ALWAYS set `artifact_path: "${{Task17b_ID.local_path}}"` (the output of
      download_artifact_from_minio — a real local path) so the file is actually attached inside MRS, not
      just referenced via downloadUrl. Without artifact_path, the portal's own download button will 404
      (NoSuchKey) even though downloadUrl still works as an external reference. Only omit artifact_path if
      the intent explicitly asks for a metadata-only record.
      Do NOT set `artifact_path` directly to an MLflow run URI like "runs:/<id>/model" or to any MinIO/HTTP URL —
      those are NOT local filesystem paths and will fail with "Artifact path not found". MUST depend on the
      download_artifact_from_minio task since artifact_path is set by default.
 
══ RULES FOR THE SPEC ══
- Each task object MUST have exactly these keys: "id" (string), "tool" (string), "depends_on" (array of strings), and "params" (object).
- Do NOT use "task_id" (use "id") and do NOT use "name" (use "tool").
- The "params" keys MUST exactly match the names listed in the tool catalog.
- For OPTIONAL parameters not mentioned in the user's intent, OMIT the key entirely from "params" — do NOT
  include it as an empty string "" or as null. An empty string overrides the tool's own built-in default and
  can cause CLI/validation errors downstream (e.g. "--duration '' is not a date, nor a duration").
  Only include a parameter if the intent specifies a value for it, or if the tool catalog/workflow rules above
  explicitly require it.
- Reference syntax: "${{id.field_name}}" (e.g., if task 2 has id "Task2", use "${{Task2.experiment_token}}").
- Dependency rule: If Task Y uses "${{TaskX.field}}", Task X MUST be in Task Y's `depends_on` list.
- Output ONLY JSON.
"""


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


async def generate_execution_spec(
    intent: str,
    workflow_id: str,
    max_attempts: int = MAX_GENERATION_ATTEMPTS,
) -> ExecutionSpecification:
    llm_client = AsyncOpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        http_client=httpx.AsyncClient(verify=False),
    )

    schemas: dict[str, dict] = {}
    mrs_session = None
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

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tool_catalog=_build_tool_catalog(schemas),
            current_time=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Intent: {intent}\n\n"
                f"Use exactly workflow_id=\"{workflow_id}\" and intent=\"{intent}\" in the spec."
            )},
        ]
        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await llm_client.chat.completions.create(
                    model=os.environ["LLM_MODEL"], messages=messages,
                )
            except openai.RateLimitError:
                import asyncio; await asyncio.sleep(6); continue
            except openai.OpenAIError as e:
                last_error = f"LLM Error: {e}"
                messages.append({"role": "user", "content": last_error}); continue

            raw_text = response.choices[0].message.content or ""
            cleaned = _strip_markdown_fences(raw_text)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON ({e}). Respond ONLY with the JSON object."
                print(f"[intent_layer] Attempt {attempt}/{max_attempts}: Invalid JSON — {e}")
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": last_error}); continue

            try:
                spec = ExecutionSpecification(**parsed)
                print(f"[intent_layer] Valid spec obtained on attempt {attempt}/{max_attempts}.")
                return spec
            except Exception as e:
                last_error = (
                    f"The spec failed validation: {e}. "
                    f"Fix ONLY the indicated fields and respond with the entire corrected JSON."
                )
                print(f"[intent_layer] Attempt {attempt}/{max_attempts}: validation failed — {e}")
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": last_error}); continue

        if mrs_session:
            try:
                await mrs_session.call_tool("record_lifecycle_event", {
                    "event_type": "intent_failed", "workflow_id": workflow_id,
                    "payload": {"intent": intent, "attempts": max_attempts, "last_error": last_error},
                })
            except Exception:
                pass

        raise IntentLayerError(
            f"Unable to obtain a valid ExecutionSpecification after {max_attempts} attempts. "
            f"Last error: {last_error}"
        )
    finally:
        await _stack.__aexit__(None, None, None)


async def main():
    import asyncio as _asyncio
    import sys as _sys

    intent = input("Enter the intent: ") if _sys.stdin.isatty() else _sys.stdin.read()
    workflow_id = f"wf-{abs(hash(intent)) % 100000}"

    spec = await generate_execution_spec(intent, workflow_id)

    print("\n── Spec generated ──")
    print(spec.model_dump_json(indent=2))

    print("\n── Execution waves ──")
    for i, wave in enumerate(spec.topological_order()):
        print(f"  wave {i}: {wave}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())