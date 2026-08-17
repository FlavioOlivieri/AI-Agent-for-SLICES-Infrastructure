import asyncio
import sys
import json
import os
import re
import openai
from contextlib import AsyncExitStack
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
import httpx

load_dotenv()

llm_client = AsyncOpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
    http_client=httpx.AsyncClient(verify=False),
)

servers = {
    "SLICES_Auth": "mcp_server_slices.py",
    "SLICES_BI":   "mcp_server_bi.py",
    "MLflow":      "mcp_server_mlflow.py",
    "MRS":         "mcp_server_mrs.py",
}

MAX_CHARS   = 1500
MAX_RETRIES = 3

# Tools that touch SLICES session auth — always run sequentially
SLICES_AUTH_TOOLS = {
    "get_slices_session",
    "slices_create_experiment",
    "slices_list_experiments",
    "post5g_get_prefix",
    "configure_post5g_experiment",
    "book_pos_calendar",
    "list_pos_calendar",
    "delete_pos_calendar",
    "post5g_get_experiment",
    "post5g_launch_experiment",
    "trigger_5g_anomaly",
}

# Params whose descriptions must never be truncated
PRESERVE_FULL_DESC = {
    "dnns", "slices",
    "start_date", "end_date",
    "pos_deployment_node",
}

# ── Context state: values extracted from tool results ─────────────────────────
# The agent injects these as a "state block" into every system turn so the model
# never has to guess or invent infrastructure values.

class AgentState:
    def __init__(self):
        self.session_cookie:            str | None = None
        self.mod_auth_openidc_session:  str | None = None
        self.experiment_token:          str | None = None
        self.experiment_name:           str | None = None
        self.nrf_lb_ip:                 str | None = None
        self.multus_network:            str | None = None
        self.lb_ip:                     str | None = None
        self.vm_ip:                     str | None = None
        self.dataset_url:               str | None = None
        self.tracking_ip:               str | None = None

    def update_from_tool(self, tool_name: str, result_text: str):
        """Parse key infrastructure values out of every tool result."""
        try:
            data = json.loads(result_text)
        except Exception:
            return

        if not isinstance(data, dict):
            return

        if tool_name == "get_slices_session":
            self.session_cookie           = data.get("session") or self.session_cookie
            self.mod_auth_openidc_session = data.get("mod_auth_openidc_session") or self.mod_auth_openidc_session

        elif tool_name in ("slices_create_experiment", "slices_list_experiments"):
            self.experiment_name = data.get("experiment_name") or self.experiment_name
            token = data.get("experiment_token")
            if not token:
                # slices_list_experiments returns a list
                tokens = data.get("experiment_tokens", [])
                token  = tokens[0] if tokens else None
            self.experiment_token = token or self.experiment_token

        elif tool_name == "post5g_get_prefix":
            self.nrf_lb_ip      = data.get("nrf_lb_ip")   or self.nrf_lb_ip
            self.multus_network = data.get("multus_network") or self.multus_network
            self.lb_ip          = data.get("lb")           or self.lb_ip

        elif tool_name == "bi_get_vm_ip":
            self.vm_ip = data.get("private_ip") or self.vm_ip

        elif tool_name == "bi_open_tunnels":
            self.tracking_ip = data.get("tracking_ip") or self.tracking_ip

        elif tool_name == "upload_csv_to_minio":
            self.dataset_url = data.get("dataset_url") or self.dataset_url

    def as_context_block(self) -> str:
        """Return a compact JSON block injected before every LLM call."""
        populated = {k: v for k, v in {
            "session_cookie":           self.session_cookie,
            "mod_auth_openidc_session": self.mod_auth_openidc_session,
            "experiment_token":         self.experiment_token,
            "experiment_name":          self.experiment_name,
            "nrf_lb_ip":               self.nrf_lb_ip,
            "multus_network":          self.multus_network,
            "lb_ip":                   self.lb_ip,
            "vm_ip":                   self.vm_ip,
            "dataset_url":             self.dataset_url,
            "tracking_ip":             self.tracking_ip,
        }.items() if v is not None}

        if not populated:
            return ""
        return (
            "\n\n[COLLECTED VALUES — use these exactly, do not invent alternatives]\n"
            + json.dumps(populated, indent=2)
        )

    def has_prefix_data(self) -> bool:
        return bool(self.nrf_lb_ip and self.multus_network)


# ── Helpers ───────────────────────────────────────────────────────────────────

def truncate_result(text: str) -> str:
    # Abbassiamo il limite massimo per singolo tool a 600 caratteri per modelli con contesti piccoli (8k)
    GLOBAL_MAX = 600 
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            # Se è il tool di addestramento, eliminiamo i report enormi e i modelli secondari
            if "all_models" in parsed:
                parsed["all_models"] = parsed["all_models"][:1]  # Tieni solo il primo
            if "best_model" in parsed and isinstance(parsed["best_model"], dict):
                if "classification_report" in parsed["best_model"]:
                    # Rimuoviamo la matrice testuale che consuma centinaia di token
                    parsed["best_model"]["classification_report"] = "Omitted to save space"
            
            # Sfoltimento generico per altri tool
            if "events" in parsed:
                parsed["events"] = parsed["events"][:3]
            if "output" in parsed and len(str(parsed["output"])) > 300:
                parsed["output"] = str(parsed["output"])[-300:]
            if "create_help" in parsed or "bi_help" in parsed:
                for key in ("create_help", "bi_help"):
                    if key in parsed:
                        parsed[key] = parsed[key][:200]
                        
            parsed["_truncated"] = True
            text = json.dumps(parsed)
    except Exception:
        pass
        
    if len(text) <= GLOBAL_MAX:
        return text
    return text[:GLOBAL_MAX] + "\n... [truncated]"


def _summarize_tool_result(text: str) -> str:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            keep = {k: v for k, v in data.items() if k in (
                "success", "error", "message", "note",
                "experiment_token", "experiment_name",
                "nrf_lb_ip", "multus_network", "lb", "subnet",
                "vm_name", "private_ip",
                "log_file_local", "download_ok",
                "best_model", "mlflow_ui", "results_minio_url",
                "digitalObjectId", "portal_url", "artifact_uploaded",
                "do_type", "total", "results",
            )}
            return json.dumps(keep)
    except Exception:
        pass
    return text[:200]


def _build_slim_schema(tool) -> dict:
    schema     = tool.inputSchema or {}
    props      = schema.get("properties", {})
    req        = set(schema.get("required", []))
    slim_props = {}

    for pname, pdef in props.items():
        entry = {"type": pdef.get("type", "string")}

        if "description" in pdef:
            if pname in PRESERVE_FULL_DESC:
                entry["description"] = pdef["description"]
            else:
                entry["description"] = pdef["description"][:80]

        # Always keep defaults and enum constraints
        if "default" in pdef:
            entry["default"] = pdef["default"]
        if "enum" in pdef:
            entry["enum"] = pdef["enum"]

        slim_props[pname] = entry

    slim_schema = {"type": "object", "properties": slim_props}
    if req:
        slim_schema["required"] = list(req)
    return slim_schema


def _validate_tool_args(tool_name: str, tool_args: dict, state: AgentState) -> str | None:
    """
    Guard rail: check that critical params are not placeholder strings.
    Returns an error string if validation fails, None if OK.
    """
    PLACEHOLDER_PATTERN = re.compile(
        r'^(nrf_lb_ip|multus_network|session_cookie|experiment_token|'
        r'mod_auth_openidc_session|lb_ip|vm_ip|dataset_url|tracking_ip)$',
        re.IGNORECASE
    )

    if tool_name == "configure_post5g_experiment":
        for field in ("nrf_lb_ip", "multus_network", "session_cookie"):
            val = tool_args.get(field, "")
            if not val or PLACEHOLDER_PATTERN.match(str(val)):
                return (
                    f"configure_post5g_experiment was called with placeholder value "
                    f"'{val}' for '{field}'. You MUST call post5g_get_prefix first "
                    f"and use its real output. "
                    + state.as_context_block()
                )
        # Strip dnns/slices if the model passes them anyway (they are now internal)
        tool_args.pop("dnns", None)
        tool_args.pop("slices", None)

    if tool_name in ("post5g_get_prefix", "post5g_get_experiment", "post5g_launch_experiment"):
        token = tool_args.get("experiment_name") or tool_args.get("experiment_token", "")
        if not token or not token.startswith("exp_"):
            return (
                f"{tool_name} requires the full experiment_token starting with 'exp_'. "
                f"You passed: '{token}'. "
                + state.as_context_block()
            )

    if tool_name == "book_pos_calendar":
        oidc = tool_args.get("mod_auth_openidc_session", "")
        if not oidc or PLACEHOLDER_PATTERN.match(str(oidc)):
            return (
                f"book_pos_calendar: mod_auth_openidc_session is missing or is a placeholder. "
                f"Call get_slices_session first. "
                + state.as_context_block()
            )

    return None


# ── Agent ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an autonomous MLOps Agent for SLICES-RI infrastructure.
Fulfill the user's intent exactly — no extra steps, no invented values.

══ CRITICAL: USE ONLY VALUES FROM [COLLECTED VALUES] ══
A [COLLECTED VALUES] block appears at the end of this prompt after each tool call.
For ALL infrastructure fields (session_cookie, mod_auth_openidc_session,
experiment_token, experiment_name, nrf_lb_ip, multus_network, lb_ip, vm_ip, dataset_url,
tracking_ip) you MUST copy the exact value from that block.
NEVER invent, guess, or use a field name as its own value.

══ MANDATORY STEP ORDER — do not skip or reorder ══

Full experiment setup:
  STEP 1 → get_slices_session()
           Collect: session_cookie, mod_auth_openidc_session
  STEP 2 → slices_create_experiment(project_name, experiment_name)
           Collect: experiment_name (the short name, e.g., "testfull") AND experiment_token (starts with exp_...)
  STEP 3 → post5g_get_prefix(experiment_name=<experiment_token from STEP 2>)
           Collect: nrf_lb_ip, multus_network, lb_ip
           !! NEVER skip this step. nrf_lb_ip and multus_network ONLY come from here.
           post5g_get_experiment requires the full experiment_token starting with 'exp_'.
  STEP 4 → get_available_nodes()
           Confirm the node name is in the list.
  STEP 5 → configure_post5g_experiment(
               experiment_name          = <experiment_name from STEP 2 (MUST use the SHORT name here, NOT the exp_ token)>,
               nrf_lb_ip                = <nrf_lb_ip from STEP 3>,
               multus_network           = <multus_network from STEP 3>,
               pos_deployment_node      = <node from STEP 4>,
               session_cookie           = <session_cookie from STEP 1>,
               mod_auth_openidc_session = <mod_auth_openidc_session from STEP 1>
           )
           Do NOT pass dnns or slices — they do not exist as parameters.
  STEP 6 → book_pos_calendar(
               mod_auth_openidc_session = <from STEP 1>,
               node        = <same node as STEP 5>,
               start_time  = "HH:MM",
               end_time    = "HH:MM"
           )
           Do NOT pass start_date or end_date (defaults to today).
  STEP 7 → post5g_get_experiment(experiment_name=<experiment_token from STEP 2>)
  STEP 8 → post5g_launch_experiment(experiment_name=<experiment_token from STEP 2>)
  STEP 9 → trigger_5g_anomaly(lb_ip=<lb from STEP 3>)

Generate anomaly only:
  get_slices_session → slices_list_experiments → post5g_get_prefix → trigger_5g_anomaly

Create BI VM:
  bi_list_infra → bi_create_mlops_vm → bi_wait_vm_ready → bi_get_vm_ip

Deploy MLOps + train:
  bi_deploy_mlops_stack → bi_open_tunnels → upload_csv_to_minio → train_generic_model

Publish to MRS:
  list_digital_object_types → get_digital_object_schema(do_type='dataset')
  → publish_digital_object(
        do_type='dataset', 
        artifact_path='/tmp/training_results.json',
        metadata={
           "identifier": "trainres-20260601-uniqueid",
           "name": "Anomaly Detection Model Training Results",
           "description": "Training results of a RandomForest classification model for status anomaly detection.",
           "resourceType": "Dataset",
           "keywords": ["mlops", "anomaly", "slices"],
           "version": "1.0",
           "accessType": "Remote",
           "accessMode": "Free",
           "license": "BSD-3-Clause",
           "copyrightsHolder": "Flavio Olivieri",
           "scientificDomains": ["networks"],
           "scientificSubdomains": ["network protocols"],
           "datasetMetadata": {
              "rows": 322,
              "features_used": ["timestamp", "ts_delta", "ts_rolling_rate"],
              "target_column": "status",
              "best_model": "RandomForest"
           }
        }
    )

List calendar:  get_slices_session → list_pos_calendar

══ RULES ══
- Execute STRICTLY one step at a time. Never batch dependent calls.
- If success:false, stop and report.
- DO NOT MIX DOMAINS: If the user asks for a 5G experiment or calendar booking, NEVER call 'bi_' (Business Intelligence) tools, even if the project name contains words like "mlops" or "ai". Only call 'bi_' tools if explicitly asked to create a VM or deploy an MLOps stack.
- MRS tools authenticate internally — do NOT call get_slices_session before them.
- For BI-only tasks, never call post5g or slices_list_experiments.
- For MRS metadata: auto-generate rich text for name/description/keywords/identifier.
  For enum fields, use ONLY exact values from allowed_values, or omit.
  Never pass empty strings "".
- NEVER pause to talk to the user between steps. Once a tool returns a result, IMMEDIATELY call the next required tool in the sequence. Do not return plain text without a tool call until the ENTIRE sequence (up to trigger_5g_anomaly) is fully complete.
- CRITICAL SYNTAX FOR EMPTY ARGUMENTS: When calling a tool with no input parameters (such as list_digital_object_types), you MUST pass a valid empty JSON object "{}" as the arguments. NEVER output empty strings, whitespaces, or malformed brackets like {""}.
- CRITICAL FOR MRS PUBLISHING: You MUST structure the 'metadata' dictionary parameter EXACTLY as shown in the example template above.
- NESTING REQUIREMENT: You MUST include a nested dictionary object under the key "datasetMetadata" containing the dataset training statistics (rows, features_used, target_column, best_model). Do NOT flatten these fields into the root of the metadata dictionary.
- STRICT ENUM CONSTRAINT: For 'accessType' use ONLY "Remote". For 'accessMode' use ONLY "Free". NEVER invent values like "Public", "remote" (lowercase), or pass empty strings "".
"""


async def run_agent(user_intent: str):
    state = AgentState()

    async with AsyncExitStack() as stack:
        tool_to_session: dict = {}
        mcp_tools_list:  list = []

        for name, script in servers.items():
            if not os.path.exists(script):
                print(f"Warning: MCP server script not found, skipping: {script}")
                continue

            server_params   = StdioServerParameters(command=sys.executable, args=[script])
            stdio_transport = await stack.enter_async_context(stdio_client(server_params))
            read, write     = stdio_transport
            session         = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools_response = await session.list_tools()
            for tool in tools_response.tools:
                tool_to_session[tool.name] = session
                mcp_tools_list.append({
                    "type": "function",
                    "function": {
                        "name":        tool.name,
                        "description": (tool.description or "").split("\n")[0][:120],
                        "parameters":  _build_slim_schema(tool),
                    }
                })

        def _build_system_message() -> dict:
            return {
                "role":    "system",
                "content": SYSTEM_PROMPT + state.as_context_block(),
            }

        messages = [
            _build_system_message(),
            {"role": "user", "content": user_intent},
        ]

        retry_count = 0

        while True:
            # Always inject the latest collected values into the system message
            messages[0] = _build_system_message()

            # ── LLM call ─────────────────────────────────────────────────────
            try:
                response = await llm_client.chat.completions.create(
                    model=os.environ["LLM_MODEL"],
                    messages=messages,
                    tools=mcp_tools_list,
                    tool_choice="auto",
                    # Force the model to call at most one tool per turn.
                    # Prevents the model from batching all 9 steps into one response,
                    # which causes BadRequestError when dependent values are not yet known.
                    parallel_tool_calls=False,
                )
                retry_count = 0

            except openai.RateLimitError as e:
                print("The server is token-limited. Waiting 6 seconds to clear the buffer and retry...")
                await asyncio.sleep(6) # Attende che passino i 5.5s richiesti dal server
                continue

            except openai.BadRequestError as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    print(f"\n[Agent] Giving up after {MAX_RETRIES} consecutive API errors.")
                    print(f"Last error: {e}")
                    break

                error_info = {}
                if hasattr(e, "body") and isinstance(e.body, dict):
                    error_info = e.body.get("error", {})

                bad_msg    = error_info.get("message", str(e))
                failed_gen = error_info.get("failed_generation", "")
                code       = error_info.get("code", "")

                print(f"\n[Agent] BadRequestError (attempt {retry_count}/{MAX_RETRIES}): {bad_msg}")

                if "not in request.tools" in bad_msg:
                    correction = (
                        f"You called a tool that does not exist: {failed_gen}. "
                        "Only call tools from the provided list."
                    )
                elif "missing properties" in bad_msg:
                    correction = (
                        f"Tool call rejected by schema validation: {bad_msg}. "
                        "Check the [COLLECTED VALUES] block and provide ALL required fields. "
                        f"Failed call: {failed_gen}"
                    )
                elif failed_gen and failed_gen.strip().startswith("<function="):
                    # Model used legacy XML function call format instead of JSON
                    # Extract tool name from <function=name>{...}
                    import re as _re
                    fn_match = _re.match(r"<function=(\w+)>(\{.*\})", failed_gen.strip(), _re.DOTALL)
                    fn_name = fn_match.group(1) if fn_match else "unknown"
                    fn_args = fn_match.group(2) if fn_match else "{}"
                    correction = (
                        f"CRITICAL FORMAT ERROR: You output '<function={fn_name}>' which is the wrong format. "
                        "You MUST use the built-in tool_call mechanism — do NOT write <function=...> tags. "
                        "Simply call the tool normally using the tool_calls interface. "
                        f"The arguments were correct: {fn_args}. "
                        f"Now call {fn_name} again using the proper tool call format."
                    )
                elif code == "tool_use_failed" or "tool_use_failed" in bad_msg:
                    correction = (
                        "Tool call rejected — malformed arguments. "
                        "Do not pass a node name into 'dnns' or 'slices'. "
                        "Do not use field names as their own values. "
                        f"Failed: {failed_gen}"
                    )
                else:
                    correction = (
                        f"API error: {bad_msg}. "
                        f"Failed: {failed_gen}. "
                        "Retry calling ONE tool at a time with corrected arguments."
                    )

                messages.append({"role": "user", "content": correction})
                continue

            # ── Process response ──────────────────────────────────────────────
            ai_message = response.choices[0].message
            messages.append(ai_message)

            if not ai_message.tool_calls:
                print(ai_message.content)
                break

            # Defensive: if the model returned multiple tool calls despite
            # parallel_tool_calls=False, keep only the first one so that
            # state is updated before dependent calls are attempted.
            if len(ai_message.tool_calls) > 1:
                print(
                    f"[Agent] Model returned {len(ai_message.tool_calls)} tool calls "
                    "— enforcing single-step execution, keeping only the first."
                )
                try:
                    # Pydantic v2
                    messages[-1] = ai_message.model_copy(
                        update={"tool_calls": [ai_message.tool_calls[0]]}
                    )
                except AttributeError:
                    # Pydantic v1 fallback
                    messages[-1] = ai_message.copy(
                        update={"tool_calls": [ai_message.tool_calls[0]]}
                    )
                ai_message = messages[-1]

            # Prune old tool messages
            tool_indices = [
                i for i, m in enumerate(messages)
                if isinstance(m, dict) and m.get("role") == "tool"
            ]
            for i in tool_indices[:-4]:
                old = messages[i].get("content", "")
                if isinstance(old, str) and len(old) > 80:
                    messages[i]["content"] = _summarize_tool_result(old)

            # ── Tool execution ────────────────────────────────────────────────
            async def _call_tool(tool_call):
                tool_name = tool_call.function.name or ""

                # Parse arguments — guard against None, empty string, or non-dict JSON
                raw_args = tool_call.function.arguments
                try:
                    tool_args = json.loads(raw_args) if raw_args else {}
                except Exception:
                    tool_args = {}
                if not isinstance(tool_args, dict):
                    tool_args = {}

                # Strip null values so Python-side defaults kick in
                tool_args = {k: v for k, v in tool_args.items() if v is not None}

                # Guard rail: validate args before calling
                validation_error = _validate_tool_args(tool_name, tool_args, state)
                if validation_error:
                    print(f"[{tool_name}] ✗ Validation failed: {validation_error[:200]}\n")
                    return tool_call.id, json.dumps({
                        "success": False,
                        "error":   validation_error,
                    })

                session = tool_to_session.get(tool_name)
                if not session:
                    result_text = json.dumps({
                        "success": False,
                        "error":   f"Tool '{tool_name}' not found in any registered MCP server.",
                    })
                    print(f"[{tool_name}] → {result_text}\n")
                    return tool_call.id, result_text

                result      = await session.call_tool(tool_name, tool_args)
                result_text = result.content[0].text if result.content else "{}"

                # Auto-refresh session if configure returns 500 (likely expired cookie)
                if tool_name == "configure_post5g_experiment":
                    try:
                        r_data = json.loads(result_text)
                        if not r_data.get("success") and r_data.get("status_code") == 500:
                            print(f"[{tool_name}] Session likely expired (500), refreshing...")
                            auth_session = tool_to_session.get("get_slices_session")
                            if auth_session:
                                fresh = await auth_session.call_tool(
                                    "get_slices_session", {"force_refresh": True}
                                )
                                fresh_text = fresh.content[0].text if fresh.content else "{}"
                                state.update_from_tool("get_slices_session", fresh_text)
                                # Retry configure with fresh cookies
                                tool_args["session_cookie"]           = state.session_cookie
                                tool_args["mod_auth_openidc_session"] = state.mod_auth_openidc_session
                                result      = await session.call_tool(tool_name, tool_args)
                                result_text = result.content[0].text if result.content else "{}"
                                print(f"[{tool_name}] Retry after session refresh → {result_text[:200]}\n")
                    except Exception:
                        pass

                # Update state with extracted values
                state.update_from_tool(tool_name, result_text)

                print(f"[{tool_name}] → {result_text[:1200]}\n")
                return tool_call.id, truncate_result(result_text)

            # Sequential for auth tools or multi-call batches; parallel otherwise
            calls    = ai_message.tool_calls
            has_auth = any(tc.function.name in SLICES_AUTH_TOOLS for tc in calls)

            if has_auth or len(calls) > 1:
                results = []
                for tc in calls:
                    results.append(await _call_tool(tc))
            else:
                results = await asyncio.gather(*[_call_tool(tc) for tc in calls])

            messages.extend([
                {"role": "tool", "tool_call_id": tc_id, "content": text}
                for tc_id, text in results
            ])


if __name__ == "__main__":
    intent = input("Insert an intent: ")
    asyncio.run(run_agent(intent))