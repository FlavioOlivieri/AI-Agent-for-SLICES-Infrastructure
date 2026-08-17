from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, UTC

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from execution_spec import ExecutionSpecification, Task, TaskStatus, TOOL_TO_SERVER

SERVERS = {
    "slices": "mcp_server_slices.py",
    "bi":     "mcp_server_bi.py",
    "mlflow": "mcp_server_mlflow.py",
    "mrs":    "mcp_server_mrs.py",
}

class WorkflowExecutionError(Exception):
    def __init__(self, task_id: str, message: str):
        self.task_id = task_id
        super().__init__(f"Task '{task_id}' failed: {message}")


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    output: dict = field(default_factory=dict)
    error: str | None = None
    attempts: int = 0


class WorkflowExecutor:
    """
    Executes a ExecutionSpecification by calling the real MCP tools.
    Typical usage:

        spec = example_ddos_workflow()
        executor = WorkflowExecutor(spec)
        await executor.run()
    """

    def __init__(self, spec: ExecutionSpecification):
        self.spec = spec
        self.results: dict[str, TaskResult] = {}
        self._tool_to_session: dict = {}
        self._mrs_session = None 

    # ── Setup MCP connections ────────────────────────────────────────────

    async def _connect_servers(self, stack: AsyncExitStack):
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
                self._tool_to_session[tool.name] = session

            if server_name == "mrs":
                self._mrs_session = session

    # ── Lifecycle tracking (MRS) ────────────────────────────────────────

    async def _record_event(self, event_type: str, task_id: str | None, payload: dict):
        """
        Call the record_lifecycle_event tool as any other MCP tool — not a direct import — to stay 
        consistent with the principle that MCP tools are the only connectors used by the engine.
        """
        if self._mrs_session is None:
            print(f"[lifecycle] MRS session not available, event '{event_type}' not recorded.")
            return
        try:
            params = {
                "event_type":  event_type,
                "workflow_id": self.spec.workflow_id,
                "payload":     payload,
            }
            if task_id is not None:
                params["task_id"] = task_id

            result = await self._mrs_session.call_tool("record_lifecycle_event", params)
            print(f"[DEBUG] raw result: {result!r}")
            text = result.content[0].text if result.content else "{}"
            data = json.loads(text)
            if not data.get("success"):
                print(f"[lifecycle] Registration of event '{event_type}' failed: {data.get('error')}")
        except Exception as e:
            print(f"[lifecycle] Error recording event '{event_type}': {e}")

    # ── Resolution of references ${task.output} ────────────────────────

    def _resolve_params(self, task: Task) -> dict:
        """
        Replaces each value '${task_id.output_name}' in the task parameters
        with the actual value from self.results — the only place where data
        "travels" between steps, instead of injecting AgentState into the prompt in agent.py.
        """
        import re
        _REF = re.compile(r"^\$\{([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\}$")
 
        def _resolve_value(value, context: str):
            if isinstance(value, str):
                m = _REF.match(value)
                if m:
                    ref_task_id, ref_field = m.group(1), m.group(2)
                    if ref_task_id not in self.results:
                        raise WorkflowExecutionError(
                            task.id,
                            f"Reference to '{ref_task_id}' in {context} "
                            f"but that task has not been executed yet.",
                        )
                    ref_output = self.results[ref_task_id].output
                    if ref_field not in ref_output:
                        raise WorkflowExecutionError(
                            task.id,
                            f"Task '{ref_task_id}' did not produce field '{ref_field}'. "
                            f"Available fields: {list(ref_output.keys())}",
                        )
                    return ref_output[ref_field]
                return value
            elif isinstance(value, dict):
                return {k: _resolve_value(v, f"{context}.{k}") for k, v in value.items()}
            elif isinstance(value, list):
                return [_resolve_value(item, f"{context}[{i}]") for i, item in enumerate(value)]
            else:
                return value
 
        return {key: _resolve_value(val, key) for key, val in task.params.items()}


    # ── Execution of a single task, with retry ────────────────────────

    async def _execute_task(self, task: Task) -> TaskResult:
        session = self._tool_to_session.get(task.tool)
        if session is None:
            error = f"Tool '{task.tool}' not found in any connected MCP server."
            await self._record_event("task_failed", task.id, {"error": error})
            return TaskResult(task.id, TaskStatus.FAILED, error=error)

        try:
            params = self._resolve_params(task)
        except WorkflowExecutionError as e:
            await self._record_event("task_failed", task.id, {"error": str(e)})
            return TaskResult(task.id, TaskStatus.FAILED, error=str(e))

        await self._record_event("task_started", task.id, {"tool": task.tool, "params": params})

        last_error = None
        for attempt in range(1, task.retry_policy.max_attempts + 1):
            try:
                result = await session.call_tool(task.tool, params)
                text = result.content[0].text if result.content else "{}"
                data = json.loads(text) if text else {}

                looks_like_failure = isinstance(data, dict) and (
                    data.get("success") is False
                    or ("error" in data and "success" not in data)
                )
                if looks_like_failure:
                    base_error = (
                        data.get("error")
                        or data.get("server_message")
                        or data.get("message")
                        or f"Tool returned success=false with no error detail: {data}"
                    )
                    extra_parts = []
                    for key in ("raw", "stderr", "stdout", "response_snippet", "status_code"):
                        val = data.get(key) if isinstance(data, dict) else None
                        if val:
                            extra_parts.append(f"{key}: {str(val)[:500]}")
                    last_error = f"{base_error} | {' | '.join(extra_parts)}" if extra_parts else base_error
                    if attempt < task.retry_policy.max_attempts:
                        await self._record_event(
                            "task_retried", task.id,
                            {"attempt": attempt, "error": last_error},
                        )
                        await asyncio.sleep(task.retry_policy.backoff_seconds)
                        continue
                    break

                # Success
                output = data if isinstance(data, dict) else {"raw": data}
                await self._record_event(
                    "task_succeeded", task.id,
                    {"attempts": attempt, "output": output},
                )
                return TaskResult(task.id, TaskStatus.SUCCEEDED, output=output, attempts=attempt)

            except Exception as e:
                last_error = str(e)
                if attempt < task.retry_policy.max_attempts:
                    await self._record_event(
                        "task_retried", task.id,
                        {"attempt": attempt, "error": last_error},
                    )
                    await asyncio.sleep(task.retry_policy.backoff_seconds)
                    continue
                break

        await self._record_event(
            "task_failed", task.id,
            {"attempts": task.retry_policy.max_attempts, "error": last_error},
        )
        return TaskResult(task.id, TaskStatus.FAILED, error=last_error, attempts=task.retry_policy.max_attempts)

    # ── Execution of the entire workflow ─────────────────────────────────

    async def run(self) -> dict[str, TaskResult]:
        async with AsyncExitStack() as stack:
            await self._connect_servers(stack)

            await self._record_event(
                "spec_generated", None,
                {
                    "intent":     self.spec.intent,
                    "task_count": len(self.spec.tasks),
                    "spec": self.spec.model_dump(mode="json"),
                },
            )

            waves = self.spec.topological_order()
            print(f"[executor] Workflow '{self.spec.workflow_id}': {len(waves)} wave to execute.")

            for wave_index, wave in enumerate(waves):
                print(f"[executor] Wave {wave_index}: {wave}")
                for task_id in wave:
                    task = self.spec.task_by_id(task_id)
                    print(f"[executor]   → executing '{task_id}' ({task.tool})")
                    task_result = await self._execute_task(task)
                    self.results[task_id] = task_result

                    if task_result.status == TaskStatus.FAILED:
                        print(f"[executor] ✗ Task '{task_id}' failed: {task_result.error}")
                        print("[executor] Interrupting execution — no subsequent waves will be executed.")
                        await self._record_event(
                            "workflow_completed", None,
                            {
                                "status": "failed",
                                "failed_task": task_id,
                                "error": task_result.error,
                                "completed_tasks": [
                                    tid for tid, r in self.results.items()
                                    if r.status == TaskStatus.SUCCEEDED
                                ],
                            },
                        )
                        return self.results

                    print(f"[executor] ✓ Task '{task_id}' completed.")

            await self._record_event(
                "workflow_completed", None,
                {
                    "status": "succeeded",
                    "completed_tasks": list(self.results.keys()),
                },
            )
            print(f"[executor] Workflow '{self.spec.workflow_id}' completed successfully.")
            return self.results