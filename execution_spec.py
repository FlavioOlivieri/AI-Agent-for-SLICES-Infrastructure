from __future__ import annotations
import re
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

MCP_SERVERS = {
    "slices": {
        "get_slices_session",
        "slices_create_experiment",
        "slices_list_experiments",
        "post5g_get_prefix",
        "get_available_nodes",
        "configure_post5g_experiment",
        "book_pos_calendar",
        "list_pos_calendar",
        "delete_pos_calendar",
        "post5g_get_experiment",
        "post5g_launch_experiment",
        "trigger_5g_anomaly",
    },
    "bi": {
        "bi_list_infra",
        "bi_get_cli_help",
        "bi_create_mlops_vm",
        "bi_wait_vm_ready",
        "bi_get_vm_ip",
        "bi_transfer_file_from_post5g",
        "bi_run_command",
    },
    "mlflow": {
        "bi_deploy_mlops_stack",
        "bi_open_tunnels",
        "upload_csv_to_minio",
        "train_generic_model",
        "bi_close_tunnels",
        "download_artifact_from_minio",
    },
    "mrs": {
        "list_digital_object_types",
        "get_digital_object_schema",
        "publish_digital_object",
        "patch_digital_object",
        "search_digital_objects",
        "get_digital_object",
        "record_lifecycle_event",
    },
}

TOOL_TO_SERVER: dict[str, str] = {
    tool: server for server, tools in MCP_SERVERS.items() for tool in tools
}

ALL_TOOLS: set[str] = set(TOOL_TO_SERVER.keys())

class TaskStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"
    SKIPPED   = "skipped"


class ResourceKind(str, Enum):
    POST5G_EXPERIMENT = "post5g_experiment"
    BI_VM             = "bi_vm"
    MLFLOW_TRACKING   = "mlflow_tracking"
    DATASET           = "dataset"

TOOL_PROVIDES_RESOURCE: dict[str, ResourceKind] = {
    "slices_create_experiment": ResourceKind.POST5G_EXPERIMENT,
    "bi_create_mlops_vm":       ResourceKind.BI_VM,
    "bi_open_tunnels":          ResourceKind.MLFLOW_TRACKING,
    "trigger_5g_anomaly":       ResourceKind.DATASET,
}

TOOL_REQUIRES_RESOURCE: dict[str, set[ResourceKind]] = {
    "post5g_launch_experiment": {ResourceKind.POST5G_EXPERIMENT},
    "trigger_5g_anomaly":       {ResourceKind.POST5G_EXPERIMENT},
    "bi_deploy_mlops_stack":    {ResourceKind.BI_VM},
    "bi_open_tunnels":          {ResourceKind.BI_VM},
    "upload_csv_to_minio":      {ResourceKind.MLFLOW_TRACKING},
    "train_generic_model":      {ResourceKind.MLFLOW_TRACKING},
}

_REF_PATTERN = re.compile(r"^\$\{([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\}$")


class OutputRef(BaseModel):
    raw: str

    @field_validator("raw")
    @classmethod
    def must_match_ref_pattern(cls, v: str) -> str:
        if not _REF_PATTERN.match(v):
            raise ValueError(
                f"'{v}' is not a valid reference. Required format: ${{task_id.output_name}}"
            )
        return v

    @property
    def task_id(self) -> str:
        return _REF_PATTERN.match(self.raw).group(1)

    @property
    def output_name(self) -> str:
        return _REF_PATTERN.match(self.raw).group(2)


ParamValue = str | int | float | bool | list | dict | None


class RetryPolicy(BaseModel):
    max_attempts:    int = Field(default=1, ge=1, le=10)
    backoff_seconds: int = Field(default=0, ge=0, le=300)


class Task(BaseModel):
    id: str = Field(description="Unique identifier for the task within the workflow")
    tool: str = Field(description="Name of the MCP tool to invoke — must exist in ALL_TOOLS")
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, ParamValue] = Field(default_factory=dict)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    declared_outputs: list[str] = Field(
        default_factory=list,
        description="Names of the fields that other tasks can reference as ${this_task.<name>}",
    )

    @field_validator("tool")
    @classmethod
    def tool_must_exist(cls, v: str) -> str:
        if v not in ALL_TOOLS:
            raise ValueError(f"Tool '{v}' does not exist in any registered MCP server.")
        return v

    @property
    def server(self) -> str:
        return TOOL_TO_SERVER[self.tool]

    def referenced_task_ids(self) -> set[str]:
        """Extract the IDs of tasks referenced in the parameters via ${task.output}."""
        found = set()
        for v in self.params.values():
            if isinstance(v, str):
                m = _REF_PATTERN.match(v)
                if m:
                    found.add(m.group(1))
        return found


class LifecycleTracking(BaseModel):
    registry: Literal["mrs"] = "mrs"
    record_intent:        bool = True
    record_spec:          bool = True
    record_task_events:   bool = True
    record_final_outputs: bool = True

class ExecutionSpecification(BaseModel):
    workflow_id: str = Field(description="Unique identifier for this execution")
    intent: str = Field(description="Original intent in natural language, verbatim")
    generated_by: Literal["llm-intent-layer"] = "llm-intent-layer"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    tasks: list[Task] = Field(min_length=1)
    lifecycle_tracking: LifecycleTracking = Field(default_factory=LifecycleTracking)

    @model_validator(mode="after")
    def task_ids_are_unique(self) -> "ExecutionSpecification":
        ids = [t.id for t in self.tasks]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Task id duplicated: {sorted(dupes)}")
        return self

    @model_validator(mode="after")
    def dependencies_exist(self) -> "ExecutionSpecification":
        ids = {t.id for t in self.tasks}
        for t in self.tasks:
            unknown = set(t.depends_on) - ids
            if unknown:
                raise ValueError(f"Task '{t.id}' depends on non-existent IDs: {sorted(unknown)}")
        return self

    @model_validator(mode="after")
    def output_refs_match_declared_dependencies(self) -> "ExecutionSpecification":
        """
        If a task references ${X.output} in its parameters, X must be
        present in depends_on. This is the check that is currently done
        at runtime (and sometimes fails) by _validate_tool_args in agent.py —
        here it is done statically, before executing anything.
        """
        by_id = {t.id: t for t in self.tasks}
        for t in self.tasks:
            referenced = t.referenced_task_ids()
            missing = referenced - set(t.depends_on)
            if missing:
                raise ValueError(
                    f"Task '{t.id}' refers to output of {sorted(missing)} "
                    f"but doesn't declare them in depends_on."
                )
            for ref_id in referenced:
                if ref_id not in by_id:
                    raise ValueError(f"Task '{t.id}' refers to non-existent task '{ref_id}'.")
        return self

    @model_validator(mode="after")
    def output_refs_are_declared(self) -> "ExecutionSpecification":
        by_id = {t.id: t for t in self.tasks}
        for t in self.tasks:
            for v in t.params.values():
                if isinstance(v, str) and _REF_PATTERN.match(v):
                    ref_id, ref_field = _REF_PATTERN.match(v).groups()
                    upstream = by_id.get(ref_id)
                    if upstream and upstream.declared_outputs and ref_field not in upstream.declared_outputs:
                        raise ValueError(
                            f"Task '{t.id}' references '{ref_id}.{ref_field}', but '{ref_id}' "
                            f"only declares outputs {upstream.declared_outputs}."
                        )
        return self

    @model_validator(mode="after")
    def graph_has_no_cycles(self) -> "ExecutionSpecification":
        """Topological check — a cycle here means the engine will never be able to execute the workflow."""
        graph = {t.id: set(t.depends_on) for t in self.tasks}
        visited, in_progress = set(), set()

        def visit(node: str):
            if node in visited:
                return
            if node in in_progress:
                raise ValueError(f"Cycle of dependencies detected involving task '{node}'.")
            in_progress.add(node)
            for dep in graph.get(node, set()):
                visit(dep)
            in_progress.remove(node)
            visited.add(node)

        for node in graph:
            visit(node)
        return self

    @model_validator(mode="after")
    def resource_dependencies_are_satisfied(self) -> "ExecutionSpecification":
        """
        If a task requires a resource (e.g., a Post5G experiment, a BI
        VM, an MLFlow tracking server), at least one of its dependencies (direct or transitive) must provide it.
        """
        by_id = {t.id: t for t in self.tasks}
        memo: dict[str, set[str]] = {}

        def ancestors(tid: str) -> set[str]:
            if tid in memo:
                return memo[tid]
            acc: set[str] = set(by_id[tid].depends_on)
            for dep in by_id[tid].depends_on:
                acc |= ancestors(dep)
            memo[tid] = acc
            return acc

        for t in self.tasks:
            required = TOOL_REQUIRES_RESOURCE.get(t.tool, set())
            if not required:
                continue
            provided = {
                TOOL_PROVIDES_RESOURCE[by_id[anc_id].tool]
                for anc_id in ancestors(t.id)
                if by_id[anc_id].tool in TOOL_PROVIDES_RESOURCE
            }
            missing = required - provided
            if missing:
                raise ValueError(
                    f"Task '{t.id}' (tool '{t.tool}') requires the resources "
                    f"{sorted(r.value for r in missing)} but no dependency "
                    f"provides this resource. Add a task to depends_on that calls "
                    f"a tool that provides it."
                )
        return self

    # ── Utility for the execution engine ─────────────────────────────────────

    def topological_order(self) -> list[list[str]]:
        """
        Returns the tasks grouped in "waves" that can be executed in parallel.
        wave[0] = tasks without dependencies, wave[1] = tasks that depend only
        on tasks in wave[0], etc. Used by the engine to decide what to run
        in parallel (today decided ad-hoc in agent.py with asyncio.gather).
        """
        remaining = {t.id: set(t.depends_on) for t in self.tasks}
        done: set[str] = set()
        waves: list[list[str]] = []

        while remaining:
            ready = [tid for tid, deps in remaining.items() if deps <= done]
            if not ready:
                raise RuntimeError("Unsolvable dependencies — this should not happen after validation.")
            waves.append(sorted(ready))
            done.update(ready)
            for tid in ready:
                remaining.pop(tid)

        return waves

    def task_by_id(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(task_id)