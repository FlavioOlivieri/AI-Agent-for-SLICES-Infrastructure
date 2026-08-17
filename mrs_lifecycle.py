from __future__ import annotations
import json
from datetime import datetime, UTC
from enum import Enum
from typing import Any
from mcp.server.fastmcp import FastMCP
from mcp_server_mrs import (
    mcp,
    _load_schemas,
    _auth,
    _portal_url,
    _err,
    _mrs_url,
    _create_mrs_entry_v2,
    _patch_mrs_entry_v2,
)
from slices_cli_dm.publisher import DigitalObjectPublisher
from slices_cli_dm.utils import create_mrs_entry, RequestError
from slices_cli_dm.utils import (
    get_mrs_entry, patch_mrs_entry,
    get_swagger, get_search_path_mapping, search_mrs_entries,
)

_WORKFLOW_DO_ID: dict[str, int] = {}

class LifecycleEventType(str, Enum):
    INTENT_RECEIVED   = "intent_received"
    INTENT_FAILED     = "intent_failed"
    SPEC_GENERATED    = "spec_generated"
    TASK_STARTED      = "task_started"
    TASK_SUCCEEDED    = "task_succeeded"
    TASK_FAILED       = "task_failed"
    TASK_RETRIED      = "task_retried"
    WORKFLOW_COMPLETED = "workflow_completed"

_PREFERRED_DO_TYPES = ["dataset"]
_FALLBACK_DO_TYPE   = "dataset"
_TYPES_WITH_DEDICATED_METADATA_BLOCK: set[str] = set()

_resolved_do_type: str | None = None


def _find_existing_workflow_do(workflow_id: str, do_type: str, mrs_token) -> int | None:
    """Retrieve the ID of an existing lifecycle digital object for this
    workflow_id"""
    try:
        swagger = get_swagger()
        path_map = get_search_path_mapping(swagger)
        search_path = path_map.get(do_type)
        if not search_path:
            return None
        result = search_mrs_entries(
            search_path=search_path, mrs_token=mrs_token,
            advanced_query=f'keywords:"workflow:{workflow_id}"',
            page_index=0, page_size=1,
        )
        items = result.get("items", [])
        return items[0].get("internalIdentifier") if items else None
    except Exception:
        return None

def _resolve_lifecycle_do_type() -> tuple[str, bool]:
    """
    Determine which do_type to use for registering lifecycle events.
    Returns (do_type, has_dedicated_metadata_block).
    has_dedicated_metadata_block=False (current case for "base") means
    the payload should be serialized inside description/provenance instead
    of in a nested *Metadata field.
    """
    global _resolved_do_type
    if _resolved_do_type is not None:
        return _resolved_do_type, _resolved_do_type in _TYPES_WITH_DEDICATED_METADATA_BLOCK

    _, _, _, do_names = _load_schemas()

    for candidate in _PREFERRED_DO_TYPES:
        if candidate in do_names:
            _resolved_do_type = candidate
            return candidate, candidate in _TYPES_WITH_DEDICATED_METADATA_BLOCK

    _resolved_do_type = _FALLBACK_DO_TYPE
    return _FALLBACK_DO_TYPE, _FALLBACK_DO_TYPE in _TYPES_WITH_DEDICATED_METADATA_BLOCK


def _build_event_metadata(
    event_type: str,
    workflow_id: str,
    task_id: str | None,
    payload: dict[str, Any],
    has_dedicated_metadata_block: bool,
) -> dict[str, Any]:
    """
    Build the metadata dictionary to pass to publish_digital_object.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    core_event = {
        "event_type":  event_type,
        "workflow_id": workflow_id,
        "task_id":     task_id,
        "timestamp":   now,
        "payload":     payload,
    }

    provenance_line = (
        f"[{now}] workflow={workflow_id} "
        f"task={task_id or '-'} event={event_type}"
    )

    keywords = ["lifecycle", "mlops", "slices", event_type, f"workflow:{workflow_id}"]
    if task_id:
        keywords.append(f"task:{task_id}")

    metadata = {
        "identifier":  f"lifecycle-{workflow_id}-{task_id or 'workflow'}-{int(datetime.now(UTC).timestamp())}",
        "name":        f"Lifecycle event: {event_type} ({workflow_id})",
        "description": json.dumps(core_event, default=str),
        "provenance":  provenance_line,
        "keywords":    keywords,
    }

    if has_dedicated_metadata_block:
        metadata["lifecycleEventMetadata"] = core_event
    else:
        metadata["accessType"] = "Other"
        metadata["accessMode"] = "Free"

    return metadata


@mcp.tool()
def record_lifecycle_event(
    event_type: str,
    workflow_id: str,
    task_id: str | None = None,
    payload: dict | None = None,
) -> str:
    """
    Logs a lifecycle event on MRS (intent, generated spec, task
    started/completed/failed/retried, or workflow completed).

    Unlike publish_digital_object—currently called only once
    at the end of the pipeline—this tool is designed to be called at every
    significant state transition during workflow execution,
    so that the entire lifecycle can be reconstructed by MRS, not just
    the final result.

    event_type: one of intent_received, spec_generated, task_started,
    task_succeeded, task_failed, task_retried, workflow_completed.
    workflow_id: workflow identifier (see ExecutionSpecification.workflow_id).
    task_id: task identifier, if the event is task-specific
    (None for workflow-wide events).
    payload: free data associated with the event—e.g., the natural language intent,
    the serialized spec, the error of a failed task,
    the outputs produced by a successful task.

    Returns the same format as publish_digital_object: success,
    digitalObjectId, portal_url, plus do_type_used (now always "base" on
    MRS staging, confirmed via list_digital_object_types/get_digital_object_schema).
    """
    valid_types = {e.value for e in LifecycleEventType}
    if event_type not in valid_types:
        return _err(f"event_type '{event_type}' not valid. Valid types: {sorted(valid_types)}")

    payload = payload or {}
    try:
        do_type, has_dedicated_metadata_block = _resolve_lifecycle_do_type()
    except Exception as e:
        return _err(f"Impossible to determine the do_type for the lifecycle event: {e}")

    try:
        _, schemas, do_keys, do_names = _load_schemas()
    except Exception as e:
        return _err(f"Failed to load Swagger: {e}")

    do_key  = do_keys[do_names.index(do_type)]
    mrs_url = _mrs_url(do_type)

    try:
        _, user, _, mrs_token = _auth()
    except Exception as e:
        return _err(f"Authentication failed: {e}")

    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    new_event = {"event_type": event_type, "task_id": task_id, "timestamp": now, "payload": payload}

    existing_do_id = _WORKFLOW_DO_ID.get(workflow_id) or _find_existing_workflow_do(workflow_id, do_type, mrs_token)

    publisher     = DigitalObjectPublisher(schemas=schemas, do_type=do_key, mrs_url=mrs_url)
    resource_type = publisher.get_type_mapping().get(do_key)

    keywords = ["lifecycle", "mlops", "slices", f"workflow:{workflow_id}"]
    provenance_line = f"[{now}] workflow={workflow_id} last_event={event_type}"

    if existing_do_id is None:
        # ── Primo evento del workflow: crea il digital object ──
        history = [new_event]
        metadata = {
            "identifier":  f"lifecycle-{workflow_id}",
            "name":        f"Lifecycle: {workflow_id}",
            "description": json.dumps({"workflow_id": workflow_id, "events": history}, default=str),
            "provenance":  provenance_line,
            "keywords":    keywords,
            "accessType":  "Other",
            "accessMode":  "Free",
        }
        defaults = {
            "creators": [{"firstName": user.first_name, "lastName": user.last_name,
                           "email": user.email, "organization": user.organisation}],
            "copyrightsHolder": f"{user.first_name} {user.last_name}",
            "createdAt": now, "dateTimeStart": now, "dateTimeEnd": now,
            "version": "1.0", "license": "BSD-3-Clause",
            "scientificDomains": ["networks"], "scientificSubdomains": ["network protocols"],
            "resourceType": resource_type,
        }
        context = {**defaults, **metadata}
        try:
            rendered = publisher._render_template(publisher.create_jinja_template(), context)
        except Exception as e:
            return _err(f"Template rendering failed: {e}")
        if "$type" in rendered:
            rendered = {"$type": rendered["$type"], **{k: v for k, v in rendered.items() if k != "$type"}}
        try:
            mrs_entry = _create_mrs_entry_v2(json.dumps(rendered), mrs_token, url=mrs_url)
        except Exception as e:
            return _err(f"MRS entry creation failed: {e}")
        digital_object_id = mrs_entry.get("digitalObjectId")
        _WORKFLOW_DO_ID[workflow_id] = digital_object_id

    else:
        # ── Eventi successivi: patch, accodando alla history esistente ──
        try:
            current = get_mrs_entry(existing_do_id, mrs_token)
            history = json.loads(current.get("description", "{}")).get("events", [])
        except Exception:
            history = []
        history.append(new_event)

        patch_payload = {
            "description": json.dumps({"workflow_id": workflow_id, "events": history}, default=str),
            "provenance":  provenance_line,
        }
        try:
            _patch_mrs_entry_v2(existing_do_id, patch_payload, mrs_token)
        except Exception as e:
            return _err(f"MRS entry patch failed: {e}")
        digital_object_id = existing_do_id
        _WORKFLOW_DO_ID[workflow_id] = existing_do_id

    return json.dumps({
        "success": True, "event_type": event_type, "workflow_id": workflow_id,
        "task_id": task_id, "do_type_used": do_type,
        "digitalObjectId": digital_object_id,
        "portal_url": _portal_url(digital_object_id),
    })


@mcp.tool()
def get_lifecycle_events(workflow_id: str, limit: int = 50) -> str:
    """
    Retrieves all lifecycle events recorded for a workflow_id,
    searching them on MRS using search_digital_objects.
    """
    try:
        do_type, _ = _resolve_lifecycle_do_type()
    except Exception as e:
        return _err(f"Impossible to determine the do_type for the lifecycle event: {e}")

    try:
        _, user, _, mrs_token = _auth()
    except Exception as e:
        return _err(f"Authentication failed: {e}")

    try:
        from slices_cli_dm.utils import get_swagger, get_search_path_mapping, search_mrs_entries
        swagger  = get_swagger()
        path_map = get_search_path_mapping(swagger)
        search_path = path_map.get(do_type)
        if not search_path:
            return _err(f"No search endpoint for type '{do_type}'.")

        result = search_mrs_entries(
            search_path=search_path, mrs_token=mrs_token,
            name_query=workflow_id,
            page_index=0, page_size=limit,
        )
    except Exception as e:
        return _err(f"Search failed: {e}")

    return json.dumps({
        "success":     True,
        "workflow_id": workflow_id,
        "total":       result.get("total", 0),
        "events":      result.get("items", []),
    })