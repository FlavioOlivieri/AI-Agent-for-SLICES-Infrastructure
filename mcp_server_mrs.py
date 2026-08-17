import logging

class _SuppressMRSFailureDump(logging.Filter):
    def filter(self, record):
        return "Failed request details" not in record.getMessage()

root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig()  
for _h in root_logger.handlers:
    _h.addFilter(_SuppressMRSFailureDump())

import sys
sys.modules.setdefault("mcp_server_mrs", sys.modules["__main__"])

import json
import re
from datetime import datetime, UTC
from pathlib import Path
import os

import requests
from mcp.server.fastmcp import FastMCP

from slices_cli_dm.publisher import DigitalObjectPublisher
from slices_cli_dm.utils import (
    RequestError,
    compress_to_temp,
    create_mrs_entry,
    get_authtoken_user_project_id,
    get_mrs_entry,
    get_mrs_token,
    get_schemas,
    get_search_path_mapping,
    get_size,
    get_swagger,
    patch_mrs_entry,
    search_mrs_entries,
    upload_artifact,
)

# ── Constants ──────────────────────────────────────────────────────────────────

MRS_BACKEND = "https://mrs-backend.slices-staging.slices-be.eu"
MRS_PORTAL  = "https://mrs-portal.slices-staging.slices-be.eu"
DMI_TOKEN_URL = "https://sts.slices-staging.slices-be.eu/realms/dmi/protocol/openid-connect/token"

_DO_RE = re.compile(r"^DigitalObjectModel_(.*)")

# Fields populated automatically from user profile – agent need not supply them
_AUTO_FILLED = {
    "$type", "creators", "copyrightsHolder", "createdAt",
    "dateTimeStart", "dateTimeEnd", "version", "license",
    "scientificDomains", "scientificSubdomains", "resourceType", "downloadUrl",
}

mcp = FastMCP("mrs")

# ── Helpers ────────────────────────────────────────────────────────────────────

class _MRSToken:
    """
    A minimal wrapper to satisfy the interface required by
    slices_cli_dm.utils (an object with an .access_token attribute),
    given that the new 'dmi' Keycloak realm flow returns the token
    string directly, rather than the auth object returned by the
    original get_mrs_token().
    """
    def __init__(self, token_response: dict):
        self.access_token = token_response["access_token"]
        self.token_type   = token_response.get("token_type", "Bearer")
        self.expires_in   = token_response.get("expires_in")
        self.refresh_token = token_response.get("refresh_token")

def _get_mrs_token_dmi() -> _MRSToken:
    """
    Token MRS from Keycloak dedicated client for external access from Slices CLI.
    """
    resp = requests.post(
        DMI_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "client_id":     os.environ["MRS_CLI_CLIENT_ID"],
            "client_secret": os.environ["MRS_CLI_CLIENT_SECRET"],
            "grant_type":    "password",
            "username":      os.environ["MRS_CLI_USERNAME"],
            "password":      os.environ["MRS_CLI_PASSWORD"],
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if "access_token" not in data:
        raise RequestError(f"No access_token in MRS token response: {resp.text[:300]}")
    return _MRSToken(data)

def _auth():
    auth_token, user, project_id = get_authtoken_user_project_id()
    mrs_token = _get_mrs_token_dmi()
    return auth_token, user, project_id, mrs_token

def _portal_url(digital_object_id: int) -> str:
    return f"{MRS_PORTAL}/app/management/digital-objects/view/{digital_object_id}"


def _load_schemas():
    """Return (swagger, schemas, do_keys, do_names)."""
    swagger = get_swagger()
    schemas = get_schemas(swagger)
    do_keys  = [s for s in schemas if _DO_RE.search(s)]
    do_names = [_DO_RE.search(k).group(1).lower() for k in do_keys]
    return swagger, schemas, do_keys, do_names


def _mrs_url(do_type_lower: str) -> str:
    return f"{MRS_BACKEND}/v0.2/digital-objects"


def _err(msg: str) -> str:
    return json.dumps({"success": False, "error": msg})

def _create_mrs_entry_v2(meta_data: str, mrs_token, url: str) -> dict:
    """
    Local replacement for slices_cli_dm.utils.create_mrs_entry: the installed
    package only accepts a 200 status with a dict body, but v0.2 now responds
    with 201 Created and a body consisting solely of the digitalObjectId (raw JSON, not a dict).
    """
    access_token = mrs_token.access_token
    token_type   = mrs_token.token_type
    headers = {
        'accept':        'text/plain',
        'Authorization': f'{token_type} {access_token}',
        'Content-Type':  'application/json',
    }
    response = requests.post(url, headers=headers, data=meta_data)

    if response.status_code in (200, 201):
        body = response.json()
        if isinstance(body, dict):
            return body
        return {"digitalObjectId": body}

    raise RequestError(
        f"Request failed (Status Code: {response.status_code})\nResponse: {response.text}"
    )

def _reorder_type_first(d: dict) -> dict:
    """v0.2 requires `$type` as the first property in the JSON (.NET/System.Text.Json polymorphic discriminator)."""
    if "$type" in d:
        return {"$type": d["$type"], **{k: v for k, v in d.items() if k != "$type"}}
    return d

_READONLY_FIELDS = {"internalIdentifier", "systemCreationDate", "metadataProfile"}

def _patch_mrs_entry_v2(digital_object_id: int, updates: dict, mrs_token,
                         url: str = "https://mrs-backend.slices-staging.slices-be.eu/v0.2/digital-objects") -> dict:
    """
    Local replacement for slices_cli_dm.utils.patch_mrs_entry: the v0.2 backend
    does not perform a true merge patch; the .NET deserializer always requires
    the complete DigitalObjectModel object, even for PATCH requests (otherwise
    it returns a 400 error: 'missing required properties: identifier,
    systemCreationDate, ...'). This wrapper performs a GET on the existing
    entry, applies the 'updates', and sends back the full body with the
    $type field reordered to the first position.
    """
    access_token = mrs_token.access_token
    token_type   = mrs_token.token_type
    headers = {
        'accept':        'text/plain',
        'Authorization': f'{token_type} {access_token}',
        'Content-Type':  'application/json',
    }

    full_entry = get_mrs_entry(digital_object_id, mrs_token)
    for field in _READONLY_FIELDS:
        full_entry[field] = None
    merged = {**full_entry, **updates}
    merged = _reorder_type_first(merged)

    response = requests.patch(f"{url}/{digital_object_id}", headers=headers, data=json.dumps(merged))

    if response.status_code in (200, 201):
        if response.text.strip():
            try:
                return response.json()
            except Exception:
                return {"digitalObjectId": digital_object_id}
        return {"digitalObjectId": digital_object_id}

    raise RequestError(
        f"Request failed (Status Code: {response.status_code})\nResponse: {response.text}"
    )

def _create_and_upload_v2(rendered_metadata: dict, artifact_location: str, mrs_token,
                           url: str = "https://mrs-backend.slices-staging.slices-be.eu/v0.2/digital-objects/upload") -> dict:
    """
    v0.2 combines create and upload into a single multipart call (CreateAndUpload),
    replacing the old two-step mini-dmi flow (create -> separate PUT
    to uploadUrl, which no longer exists in v0.2). The response is simply
    the digitalObjectId (int) as the raw body, not a dict.
    """
    access_token = mrs_token.access_token
    token_type   = mrs_token.token_type
    headers = {
        'accept':        'text/plain',
        'Authorization': f'{token_type} {access_token}',
    }
    with open(artifact_location, 'rb') as f:
        files = {
            'file':            (Path(artifact_location).name, f, 'application/octet-stream'),
            'metadata': (None, json.dumps(rendered_metadata)),
        }
        response = requests.post(url, headers=headers, files=files)

    if response.status_code in (200, 201):
        body = response.json()
        return body if isinstance(body, dict) else {"digitalObjectId": body}

    raise RequestError(f"Request failed (Status Code: {response.status_code})\nResponse: {response.text}")

# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_digital_object_types() -> str:
    """List all MRS digital object types. Call first to get valid do_type values."""
    try:
        _, _, _, do_names = _load_schemas()
    except Exception as e:
        return _err(str(e))
    return json.dumps({"success": True, "types": do_names})


@mcp.tool()
def get_digital_object_schema(do_type: str) -> str:
    """
    Return fields for a digital object type (compact).
    auto_filled fields (creators, dates, license…) are set automatically.
    Only provide fields listed under 'fields'; required=true ones are mandatory.
    """
    try:
        _, schemas, do_keys, do_names = _load_schemas()
    except Exception as e:
        return _err(str(e))

    do_type_lower = do_type.lower()
    if do_type_lower not in do_names:
        return _err(f"Unknown type '{do_type}'. Available: {do_names}")

    do_key    = do_keys[do_names.index(do_type_lower)]
    publisher = DigitalObjectPublisher(schemas=schemas, do_type=do_key, mrs_url="")
    props     = publisher.get_properties()

    fields = []
    for name, defn in props.items():
        if name in _AUTO_FILLED:
            continue

        one_of = []
        for ref in defn.get("oneOf", []):
            s = schemas.get(ref.get("$ref", "").split("/")[-1], {})
            if s.get("type") in ("string", "number", "integer", "boolean"):
                one_of.extend(s.get("enum", []))

        entry = {
            "name":     name,
            "type":     defn.get("type") or defn.get("$ref", "").split("/")[-1],
            "required": publisher._check_required(name),
        }
        if defn.get("format"):
            entry["format"] = defn["format"]
        if one_of:
            entry["allowed_values"] = one_of
        if defn.get("type") == "array":
            entry["items_type"] = defn.get("items", {}).get("type")
        fields.append(entry)

    return json.dumps({
        "success":    True,
        "do_type":    do_type_lower,
        "auto_filled": sorted(_AUTO_FILLED),
        "fields":     fields,
    })

 
@mcp.tool()
def publish_digital_object(
    do_type: str,
    metadata: dict,
    artifact_path: str = None,
) -> str:
    """
    Publish a digital object to SLICES MRS. Mirrors 'slices dm publish'.
    auto_filled fields (creators, dates, license) are set from user profile.
    Do NOT set '$type' in metadata – it is derived from the schema.
    Use get_digital_object_schema for the list of required/optional fields.
    artifact_path: optional local file or directory to compress and upload.
    Returns: success, digitalObjectId, portal_url.
    """
    # 1. Load Swagger / schemas
    try:
        _, schemas, do_keys, do_names = _load_schemas()
    except Exception as e:
        return _err(f"Failed to load Swagger: {e}")
 
    do_type_lower = do_type.lower()
    if do_type_lower not in do_names:
        return _err(f"Unknown type '{do_type}'. Available: {do_names}")
 
    do_key  = do_keys[do_names.index(do_type_lower)]
    mrs_url = _mrs_url(do_type_lower)
 
    # 2. Authenticate
    try:
        _, user, _, mrs_token = _auth()
    except Exception as e:
        return _err(f"Authentication failed: {e}")
 
    # 3. Publisher (used for schema helpers only)
    publisher     = DigitalObjectPublisher(schemas=schemas, do_type=do_key, mrs_url=mrs_url)
    resource_type = publisher.get_type_mapping().get(do_key)
 
    # 4. Defaults (same as publisher.execute())
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    defaults = {
        "creators": [{
            "firstName":    user.first_name,
            "lastName":     user.last_name,
            "email":        user.email,
            "organization": user.organisation,
        }],
        "copyrightsHolder":    f"{user.first_name} {user.last_name}",
        "createdAt":           now,
        "dateTimeStart":       now,
        "dateTimeEnd":         now,
        "version":             "1.0",
        "license":             "BSD-3-Clause",
        "scientificDomains":   ["networks"],
        "scientificSubdomains": ["network protocols"],
        "resourceType":        resource_type,
        "accessType":          "Other",   
        "accessMode":          "Free",
    }
 
    # 5. Artifact
    artifact_location = None
    if artifact_path:
        if not Path(artifact_path).exists():
            return _err(f"Artifact path not found: {artifact_path}")
        defaults.update({
            "format":            "application/x-tar",
            "compressionFormat": "application/gzip",
            "byteSize": get_size(artifact_path),
        })
        artifact_location = compress_to_temp(artifact_path)
 
    _VALID_ACCESS_TYPES = {"Remote", "Physical", "Virtual", "Other"}
    _VALID_ACCESS_MODES = {"Free", "FreeConditionally", "ExcellenceBased"}
 
    if metadata.get("accessType") not in _VALID_ACCESS_TYPES:
        metadata.pop("accessType", None)
    if metadata.get("accessMode") not in _VALID_ACCESS_MODES:
        metadata.pop("accessMode", None)
 
    # 6. Merge: defaults < caller metadata
    context = {**defaults, **{k: v for k, v in metadata.items() if v is not None}}
 
    # 7. Render Jinja2 template → validated JSON
    try:
        rendered = publisher._render_template(publisher.create_jinja_template(), context)
    except Exception as e:
        return _err(f"Template rendering failed: {e}")
 
    if "$type" in rendered:
        rendered = {"$type": rendered["$type"], **{k: v for k, v in rendered.items() if k != "$type"}}
 
    # 8. Create MRS entry
    try:
        if artifact_location:
            mrs_entry = _create_and_upload_v2(rendered, artifact_location, mrs_token)
        else:
            mrs_entry = _create_mrs_entry_v2(json.dumps(rendered), mrs_token, url=mrs_url)
    except RequestError as e:
        return _err(f"MRS entry creation failed: {e}")
    except Exception as e:
        return _err(f"MRS entry creation failed: {e}")
 
    if not isinstance(mrs_entry, dict):
        return _err(f"Unexpected MRS response shape (expected dict, got {type(mrs_entry).__name__}): {mrs_entry!r}")
 
    digital_object_id = mrs_entry.get("digitalObjectId")
 
    return json.dumps({
        "success":           True,
        "do_type":           do_type_lower,
        "digitalObjectId":   digital_object_id,
        "portal_url":        _portal_url(digital_object_id),
        "artifact_uploaded": artifact_location is not None,
    })



@mcp.tool()
def patch_digital_object(internal_identifier: int, metadata: dict) -> str:
    """
    Patch an existing MRS digital object by internal ID. Mirrors 'slices dm patch'.
    Only fields present in metadata are updated; others remain unchanged.
    """
    try:
        _, _, _, mrs_token = _auth()
    except Exception as e:
        return _err(f"Authentication failed: {e}")
    try:
        result = patch_mrs_entry(internal_identifier, json.dumps(metadata), mrs_token)
        return json.dumps({"success": True, "updated_entry": result})
    except (RequestError, Exception) as e:
        return _err(f"Patch failed: {e}")


@mcp.tool()
def search_digital_objects(
    do_type: str = "global",
    name_query: str = None,
    advanced_query: str = None,
    limit: int = 10,
    page: int = 0,
) -> str:
    """
    Search MRS digital objects. do_type='global' searches all types.
    Provide name_query (free text) or advanced_query (Lucene, e.g. 'keywords:5G').
    Returns: total, page, results list with id/type/name/identifier/portal_url.
    """
    if not name_query and not advanced_query:
        return _err("Provide name_query or advanced_query.")
    try:
        _, _, _, mrs_token = _auth()
    except Exception as e:
        return _err(f"Authentication failed: {e}")
 
    # get_search_path_mapping() from slices_cli_dm.utils returns an empty map
    # against the current v0.2 swagger (verified via /swagger/v0.2/swagger.json —
    # it likely expects a different URL pattern than the backend actually uses).
    # The real search paths are /v0.2/digital-objects/search/{plural-type}, e.g.
    # 'dataset' -> 'datasets', 'catalogue' -> 'catalogues', 'slicesnode' -> 'slicesNodes'.
    # Hardcoded here instead of depending on that (currently broken) helper.
    _SEARCH_PATH_MAP = {
        "global":      "/v0.2/digital-objects/search/global",
        "base":        "/v0.2/digital-objects/search/base",
        "dataset":     "/v0.2/digital-objects/search/datasets",
        "snsdataset":  "/v0.2/digital-objects/search/sns-datasets",
        "publication": "/v0.2/digital-objects/search/publications",
        "service":     "/v0.2/digital-objects/search/services",
        "catalogue":   "/v0.2/digital-objects/search/catalogues",
        "resource":    "/v0.2/digital-objects/search/resources",
        "slicesnode":  "/v0.2/digital-objects/search/slicesNodes",
    }
    path_map = _SEARCH_PATH_MAP
 
    search_path = path_map.get(do_type.lower())
    if not search_path:
        return _err(f"No search endpoint for '{do_type}'. Available: {list(path_map)}")
 
    try:
        result = search_mrs_entries(
            search_path=search_path, mrs_token=mrs_token,
            name_query=name_query, advanced_query=advanced_query,
            page_index=page, page_size=limit,
        )
    except (RequestError, Exception) as e:
        return _err(f"Search failed: {e}")
 
    return json.dumps({
        "success": True,
        "total":   result.get("total", 0),
        "page":    page,
        "results": [
            {
                "id":         item.get("internalIdentifier"),
                "type":       item.get("$type"),
                "name":       item.get("name"),
                "createdAt":  item.get("createdAt"),
                "identifier": item.get("identifier"),
                "portal_url": _portal_url(item.get("internalIdentifier")),
            }
            for item in result.get("items", [])
        ],
    })


@mcp.tool()
def get_digital_object(digital_object_id: int) -> str:
    """Retrieve full metadata of a digital object by its internal integer ID."""
    try:
        _, _, _, mrs_token = _auth()
    except Exception as e:
        return _err(f"Authentication failed: {e}")
    try:
        entry = get_mrs_entry(digital_object_id, mrs_token)
        entry["portal_url"] = _portal_url(digital_object_id)
        return json.dumps({"success": True, "entry": entry})
    except (RequestError, Exception) as e:
        return _err(f"Retrieval failed: {e}")

import mrs_lifecycle

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")