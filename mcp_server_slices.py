from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import json
import subprocess
import re
import time
from typing import Optional
import base64
from datetime import date

mcp = FastMCP("slices")
load_dotenv()

PORTAL_URL     = "https://portal.slices-ri.eu"
CALENDAR_URL   = "https://duckburg.net.cit.tum.de/calendar"
BASE           = "https://duckburg.net.cit.tum.de"
SSH_HOST       = "duckburg.net.in.tum.de"
SSH_PORT       = 10022
SLICES_VENV    = os.environ.get("SLICES_VENV_PATH", os.path.expanduser("~/slices-venv"))
SLICES_BIN     = f"{SLICES_VENV}/bin"
POST5G_URL     = "https://post-5g-web.slices-ri.eu"
SUCCESS_BANNER = "Experiment has been uploaded to the pos nodes"

NODE_SPECS = {
    "standard-2-1":  {"cpu": 8,  "ram_gb": 16,  "gpu": False, "bare_metal": False},
    "standard-2-2":  {"cpu": 8,  "ram_gb": 16,  "gpu": False, "bare_metal": False},
    "standard-4-1":  {"cpu": 16, "ram_gb": 32,  "gpu": False, "bare_metal": False},
    "standard-4-2":  {"cpu": 16, "ram_gb": 32,  "gpu": False, "bare_metal": False},
    "sopnode-f1":    {"cpu": 32, "ram_gb": 64,  "gpu": False, "bare_metal": False},
    "sopnode-f2":    {"cpu": 32, "ram_gb": 64,  "gpu": False, "bare_metal": False},
    "sopnode-w2":    {"cpu": 32, "ram_gb": 64,  "gpu": False, "bare_metal": False},
    "sopnode-w3":    {"cpu": 32, "ram_gb": 64,  "gpu": False, "bare_metal": False},
    "sopnode-bm100": {"cpu": 64, "ram_gb": 128, "gpu": True,  "bare_metal": True},
    "sopnode-bm101": {"cpu": 64, "ram_gb": 128, "gpu": True,  "bare_metal": True},
    "sopnode-bm102": {"cpu": 64, "ram_gb": 128, "gpu": True,  "bare_metal": True},
    "sopnode-bm103": {"cpu": 64, "ram_gb": 128, "gpu": True,  "bare_metal": True},
    "r2lab":         {"cpu": 16, "ram_gb": 32,  "gpu": False, "bare_metal": False},
    "scrooge":       {"cpu": 32, "ram_gb": 64,  "gpu": False, "bare_metal": True},
}

_UA = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:149.0) Gecko/20100101 Firefox/149.0"

_SESSION_CACHE: dict | None = None

# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_cmd(args: list, timeout: int = 60) -> dict:
    """Run a CLI command inside the slices venv."""
    env = os.environ.copy()
    env["PATH"] = f"{SLICES_BIN}:{env.get('PATH', '')}"
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
        return {"returncode": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": f"Command timed out after {timeout}s"}
    except FileNotFoundError:
        return {"returncode": -1, "stdout": "", "stderr": f"Command not found: {args[0]}"}


def _sync_remote_slices_auth() -> dict:
    """
    Copies the local ~/.slices/auth.json to duckburg before running any `post5g`
    command there. `post5g` shells out to `slices auth id-token` / `slices experiment jwt`
    using ITS OWN local SLICES CLI session on duckburg — which is independent from the
    session used by _run_cmd() on this machine. Without this sync, an expired/missing
    session on duckburg causes post5g to send an empty/invalid Bearer token, and the
    backend returns a generic "Internal Server Error" instead of a clear auth error.

    access_token is short-lived (~1h), so this is called before every post5g invocation
    rather than once per workflow run.
    """
    ssh_user = os.environ.get("SLICES_USER")
    local_auth_path = os.path.expanduser("~/.slices/auth.json")

    if not os.path.exists(local_auth_path):
        return {"success": False, "error": f"Local auth file not found: {local_auth_path}"}

    ssh_key_path = os.path.expanduser(os.environ.get("SSH_KEY_PATH", "~/.ssh/id_ed25519"))
    control_path = f"/tmp/slices_ssh_{SSH_HOST}_{SSH_PORT}"

    # Detect a stale ControlMaster socket (file present but master connection dead)
    # and remove it before attempting scp — otherwise scp tries to reuse the dead
    # multiplexed connection, hangs, and only fails once our own timeout fires.
    if os.path.exists(control_path):
        check_cmd = [
            "ssh", "-O", "check",
            "-o", f"ControlPath={control_path}",
            f"{ssh_user}@{SSH_HOST}",
        ]
        try:
            chk = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
            if chk.returncode != 0:
                os.remove(control_path)
        except Exception:
            try:
                os.remove(control_path)
            except OSError:
                pass

    def _do_scp() -> dict:
        scp_cmd = [
            "scp", "-P", str(SSH_PORT), "-i", ssh_key_path,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={control_path}",
            "-o", "ControlPersist=120",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2",
            local_auth_path,
            f"{ssh_user}@{SSH_HOST}:.slices/auth.json",
        ]
        try:
            r = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                return {"success": False, "error": r.stderr.strip()}
            return {"success": True}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "scp timed out after 30s"}
        except Exception as e:
            return {"success": False, "error": f"Local scp error: {e}"}

    result = _do_scp()
    if not result["success"]:
        # One retry after nuking a possibly-stale socket: covers the case where
        # the master died between our check above and the actual scp call.
        try:
            if os.path.exists(control_path):
                os.remove(control_path)
        except OSError:
            pass
        result = _do_scp()
    return result


def _run_remote_cmd(command: str, timeout: int = 300) -> dict:
    """Run a command on the SLICES webshell via SSH."""
    ssh_user     = os.environ.get("SLICES_USER")
    ssh_key_path = os.path.expanduser(os.environ.get("SSH_KEY_PATH", "~/.ssh/id_ed25519"))
    ssh_cmd = [
        "ssh", "-p", str(SSH_PORT), "-i", ssh_key_path,
        "-A",                              # forward SSH agent for onward SCP/SSH from remote
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=15",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath=/tmp/slices_ssh_{SSH_HOST}_{SSH_PORT}",
        "-o", "ControlPersist=120",
        f"{ssh_user}@{SSH_HOST}", command,
    ]
    try:
        r   = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        out = r.stdout.strip()
        m   = re.search(r'\{.*\}', out, re.DOTALL)
        if m:
            out = m.group(0).strip()
        return {"returncode": r.returncode, "stdout": out, "stderr": r.stderr.strip()}
    except subprocess.TimeoutExpired as e:
        # e.stdout/e.stderr hold whatever the process produced before being killed —
        # capture it instead of discarding, so we can tell "still deploying, slow"
        # apart from "hung waiting on an interactive prompt with no output".
        partial_out = (e.stdout or "").strip() if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="replace").strip()
        partial_err = (e.stderr or "").strip() if isinstance(e.stderr, str) else (e.stderr or b"").decode(errors="replace").strip()
        return {
            "returncode": -1,
            "stdout": partial_out,
            "stderr": f"Command timed out after {timeout}s" + (f" — partial stderr: {_tail(partial_err)}" if partial_err else ""),
        }
    except Exception as e:
        return {"returncode": -1, "stdout": "", "stderr": f"Local SSH error: {e}"}


def _tail(text: str, lines: int = 20) -> str:
    """Return the last N lines of a string."""
    return "\n".join(text.strip().splitlines()[-lines:])


def _make_session(
    session_cookie: str = None,
    oidc_session: str = None,
    domain_session: str = None,
    domain_oidc: str = None,
) -> requests.Session:
    """Build a requests.Session with the standard User-Agent and optional cookies."""
    s = requests.Session()
    s.headers["User-Agent"] = _UA
    if session_cookie and domain_session:
        s.cookies.set("session", session_cookie, domain=domain_session)
    if oidc_session and domain_oidc:
        s.cookies.set("mod_auth_openidc_session", oidc_session, domain=domain_oidc)
    return s


def _build_form_data(
    experiment_name, pos_deployment_node, xp_url,
    core_namespace, core_cleanup, nrf_lb_ip,
    ran_cleanup, ran_split_f1, ran_split_e1, ran_split_du_present,
    gnb_id, ue_present, ue_cleanup, ue_namespace,
    flexric_present, multus_network, multus_host_interface,
    mcc, mnc, tac, sst, dnns, slices,
) -> dict:
    """Build the form payload for configure_post5g_experiment."""
    form = {
        "POS.deploymentNode":         pos_deployment_node,
        "exp-name":                   experiment_name,
        "expFile":                    "",
        "xp-url":                     xp_url,
        "GCN.config_files":           "oai-cn5g-fed/",
        "GCN.core.present":           "true",
        "GCN.core.namespace":         core_namespace,
        "GCN.core.nrfLoadBalancerIP": nrf_lb_ip,
        "GCN.core.cleanup":           str(core_cleanup).lower(),
        "GCN.RAN.present":            "true",
        "GCN.RAN.namespace":          "ran",
        "GCN.RAN.cleanup":            str(ran_cleanup).lower(),
        "GCN.RAN.split.f1":           str(ran_split_f1).lower(),
        "GCN.RAN.split.e1":           str(ran_split_e1).lower(),
        "GCN.RAN.split.du_present":   str(ran_split_du_present).lower(),
        "GCN.gNB.gnbid":              gnb_id,
        "GCN.UE.present":             str(ue_present).lower(),
        "GCN.UE.cleanup":             str(ue_cleanup).lower(),
        "GCN.UE.namespace":           ue_namespace,
        "GCN.flexric.present":        str(flexric_present).lower(),
        "GCN.multus.network":         multus_network,
        "GCN.multus.hostInterface":   multus_host_interface,
        "GCN.mcc":                    mcc,
        "GCN.mnc":                    mnc,
        "GCN.tac":                    tac,
        "GCN.sst":                    sst,
        "ueList":                     "[]",
        "ranConfig":                  "",
    }

    for i, dnn in enumerate(dnns):
        form[f"GCN.dnns.{i}.dnn"]              = dnn["dnn"]
        form[f"GCN.dnns.{i}.pdu_session_type"] = dnn["pdu_session_type"]
        form[f"GCN.dnns.{i}.ipv4_subnet"]      = dnn["ipv4_subnet"]

    for i, sl in enumerate(slices):
        form[f"GCN.slices.{i}.snssai.sst"] = sl["snssai_sst"]
        if sl.get("snssai_sd"):
            form[f"GCN.slices.{i}.snssai.sd"] = sl["snssai_sd"]
        for j, plnm in enumerate(sl.get("plnms", [])):
            form[f"GCN.slices.{i}.plnms.{j}.mcc"] = plnm["mcc"]
            form[f"GCN.slices.{i}.plnms.{j}.mnc"] = plnm["mnc"]
        for j, dnn in enumerate(sl.get("dnns", [])):
            form[f"GCN.slices.{i}.dnns.{j}"] = dnn
        form[f"GCN.slices.{i}.qos_profile.5qi"] = sl["qos_5qi"]
        if sl.get("qos_ul"):
            form[f"GCN.slices.{i}.qos_profile.session_ambr_ul"] = sl["qos_ul"]
        if sl.get("qos_dl"):
            form[f"GCN.slices.{i}.qos_profile.session_ambr_dl"] = sl["qos_dl"]

    return form


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_slices_session(force_refresh: bool = False) -> str:
    """
    Login to SLICES portal. Completes OIDC for BOTH duckburg AND post-5g-web.
    ALWAYS call this first. Results are cached for the lifetime of the process.
    Pass force_refresh=True only if you get a 401 error.
    """
    global _SESSION_CACHE

    if _SESSION_CACHE is not None and not force_refresh:
        return json.dumps(_SESSION_CACHE, indent=2)

    username = os.environ.get("SLICES_USER")
    password = os.environ.get("SLICES_PASS")

    def _follow_oidc(req_session: requests.Session, start_url: str, final_url: str) -> dict:
        try:
            resp  = req_session.get(start_url, allow_redirects=True, timeout=20)
            steps = 0
            while steps < 8:
                steps += 1
                if resp.url.rstrip("/") == start_url.rstrip("/"):
                    break
                if "/login" in resp.url and resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    csrf = soup.find("input", {"name": "csrf_token"})
                    if csrf:
                        resp = req_session.post(
                            resp.url,
                            data={"csrf_token": csrf.get("value", ""), "username": username, "password": password},
                            headers={"Origin": f"https://{urlparse(resp.url).netloc}", "Referer": resp.url},
                            allow_redirects=True, timeout=20,
                        )
                        continue
                if resp.status_code == 200 and "<form" in resp.text:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    form = soup.find("form")
                    if not form:
                        break
                    action = form.get("action") or resp.url
                    if action.startswith("/"):
                        p = urlparse(resp.url)
                        action = f"{p.scheme}://{p.netloc}{action}"
                    form_data = {
                        inp.get("name"): inp.get("value", "")
                        for inp in form.find_all("input") if inp.get("name")
                    }
                    resp = req_session.post(action, data=form_data, allow_redirects=True, timeout=20)
                    continue
                break

            cookies_here = [
                f"{c.name}@{c.domain}" for c in req_session.cookies
                if final_url.replace("https://", "") in c.domain or c.domain in final_url
            ]
            return {"ok": bool(cookies_here), "final_url": resp.url,
                    "cookies_added": cookies_here, "steps": steps}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    try:
        req_session = requests.Session()
        req_session.headers["User-Agent"] = _UA

        # Portal login
        login_url  = f"{PORTAL_URL}/login"
        login_page = req_session.get(login_url)
        login_page.raise_for_status()
        soup = BeautifulSoup(login_page.text, "html.parser")
        csrf = soup.find("input", {"name": "csrf_token"})
        if csrf:
            req_session.post(
                login_url,
                data={"csrf_token": csrf["value"], "username": username, "password": password},
                headers={"Origin": PORTAL_URL, "Referer": login_url},
            )

        duck_result = _follow_oidc(req_session, CALENDAR_URL, "duckburg.net.cit.tum.de")
        p5g_result  = _follow_oidc(req_session, f"{POST5G_URL}/edit", "post-5g-web.slices-ri.eu")
        if not p5g_result["ok"]:
            p5g_result = _follow_oidc(req_session, f"{POST5G_URL}/login", "post-5g-web.slices-ri.eu")

        def _get_cookie(domain_fragment: str, name: str):
            return next(
                (c.value for c in req_session.cookies if c.name == name and domain_fragment in c.domain),
                None,
            )

        result = {
            "session":                  _get_cookie("post-5g-web", "session")
                                        or next((c.value for c in req_session.cookies if c.name == "session"), None),
            "mod_auth_openidc_session": _get_cookie("duckburg", "mod_auth_openidc_session"),
            "_post5g_oidc":             p5g_result,
            "_duck_oidc":               duck_result,
            "_all_cookie_domains":      [f"{c.name}@{c.domain}" for c in req_session.cookies],
        }

        _SESSION_CACHE = result
        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def slices_create_experiment(project_name: str, experiment_name: str, duration: str = "2h") -> str:
    """
    Creates a new SLICES experiment under a specific project via CLI.
    If an experiment with the same name already exists, returns it as-is.
    Duration format: 2h, 1d, 2d, etc.
    """
    list_cmd = _run_cmd(["slices", "project", "list"])
    if list_cmd["returncode"] != 0:
        return json.dumps({"success": False, "error": "Failed to fetch project list.", "details": list_cmd["stderr"]}, indent=2)
    if project_name not in list_cmd["stdout"]:
        return json.dumps({"success": False, "error": f"Project '{project_name}' does not exist."}, indent=2)

    use_cmd = _run_cmd(["slices", "project", "use", project_name])
    if use_cmd["returncode"] != 0:
        return json.dumps({"success": False, "error": f"Failed to select project '{project_name}'.", "details": use_cmd["stderr"]}, indent=2)

    r = _run_cmd(["slices", "experiment", "create", experiment_name, "--duration", duration])
    if r["returncode"] != 0:
        if any(k in r["stderr"].lower() for k in ("already exists", "bad request")):
            token_match = re.search(r"exp_\S+", r["stderr"])
            token = token_match.group(0) if token_match else None
            return json.dumps({
                "success":          True,
                "experiment_name":  experiment_name,
                "experiment_token": token,
                "duration":         duration,
                "note": (
                    "Experiment already exists — reusing it. "
                    "Use experiment_token (not experiment_name) for post5g_get_prefix, "
                    "post5g_get_experiment and post5g_launch_experiment."
                ),
                "raw": r["stderr"],
            }, indent=2)
        return json.dumps({"success": False, "error": r["stderr"], "raw": r["stdout"]}, indent=2)

    token_match = re.search(r"exp_\S+", r["stdout"])
    token = token_match.group(0) if token_match else r["stdout"]

    return json.dumps({
        "success":          True,
        "experiment_name":  experiment_name,
        "experiment_token": token,
        "duration":         duration,
        "note": (
            "Use experiment_token (not experiment_name) for post5g_get_prefix, "
            "post5g_get_experiment and post5g_launch_experiment."
        ),
    }, indent=2)


@mcp.tool()
def slices_list_experiments(project_name: str) -> str:
    """
    Lists all active experiments in a project and extracts their full experiment tokens.

    Call this to get the experiment_token for an EXISTING experiment before calling
    post5g_get_prefix, post5g_get_experiment or post5g_launch_experiment.
    The experiment_token (e.g. exp_expauth.ilabt.imec.be_01ks...) is REQUIRED
    by all post5g commands — the short name alone is not enough.
    """
    use_cmd = _run_cmd(["slices", "project", "use", project_name])
    if use_cmd["returncode"] != 0:
        return json.dumps({"success": False, "error": use_cmd["stderr"]}, indent=2)

    r = _run_cmd(["slices", "experiment", "list"])
    if r["returncode"] != 0:
        return json.dumps({"success": False, "error": r["stderr"]}, indent=2)

    # Extract tokens (format: exp_<authority>_<id>) from the CLI table output
    # Strip ANSI escape codes and unicode box-drawing chars before regex search
    clean = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", r["stdout"])
    tokens = re.findall(r"exp_[A-Za-z0-9._-]+", clean)

    return json.dumps({
        "success":            True,
        "project":            project_name,
        "experiment_tokens":  tokens,
        "raw_output":         r["stdout"],
    }, indent=2)


@mcp.tool()
def post5g_get_prefix(experiment_name: str) -> str:
    """
    Runs: post5g experiment prefix <name> on the SLICES webshell via SSH.
    Returns nrf_lb_ip and multus_network needed for configure_post5g_experiment.
    MUST be called after slices_create_experiment and before configure_post5g_experiment.
    """
    MAX_ATTEMPTS = 4
    RETRY_DELAY  = 10  # seconds between attempts

    sync = _sync_remote_slices_auth()
    if not sync["success"]:
        return json.dumps({
            "success": False,
            "error": f"Failed to sync SLICES auth session to duckburg: {sync['error']}",
        }, indent=2)

    last_error = {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        r = _run_remote_cmd(f"post5g experiment prefix {experiment_name}", timeout=300)

        if r["returncode"] != 0:
            last_error = {"error": r["stderr"] or "Non-zero return code", "raw": r["stdout"]}
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY)
            continue

        try:
            data = json.loads(r["stdout"])
            return json.dumps({
                "success":         True,
                "subnet":          data["subnet"],
                "lb":              data["lb"],
                "expiration_time": data["expiration_time"],
                "nrf_lb_ip":       data["lb"],
                "multus_network":  data["subnet"],
                "_attempts":       attempt,
            }, indent=2)
        except json.JSONDecodeError:
            last_error = {
                "error": "Failed to parse JSON output",
                "raw": r["stdout"],
                "stderr": r["stderr"],
            }
        except KeyError as e:
            last_error = {
                "error": f"Missing field: {e}",
                "raw": r["stdout"],
                "stderr": r["stderr"],
            }

        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_DELAY)

    return json.dumps({"success": False, **last_error, "_attempts": MAX_ATTEMPTS}, indent=2)


@mcp.tool()
def get_available_nodes() -> str:
    """
    Returns all available SLICES nodes and their hardware specs.
    Call this before configure_post5g_experiment to pick a valid pos_deployment_node.
    """
    return json.dumps({"success": True, "available_nodes": list(NODE_SPECS.keys()), "node_details": NODE_SPECS}, indent=2)


@mcp.tool()
def configure_post5g_experiment(
    session_cookie: str,
    mod_auth_openidc_session: str,
    experiment_name: str,
    pos_deployment_node: str,
    nrf_lb_ip: str,
    multus_network: str,
    core_namespace: str = "core",
    core_cleanup: bool = True,
    ran_cleanup: bool = True,
    ran_split_f1: bool = True,
    ran_split_e1: bool = True,
    ran_split_du_present: bool = True,
    gnb_id: str = "0xe01",
    ue_present: bool = False,
    ue_cleanup: bool = False,
    ue_namespace: str = "ran",
    flexric_present: bool = True,
    multus_host_interface: str = "br0",
    mcc: str = "001",
    mnc: str = "01",
    tac: str = "0x0001",
    sst: str = "1",
    xp_url: str = (
        "https://gitlab.inria.fr/slices-ri/blueprints/post-5g/examples"
        "/-/archive/simple_ping/examples-simple_ping.tar.gz"
    ),
) -> str:
    """
    Sends the full Post-5G experiment configuration to the portal.

    PARAMETER ORIGIN (never invent these — copy from previous tool results):
      session_cookie      ← get_slices_session()   → "session"
      nrf_lb_ip           ← post5g_get_prefix()    → "nrf_lb_ip"
      multus_network      ← post5g_get_prefix()    → "multus_network"
      pos_deployment_node ← get_available_nodes()  → any name from "available_nodes"
                            Example valid values: "standard-2-1", "sopnode-f1"

    dnns and slices are fixed internally — do NOT expose them as parameters.
    """
    # Fixed defaults — never passed by the model
    dnns = [
        {"dnn": "oai", "pdu_session_type": "IPV4",   "ipv4_subnet": "12.1.1.0/24"},
        {"dnn": "ims", "pdu_session_type": "IPV4V6", "ipv4_subnet": "14.1.1.0/24"},
    ]
    slices = [
        {
            "snssai_sst": "1", "snssai_sd": None,
            "plnms": [{"mcc": "001", "mnc": "01"}],
            "dnns": ["oai"], "qos_5qi": "5", "qos_ul": "200Mbps", "qos_dl": "400Mbps",
        },
        {
            "snssai_sst": "1", "snssai_sd": "FFFFFF",
            "plnms": [{"mcc": "001", "mnc": "01"}],
            "dnns": ["ims"], "qos_5qi": "2", "qos_ul": None, "qos_dl": None,
        },
    ]

    form_data = _build_form_data(
        experiment_name, pos_deployment_node, xp_url,
        core_namespace, core_cleanup, nrf_lb_ip,
        ran_cleanup, ran_split_f1, ran_split_e1, ran_split_du_present,
        gnb_id, ue_present, ue_cleanup, ue_namespace,
        flexric_present, multus_network, multus_host_interface,
        mcc, mnc, tac, sst, dnns, slices,
    )

    try:
        s    = _make_session(session_cookie, mod_auth_openidc_session, "post-5g-web.slices-ri.eu", "post-5g-web.slices-ri.eu")
        resp = s.post(
            f"{POST5G_URL}/edit", data=form_data,
            headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                     "Content-Type": "application/x-www-form-urlencoded",
                     "Origin": POST5G_URL, "Referer": f"{POST5G_URL}/edit"},
            allow_redirects=True, timeout=120,
        )
        success = resp.status_code == 200 and SUCCESS_BANNER in resp.text
        return json.dumps({
            "success":            success,
            "status_code":        resp.status_code,
            "experiment_name":    experiment_name,
            "node":               pos_deployment_node,
            "nrf_lb_ip":          nrf_lb_ip,
            "multus_network":     multus_network,
            "mcc":                mcc, "mnc": mnc,
            "ue_present":         ue_present,
            "flexric_present":    flexric_present,
            "dnns_count":         len(dnns),
            "slices_count":       len(slices),
            "message":            SUCCESS_BANNER if success else "Check response_snippet for errors",
            "new_session_cookie": resp.cookies.get("session", session_cookie),
            "response_snippet":   resp.text[:400],
        }, indent=2)
    except requests.exceptions.RequestException as e:
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
def book_pos_calendar(
    mod_auth_openidc_session: str,
    node: str,
    start_time: str,
    end_time: str,
    owner: str = None,
    session_cookie: str = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None, 
) -> str:
    """
    Reserves a node on the POS calendar.
    start_date/end_date: YYYY-MM-DD format. Defaults to TODAY if not specified.
    owner: SLICES username. If omitted, uses SLICES_USER from environment.
    node: POS node name. If the user specifies a VM flavor like 'standard-2-1',
    use that value directly as the node name.
    """
    today = date.today().isoformat()
    start_date = start_date or today
    end_date   = end_date   or today
    owner = owner or os.environ.get("SLICES_USER", "unknown")
    event_id = int(time.time() * 1000)
    payload  = {
        "data": {
            str(event_id): {
                "start_date": f"{start_date} {start_time}",
                "end_date":   f"{end_date} {end_time}",
                "id":         event_id,
                "owner":      owner,
                "script":     None,
                "repeat":     None,
                "nodes":      [node],
                "!nativeeditor_status": "inserted",
            }
        }
    }
    try:
        s    = _make_session(session_cookie, mod_auth_openidc_session, "duckburg.net.cit.tum.de", "duckburg.net.cit.tum.de")
        resp = s.post(
            f"{BASE}/calendar_update?editing=true", json=payload,
            headers={"Accept": "*/*", "Content-Type": "application/json",
                     "X-Requested-With": "XMLHttpRequest",
                     "Origin": BASE, "Referer": f"{BASE}/calendar"},
            timeout=30,
        )
        try:
            body       = resp.json()
            event_data = body.get("data", {}).get(str(event_id), {})
            action     = event_data.get("action", "")
            srv_msg    = event_data.get("message", "")
            success    = resp.status_code in (200, 201) and action == "inserted"
        except Exception:
            action  = ""
            srv_msg = ""
            success = False 
        return json.dumps({
            "success":          success,
            "status_code":      resp.status_code,
            "owner":            owner,
            "node":             node,
            "time_period":      f"{start_date} {start_time} to {end_date} {end_time}",
            "event_id":         event_id,
            "server_action":    action,
            "server_message":   srv_msg,
            "message":          "Booking successful" if success else (srv_msg or "Booking not confirmed by server"),
            "response_snippet": resp.text[:400],
        }, indent=2)
    except requests.exceptions.RequestException as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
def list_pos_calendar(
    mod_auth_openidc_session: str,
    owner: str = None,
    date_from: str = None,
    date_to: str = None,
) -> str:
    """
    Fetches bookings from the POS calendar (duckburg).
    Call this before delete_pos_calendar to get the server-assigned event id.

    REQUIRED: mod_auth_openidc_session from get_slices_session().
    OPTIONAL: owner, date_from, date_to ("YYYY-MM-DD HH:MM") for filtering.
    """
    try:
        s = _make_session(oidc_session=mod_auth_openidc_session, domain_oidc="duckburg.net.cit.tum.de")
        common_headers = {"Referer": f"{BASE}/calendar"}

        js_resp = s.get(f"{BASE}/files/calendar/js/pos_scheduler.js", headers=common_headers, timeout=15)
        data_url = None
        if js_resp.status_code == 200:
            m = re.search(r'scheduler\.load\s*\(\s*["\']([^"\']+)["\']', js_resp.text)
            if m:
                path     = m.group(1)
                data_url = path if path.startswith("http") else BASE + ("" if path.startswith("/") else "/") + path

        if not data_url:
            return json.dumps({
                "success": False,
                "message": "scheduler.load() not found in pos_scheduler.js",
                "js_status": js_resp.status_code,
                "js_snippet": js_resp.text[:2000],
            }, indent=2)

        r = s.get(data_url, headers={**common_headers, "Accept": "application/json, text/xml, */*",
                                     "X-Requested-With": "XMLHttpRequest"}, timeout=15)

        if r.status_code == 401:
            return json.dumps({"success": False, "status_code": 401, "message": "Unauthorized"}, indent=2)
        if r.status_code != 200:
            return json.dumps({"success": False, "status_code": r.status_code, "data_url": data_url, "response": r.text[:400]}, indent=2)

        try:
            raw    = r.json()
            events = raw.get("data", raw) if isinstance(raw, dict) else raw
        except Exception:
            try:
                import xml.etree.ElementTree as ET
                root   = ET.fromstring(r.text)
                events = []
                for ev in root.findall(".//event"):
                    event       = {child.tag: child.text for child in ev}
                    event["id"] = ev.get("id")
                    events.append(event)
            except Exception:
                return json.dumps({"success": False, "data_url": data_url,
                                   "message": "Response is neither JSON nor XML",
                                   "response": r.text[:400]}, indent=2)

        if isinstance(events, list):
            if owner:
                events = [ev for ev in events if ev.get("owner") == owner]
            if date_from:
                events = [ev for ev in events if ev.get("start_date", "") >= date_from]
            if date_to:
                events = [ev for ev in events if ev.get("start_date", "") <= date_to]

        return json.dumps({"success": True, "data_url": data_url,
                           "count": len(events) if isinstance(events, list) else "?",
                           "events": events}, indent=2)

    except requests.exceptions.RequestException as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
def delete_pos_calendar(
    mod_auth_openidc_session: str,
    event_id: str,
    start_date: str,
    end_date: str,
    owner: str,
    nodes: list,
) -> str:
    """
    Deletes a booking from the POS calendar.
    ALWAYS call list_pos_calendar first to get the real server-assigned event_id.
    """
    payload = {
        "data": {
            str(event_id): {
                "id":                   event_id,
                "start_date":           start_date,
                "end_date":             end_date,
                "owner":                owner,
                "nodes":                nodes,
                "script":               None,
                "repeat":               "",
                "text":                 "",
                "!nativeeditor_status": "deleted",
            }
        }
    }
    try:
        s    = _make_session(oidc_session=mod_auth_openidc_session, domain_oidc="duckburg.net.cit.tum.de")
        resp = s.post(
            f"{BASE}/calendar_update?editing=true", json=payload,
            headers={"Accept": "*/*", "Content-Type": "application/json",
                     "X-Requested-With": "XMLHttpRequest",
                     "Origin": BASE, "Referer": f"{BASE}/calendar"},
            timeout=20,
        )

        if resp.status_code == 401:
            return json.dumps({"success": False, "status_code": 401,
                               "message": "Unauthorized — mod_auth_openidc_session invalid or expired"}, indent=2)

        try:
            body       = resp.json()
            event_data = body.get("data", {}).get(str(event_id), {})
            action     = event_data.get("action", "")
            srv_msg    = event_data.get("message", "")
            success    = resp.status_code in (200, 201) and action == "deleted"
        except Exception:
            action  = ""
            srv_msg = ""
            success = resp.status_code in (200, 201)

        return json.dumps({
            "success":        success,
            "status_code":    resp.status_code,
            "server_action":  action,
            "server_message": srv_msg,
            "event_id":       event_id,
            "owner":          owner,
            "nodes":          nodes,
            "time_period":    f"{start_date} to {end_date}",
            "message":        "Booking deleted successfully" if success else (srv_msg or "Error deleting booking"),
        }, indent=2)

    except requests.exceptions.RequestException as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
def post5g_get_experiment(experiment_name: str) -> str:
    """
    Runs: post5g experiment get <experiment_name> via SSH.
    Returns current status and details of a configured Post-5G experiment.
    Call this after configure_post5g_experiment to verify the configuration.
    """
    sync = _sync_remote_slices_auth()
    if not sync["success"]:
        return json.dumps({
            "success": False,
            "error": f"Failed to sync SLICES auth session to duckburg: {sync['error']}",
        }, indent=2)

    r = _run_remote_cmd(f"post5g experiment get {experiment_name}", timeout=300)
    if r["returncode"] != 0:
        return json.dumps({"success": False, "error": _tail(r["stderr"]), "raw": _tail(r["stdout"])}, indent=2)
    return json.dumps({"success": True, "experiment_name": experiment_name, "output": _tail(r["stdout"])}, indent=2)


@mcp.tool()
def post5g_launch_experiment(experiment_name: str) -> str:
    """
    Runs: post5g experiment launch <experiment_name> via SSH.
    Deploys and starts the Post-5G experiment on the reserved POS nodes.
    Call this ONLY after configure_post5g_experiment and book_pos_calendar succeeded.
    This command can take several minutes to complete.
    """
    sync = _sync_remote_slices_auth()
    if not sync["success"]:
        return json.dumps({
            "success": False,
            "error": f"Failed to sync SLICES auth session to duckburg: {sync['error']}",
        }, indent=2)

    r = _run_remote_cmd(
        f"post5g experiment launch {experiment_name} 2>&1 | grep -v '^-\\|^d\\|^l' | tail -50",
        timeout=800,
    )
    if r["returncode"] != 0:
        return json.dumps({"success": False, "error": r["stderr"], "raw": r["stdout"]}, indent=2)
    return json.dumps({"success": True, "experiment_name": experiment_name, "output": _tail(r["stdout"], lines=30)}, indent=2)

@mcp.tool()
def trigger_5g_anomaly(lb_ip: str, anomaly_type: str = "ddos") -> str:
    """
    Generate anomalous traffic. lb_ip: the 'lb' field from post5g_get_prefix().
    """
    import base64
    
    if anomaly_type == "ddos":
        script_content = f"""
import socket, time, random
rows = []
# Traffico normale simulato (label=0)
for i in range(150):
    rows.append(f"{{time.time()}},{lb_ip},normal,{{random.uniform(10,50):.3f}},0")
    time.sleep(0.01)
# Traffico DDoS reale (label=1)
for i in range(150):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.05)
        s.connect_ex(('{lb_ip}', 80))
        s.close()
    except Exception:
        pass
    rows.append(f"{{time.time()}},{lb_ip},ddos,{{random.uniform(0.1,2):.3f}},1")
    time.sleep(0.005)
with open('anomaly_log.csv', 'w') as f:
    f.write('timestamp,target,action,latency,status\\n')
    for r in rows:
        f.write(r + '\\n')
print(f"Generated {{len(rows)}} rows")
"""
        b64_script = base64.b64encode(script_content.encode('utf-8')).decode('utf-8') 
        
        command = (
            f"echo '{b64_script}' | base64 -d > ddos.py && "
            f"python3 ddos.py && echo done"
        )
    elif anomaly_type == "high_latency":
        command = f"tc qdisc add dev eth0 root netem delay 500ms 2>/dev/null; echo done"
    else:
        command = f"echo 'anomaly,timestamp,type' > anomaly_log.csv && echo '1,$(date -u +%s),packet_loss' >> anomaly_log.csv"

    r = _run_remote_cmd(command)
    if r["returncode"] != 0:
        return json.dumps({"success": False, "error": r["stderr"]}, indent=2)
    
    ssh_user     = os.environ.get("SLICES_USER")
    ssh_key_path = os.path.expanduser(os.environ.get("SSH_KEY_PATH", "~/.ssh/id_rsa"))
    local_path   = "/tmp/anomaly_log.csv"

    scp_cmd = [
        "scp",
        "-P", str(SSH_PORT),
        "-i", ssh_key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes", 
        f"{ssh_user}@{SSH_HOST}:anomaly_log.csv",
        local_path,
    ]
    try:
        r2 = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=180, env=os.environ.copy())
        download_ok    = r2.returncode == 0
        download_error = r2.stderr if not download_ok else None
    except Exception as e:
        download_ok    = False
        download_error = str(e)

    return json.dumps({
        "success":           True,
        "anomaly_type":      anomaly_type,
        "target_ip":         lb_ip,
        "log_file_remote":   "anomaly_log.csv",
        "log_file_local":    local_path if download_ok else None,
        "download_ok":       download_ok,
        "download_error":    download_error,
        "message": (
            f"Anomaly '{anomaly_type}' generated. CSV downloaded to {local_path}"
            if download_ok else
            f"Anomaly generated but CSV download failed: {download_error}"
        ),
    }, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")