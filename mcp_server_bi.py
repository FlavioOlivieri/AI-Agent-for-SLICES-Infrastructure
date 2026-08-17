import subprocess
import json
import re
import os
from mcp.server.fastmcp import FastMCP
import time
from dotenv import load_dotenv

mcp = FastMCP("bi")
load_dotenv()

SSH_KEY_PATH = os.environ.get("SSH_KEY_PATH", os.path.expanduser("~/.ssh/id_rsa"))
SSH_USER     = os.environ.get("BI_VM_USER", "ubuntu")  
INFRA_ID     = os.environ.get("SLICES_BI_INFRA_ID",
               os.environ.get("SLICES_BI_SITE_ID", "be-gent1-bi-vm1"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_cmd(args: list, timeout: int = 300) -> dict:
    """Execute a local SLICES CLI command."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"returncode": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": f"Command timed out after {timeout}s"}


def _run_ssh(vm_ip: str, command: str, timeout: int = 120) -> dict:
    """Execute a command on the BI VM via SSH."""
    ssh_cmd = [
        "ssh",
        "-i", SSH_KEY_PATH,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=10",
        f"{SSH_USER}@{vm_ip}",
        command,
    ]
    try:
        r = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return {"returncode": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": f"SSH command timed out after {timeout}s"}
    except Exception as e:
        return {"returncode": -1, "stdout": "", "stderr": str(e)}

# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def bi_list_infra() -> str:
    """
    List available SLICES BI site IDs and flavors (compact output).
    Call this before bi_create_mlops_vm to get a valid site_id and flavor.
    """
    # Flavors: extract only name column
    r1 = _run_cmd(["slices", "bi", "flavor", "list"])
    flavors = []
    for line in r1["stdout"].splitlines():
        # Prende solo le righe con dati (contengono m1.)
        if "m1." in line or "baremetal" in line.lower():
            parts = line.split()
            if parts:
                flavors.append(parts[0].strip("│").strip())

    # Site IDs: estrai dal primo token dell'output
    r2 = _run_cmd(["slices", "bi", "diskimage", "list"])
    site_ids = set()
    for line in r2["stdout"].splitlines():
        if "be-" in line or "nl-" in line or "gr-" in line:
            parts = line.split()
            for p in parts:
                p = p.strip("│").strip()
                if p.startswith(("be-", "nl-", "gr-")):
                    site_ids.add(p)

    return json.dumps({
        "success":  True,
        "site_ids": ["be-gent1-bi-vm1", "be-gent1-bi-baremetal1"],
        "flavors":  ["m1.tiny", "m1.small", "m1.medium", "m1.large"],
        "default_image": "Ubuntu 22.04.5",
        "hint": "Use site_id='be-gent1-bi-vm1' and flavor='m1.small' unless specified otherwise.",
    }, indent=2)


@mcp.tool()
def bi_get_cli_help() -> str:
    """
    Get the CLI help for 'slices bi create' and 'slices bi'.
    Call this if bi_create_mlops_vm fails to check available options and correct syntax.
    """
    r  = _run_cmd(["slices", "bi", "create", "--help"])
    r2 = _run_cmd(["slices", "bi", "--help"])
    return json.dumps({
        "create_help": r["stdout"]  or r["stderr"],
        "bi_help":     r2["stdout"] or r2["stderr"],
    }, indent=2)


@mcp.tool()
def bi_create_mlops_vm(
    experiment_name: str,
    site_id: str = "be-gent1-bi-vm1",
    vm_name: str = "mlops-server",
    image: str = "Ubuntu 22.04.5",
    flavor: str = "m1.small",
    duration: str = "4h",
) -> str:
    """Create a VM on the SLICES Basic Infrastructure."""
    env = os.environ.copy()
    env["SLICES_BI_INFRA_ID"] = site_id          

    cmd = [
        "slices", "bi", "create",
        "--experiment", experiment_name,
        "--duration",   duration,
        "--image",      image,
        "--flavor",     flavor,
        "--ssh-key-file", SSH_KEY_PATH + ".pub",
        "--wait",
        vm_name,                                 
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        result_text = r.stdout.strip() + r.stderr.strip()
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Command timed out after 600s"}, indent=2)

    if r.returncode != 0:
        if "already exists" in r.stderr:
            return json.dumps({
                "success": True,
                "vm_name": vm_name,
                "message": "VM already exists, reusing it. Call bi_get_vm_ip next.",
            }, indent=2)
        return json.dumps({"success": False, "error": r.stderr.strip()}, indent=2)
    
    return json.dumps({
        "success": True, 
        "vm_name": vm_name, 
        "message": "VM created successfully.",
        "raw_output": result_text
    }, indent=2)


@mcp.tool()
def bi_wait_vm_ready(experiment_name: str, vm_name: str = "mlops-server") -> str:
    """Wait until the VM is reachable via slices bi ssh."""
    MAX_RETRIES = 8
    time.sleep(20)  

    for attempt in range(MAX_RETRIES):
        r = _run_cmd([
            "slices", "bi", "ssh",
            "--experiment", experiment_name,
            vm_name, "echo ready"
        ], timeout=30)

        if r["returncode"] == 0 and "ready" in r["stdout"]:
            return json.dumps({"success": True, "message": "VM is reachable."}, indent=2)

        time.sleep(15)

    return json.dumps({"success": False, "error": "VM did not become reachable in time."}, indent=2)

@mcp.tool()
def bi_get_vm_ip(experiment_name: str, vm_name: str = "mlops-server") -> str:
    """
    Returns the private IP of a BI VM by parsing 'slices bi ssh' output.
    Call this after bi_wait_vm_ready to get the IP before SSH commands.
    """
    r = _run_cmd([
        "slices", "bi", "ssh",
        "--experiment", experiment_name,
        vm_name, "echo", "ok"
    ], timeout=30)

    first_line = (r["stdout"] or r["stderr"]).splitlines()[0] if (r["stdout"] or r["stderr"]) else ""
    import re
    m = re.search(r'(\d{1,3}(?:\.\d{1,3}){3})', first_line)
    if not m:
        return json.dumps({"success": False, "error": f"Could not parse IP from: {first_line!r}"}, indent=2)

    return json.dumps({
        "success":    True,
        "vm_name":    vm_name,
        "private_ip": m.group(1),
        "hint":       "Use private_ip only for direct SSH. For tunnels, pass experiment_name+vm_name to bi_open_tunnels.",
    }, indent=2)


@mcp.tool()
def bi_transfer_file_from_post5g(
    experiment_name: str,
    local_file: str = "/tmp/anomaly_log.csv",
    vm_name: str = "mlops-server",
    vm_dest: str = "/home/ubuntu/",
) -> str:
    """
    Upload a local file (previously downloaded from duckburg) to the BI VM
    using slices bi scp.
    """
    cmd = [
        "slices", "bi", "scp",
        "--experiment", experiment_name,
        local_file,
        f"{vm_name}:{vm_dest}",
    ]
    
    MAX_RETRIES = 6
    last_error = ""
    
    for attempt in range(MAX_RETRIES):
        r = _run_cmd(cmd, timeout=60)

        if r["returncode"] == 0:
            return json.dumps({
                "success":     True,
                "source":      local_file,
                "destination": f"{vm_name}:{vm_dest}",
                "message":     f"File uploaded to BI VM successfully after {attempt + 1} attempts.",
            }, indent=2)

        last_error = (r["stderr"] + " " + r["stdout"]).strip()
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(15)
            
    return json.dumps({"success": False, "error": last_error}, indent=2)

@mcp.tool()
def bi_run_command(
    experiment_name: str,
    vm_name: str,
    command: str,
    timeout: int = 120,
) -> str:
    """Execute a shell command on the BI VM via slices bi ssh."""
    cmd = [
        "slices", "bi", "ssh",
        "--experiment", experiment_name,
        vm_name,
        "--", command,          # passa il comando direttamente
    ]
    r = _run_cmd(cmd, timeout=timeout)
    return json.dumps({
        "success":    r["returncode"] == 0,
        "command":    command,
        "stdout":     r["stdout"],
        "stderr":     r["stderr"] if r["returncode"] != 0 else "",
        "returncode": r["returncode"],
    }, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")