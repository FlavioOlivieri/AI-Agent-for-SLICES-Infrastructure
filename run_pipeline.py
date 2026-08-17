from __future__ import annotations

import argparse
import asyncio
import sys
import time

from intent_layer import generate_execution_spec, IntentLayerError
from workflow_executor import WorkflowExecutor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline end-to-end: intent in natural language -> "
                     "ExecutionSpecification -> deterministic execution on MCP tools."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Executes the workflow on real MCP tools without asking for interactive confirmation. "
             "Required in non-interactive mode (stdin is not a tty, so we can't ask for confirmation at runtime).",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()

    intent = (input("Enter the scientific intent: ")
              if sys.stdin.isatty() else sys.stdin.read()).strip()
    if not intent:
        print("[pipeline] Intent is empty, exiting.")
        return

    workflow_id = f"wf-{int(time.time())}"
    print(f"[pipeline] workflow_id = {workflow_id}")

    try:
        spec = await generate_execution_spec(intent, workflow_id)
    except IntentLayerError as e:
        print(f"[pipeline] Impossible to generate a valid spec: {e}")
        return

    print("\n── Spec generated (validated) ──")
    print(spec.model_dump_json(indent=2))

    print("\n── Execution waves ──")
    for i, wave in enumerate(spec.topological_order()):
        print(f"  wave {i}: {wave}")

    if args.execute:
        proceed = True
    else:
        try:
            answer = input("\nExecute this workflow on real MCP tools? [y/N] ")
            proceed = answer.strip().lower() == "y"
        except EOFError:
            print(
                "\n[pipeline] No stdin available for confirmation (EOF). "
                "Rerun with --execute if you want to execute this workflow."
            )
            proceed = False

    if not proceed:
        print("[pipeline] Execution not started. The spec above remains valid "
              "and you can rerun it by passing it to WorkflowExecutor.")
        return

    print("\n[pipeline] Starting execution on real MCP tools...")
    executor = WorkflowExecutor(spec)
    results = await executor.run()

    print("\n── Execution Summary ──")
    for task_id, result in results.items():
        icon = "✓" if result.status.value == "succeeded" else "✗"
        line = f"{icon} {task_id}: {result.status.value}"
        if result.error:
            line += f"  ← {result.error}"
        if result.output.get("portal_url"):
            line += f"  → {result.output['portal_url']}"
        print(line)

        # For tasks with no portal_url/error to show inline (e.g. search_digital_objects,
        # get_digital_object), print the actual output — otherwise it's invisible even
        # though it's already sitting in result.output.
        if not result.error and not result.output.get("portal_url") and result.output:
            import json as _json
            printable = {k: v for k, v in result.output.items() if k != "success"}
            if printable:
                print(f"    output: {_json.dumps(printable, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())