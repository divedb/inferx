#!/usr/bin/env python3
"""Render/check the compiled request transition inventory in lifecycle docs."""

import argparse
import json
import pathlib
import subprocess
import sys


BEGIN = "<!-- BEGIN GENERATED FSM -->"
END = "<!-- END GENERATED FSM -->"


def effect_name(value: int) -> str:
    if value == 0:
        return "none"
    names = []
    if value & 1:
        names.append("release-reservation")
    if value & 2:
        names.append("increment-epoch")
    if value & 4:
        names.append("clear-in-flight")
    return ", ".join(names)


def render(schema: dict) -> str:
    if schema.get("schema_version") != 1 or not isinstance(schema.get("transitions"), list):
        raise ValueError("unsupported FSM schema output")
    lines = [
        BEGIN,
        "",
        "## Generated transition inventory",
        "",
        "Generated from the compiled declarative table; do not edit this section manually.",
        "",
        "| From | Event | To | Effects | Terminal |",
        "|---|---|---|---|---|",
    ]
    for transition in schema["transitions"]:
        destination = "conditional" if transition["conditional"] else transition["to"]
        lines.append(
            f"| `{transition['from']}` | `{transition['event']}` | `{destination}` | "
            f"{effect_name(transition['effects'])} | "
            f"{'yes' if transition['terminal'] else 'no'} |"
        )
    lines.extend(["", END])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    completed = subprocess.run(
        [str(args.binary), "inspect", "--fsm-schema"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        return completed.returncode
    generated = render(json.loads(completed.stdout))
    original = args.output.read_text(encoding="utf-8")
    begin = original.find(BEGIN)
    end = original.find(END, begin)
    if begin < 0 or end < 0:
        raise ValueError(f"generated markers missing from {args.output}")
    expected = original[:begin] + generated + original[end + len(END) :]
    if args.check:
        if original != expected:
            sys.stderr.write(
                f"error: {args.output} is stale; run {pathlib.Path(__file__).name} without --check\n"
            )
            return 1
        print(f"FSM docs current: {len(json.loads(completed.stdout)['transitions'])} transitions")
        return 0
    args.output.write_text(expected, encoding="utf-8")
    print(f"updated {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
