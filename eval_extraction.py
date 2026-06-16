"""
Extraction eval harness — tests Haiku scope extraction against a golden set.

Calls the real Haiku API on each example in extraction_golden.json and checks
per-field accuracy against hand-labeled expected values.

Acceptance bars (exit 1 if missed):
  labor_only      >= 90%  exact match   (most load-bearing feature)
  has_area_measure>= 90%  exact match
  task_count      >= 80%  within ±1
  complexity_tier >= 80%  exact OR adjacent tier (low↔medium, medium↔high)

Usage:
  ANTHROPIC_API_KEY=your_key python3 eval_extraction.py
  ANTHROPIC_API_KEY=your_key python3 eval_extraction.py --verbose
"""
import argparse
import json
import os
import sys
from pathlib import Path

import anthropic

GOLDEN_FILE = Path(__file__).parent / "extraction_golden.json"

EXTRACTION_TOOL = {
    "name": "extract_scope",
    "description": "Extract scope features from a home service job description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labor_only": {
                "type": "boolean",
                "description": "True if the customer supplies all parts/materials (pure labor job).",
            },
            "task_count": {
                "type": "integer",
                "description": "Number of distinct tasks or items in the description.",
            },
            "unit_count": {
                "type": "integer",
                "description": "Number of physical units/items if countable (e.g. 3 shutters, 2 fans). Defaults to 1.",
            },
            "complexity_tier": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "low=single simple task, medium=multiple tasks or moderate skill, high=multi-trade or large scope.",
            },
            "has_area_measure": {
                "type": "boolean",
                "description": "True if the description mentions square footage, acreage, room count, or linear footage.",
            },
        },
        "required": ["labor_only", "task_count", "unit_count", "complexity_tier", "has_area_measure"],
    },
}

ACCEPTANCE_BARS = {
    "labor_only":       0.90,
    "has_area_measure": 0.90,
    "task_count":       0.80,
    "complexity_tier":  0.80,
}

TIER_ORDER = ["low", "medium", "high"]

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def extract(client: anthropic.Anthropic, description: str) -> dict:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_scope"},
        messages=[{"role": "user", "content": f"Extract scope features:\n\n{description}"}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    return {}


def complexity_adjacent(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    ai = TIER_ORDER.index(actual) if actual in TIER_ORDER else -1
    ei = TIER_ORDER.index(expected) if expected in TIER_ORDER else -1
    return abs(ai - ei) <= 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    golden = json.loads(GOLDEN_FILE.read_text())
    client = anthropic.Anthropic(api_key=api_key)

    n = len(golden)
    correct = {"labor_only": 0, "has_area_measure": 0, "task_count": 0, "complexity_tier": 0}
    failures = []

    print(f"\nRunning extraction eval on {n} examples...\n")

    for i, example in enumerate(golden, 1):
        eid  = example["id"]
        desc = example["description"]
        exp  = example["expected"]

        try:
            got = extract(client, desc)
        except Exception as e:
            print(f"  [{FAIL}] {eid} — API error: {e}")
            failures.append({"id": eid, "error": str(e)})
            continue

        row_ok = True
        row_notes = []

        # labor_only — exact
        if got.get("labor_only") == exp["labor_only"]:
            correct["labor_only"] += 1
        else:
            row_ok = False
            row_notes.append(f"labor_only: got {got.get('labor_only')} expected {exp['labor_only']}")

        # has_area_measure — exact
        if got.get("has_area_measure") == exp["has_area_measure"]:
            correct["has_area_measure"] += 1
        else:
            row_ok = False
            row_notes.append(f"has_area_measure: got {got.get('has_area_measure')} expected {exp['has_area_measure']}")

        # task_count — within ±1
        got_tc = got.get("task_count", 0)
        exp_tc = exp["task_count"]
        if abs(got_tc - exp_tc) <= 1:
            correct["task_count"] += 1
        else:
            row_ok = False
            row_notes.append(f"task_count: got {got_tc} expected {exp_tc} (±1 tolerance)")

        # complexity_tier — exact or adjacent
        got_ct = got.get("complexity_tier", "")
        exp_ct = exp["complexity_tier"]
        if complexity_adjacent(got_ct, exp_ct):
            correct["complexity_tier"] += 1
        else:
            row_ok = False
            row_notes.append(f"complexity_tier: got {got_ct!r} expected {exp_ct!r}")

        if args.verbose or not row_ok:
            status = PASS if row_ok else FAIL
            print(f"  [{status}] {eid}")
            if row_notes:
                for note in row_notes:
                    print(f"           {note}")
        else:
            print(f"  [{PASS}] {eid}")

        if not row_ok:
            failures.append({"id": eid, "notes": row_notes, "got": got, "expected": exp})

    # ── Per-field report ──────────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print(f"{'Field':<20} {'Correct':>7} {'Total':>6} {'Accuracy':>9}  {'Bar':>6}  Result")
    print(f"{'─'*52}")

    all_pass = True
    for field, bar in ACCEPTANCE_BARS.items():
        acc = correct[field] / n
        passed = acc >= bar
        if not passed:
            all_pass = False
        status = PASS if passed else FAIL
        print(f"  {field:<18} {correct[field]:>7}/{n:<5}  {acc*100:>6.1f}%  {bar*100:>5.0f}%  [{status}]")

    print(f"{'─'*52}")
    print(f"\n  {n - len(failures)}/{n} examples fully correct\n")

    if not all_pass:
        print(f"[{FAIL}] One or more fields missed acceptance bar — extraction quality degraded\n")
        sys.exit(1)
    else:
        print(f"[{PASS}] All fields meet acceptance bars\n")


if __name__ == "__main__":
    main()
