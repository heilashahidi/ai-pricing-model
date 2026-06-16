"""
Batch Haiku scope extraction on all 1,432 rows.
Saves to features.csv. Resumable — skips rows already extracted.

Usage:
    ANTHROPIC_API_KEY=your_key python3 extract_features.py

Cost estimate: ~$0.60 for all 1,432 rows at Haiku pricing.
Runtime: ~5-8 minutes at 10 concurrent calls.
"""
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

import anthropic

INPUT_CSV = "houseaccount_pricing_sample.csv"
OUTPUT_CSV = "features.csv"
CONCURRENCY = 10

EXTRACTION_TOOL = {
    "name": "extract_scope",
    "description": "Extract scope features from a home service job description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labor_only": {
                "type": "boolean",
                "description": "True if the customer supplies all parts/materials (pure labor job). Examples: 'we supply', 'you supply', 'I ordered it', assembly jobs where customer owns the item."
            },
            "task_count": {
                "type": "integer",
                "description": "Number of distinct task types. 'Install shutters + patch wall' = 2. 'Install 3 shutters' = 1 (one task type, multiple units)."
            },
            "unit_count": {
                "type": "integer",
                "description": "Number of physical items/units to work on. 'Install 3 shutters' = 3, 'replace 5 outlets' = 5, 'patch 2 holes' = 2, 'fix faucet' = 1. Counts the items, not the task types."
            },
            "complexity_tier": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "low=single simple task (hang shelf, patch hole), medium=multiple tasks or moderate skill (assemble furniture + mount items), high=multi-trade or large scope (electrical + plumbing + carpentry)."
            },
            "has_area_measure": {
                "type": "boolean",
                "description": "True if square footage, room count, number of windows, linear feet, or similar area/quantity measure is mentioned."
            }
        },
        "required": ["labor_only", "task_count", "unit_count", "complexity_tier", "has_area_measure"]
    }
}

DEFAULTS = {"labor_only": False, "task_count": 1, "unit_count": 1, "complexity_tier": "medium", "has_area_measure": False}
FIELDNAMES = [
    "job_id", "service_category", "service_subtype", "zip_code",
    "booking_month", "estimate_lo", "estimate_hi", "original_estimate",
    "final_price", "deadline",
    "labor_only", "task_count", "unit_count", "complexity_tier", "has_area_measure",
    "extraction_ok"
]


def load_input():
    with open(INPUT_CSV) as f:
        return list(csv.DictReader(f))


def load_done(output_path):
    if not Path(output_path).exists():
        return set()
    with open(output_path) as f:
        return {row["job_id"] for row in csv.DictReader(f)}


async def extract_one(client, sem, row):
    async with sem:
        desc = row["job_description"]
        for attempt in range(2):
            try:
                response = await client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=256,
                    tools=[EXTRACTION_TOOL],
                    tool_choice={"type": "tool", "name": "extract_scope"},
                    messages=[{
                        "role": "user",
                        "content": f"Extract scope features from this home service job description:\n\n{desc}"
                    }]
                )
                for block in response.content:
                    if block.type == "tool_use":
                        return {**row, **block.input, "extraction_ok": "true"}
            except Exception as e:
                if attempt == 0:
                    await asyncio.sleep(2)
                else:
                    print(f"\n  WARN: extraction failed for {row['job_id'][:12]}: {e}", file=sys.stderr)
        return {**row, **DEFAULTS, "extraction_ok": "false"}


async def main():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY not set")

    rows = load_input()
    done = load_done(OUTPUT_CSV)
    remaining = [r for r in rows if r["job_id"] not in done]

    print(f"Total rows: {len(rows)} | Already done: {len(done)} | Remaining: {len(remaining)}")
    if not remaining:
        print("All rows already extracted. See features.csv.")
        return

    client = anthropic.AsyncAnthropic(api_key=key)
    sem = asyncio.Semaphore(CONCURRENCY)
    write_header = not Path(OUTPUT_CSV).exists()

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        completed = 0
        tasks = [extract_one(client, sem, row) for row in remaining]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            writer.writerow(result)
            f.flush()
            completed += 1
            pct = 100 * completed // len(remaining)
            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
            print(f"\r  [{bar}] {completed}/{len(remaining)} ({pct}%)", end="", flush=True)

    print(f"\nDone. Features saved to {OUTPUT_CSV}")

    # Quick summary
    with open(OUTPUT_CSV) as f:
        extracted = list(csv.DictReader(f))
    labor_only_count = sum(1 for r in extracted if r["labor_only"] == "True")
    failed = sum(1 for r in extracted if r["extraction_ok"] == "false")
    tiers = {t: sum(1 for r in extracted if r["complexity_tier"] == t) for t in ["low", "medium", "high"]}
    print(f"\nSummary ({len(extracted)} rows):")
    print(f"  labor_only=True: {labor_only_count} ({100*labor_only_count//len(extracted)}%)")
    print(f"  complexity: low={tiers['low']} medium={tiers['medium']} high={tiers['high']}")
    print(f"  extraction failures (used defaults): {failed}")


if __name__ == "__main__":
    asyncio.run(main())
