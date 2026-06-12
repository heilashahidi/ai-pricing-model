"""
Validates Haiku scope extraction quality on the 14 priced Handyman rows.
Acceptance bar: labor_only correct on >=12/14 rows.

Usage:
    ANTHROPIC_API_KEY=your_key python3 validate_extraction.py
"""
import csv
import json
import os
import anthropic

EXTRACTION_TOOL = {
    "name": "extract_scope",
    "description": "Extract scope features from a home service job description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labor_only": {
                "type": "boolean",
                "description": "True if the customer supplies all parts/materials (pure labor job)."
            },
            "task_count": {
                "type": "integer",
                "description": "Number of distinct tasks or items in the description."
            },
            "complexity_tier": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "low=single simple task, medium=multiple tasks or moderate skill, high=multi-trade or large scope."
            },
            "has_area_measure": {
                "type": "boolean",
                "description": "True if the description mentions square footage, room count, or linear footage."
            }
        },
        "required": ["labor_only", "task_count", "complexity_tier", "has_area_measure"]
    }
}

def extract_features(client, description):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_scope"},
        messages=[{
            "role": "user",
            "content": f"Extract scope features from this home service job description:\n\n{description}"
        }]
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    return None

def load_handyman_priced():
    rows = []
    with open("houseaccount_pricing_sample.csv") as f:
        for row in csv.DictReader(f):
            if row["service_category"] == "Handyman" and row["final_price"]:
                rows.append(row)
    return rows

def main():
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rows = load_handyman_priced()
    print(f"Validating extraction on {len(rows)} priced Handyman rows\n")
    print(f"{'ACTUAL':>7} {'EST':>7}  {'labor_only':10} {'tasks':5} {'tier':8} {'area':5}  DESCRIPTION")
    print("-" * 100)

    labor_only_correct = 0
    results = []

    for row in rows:
        actual = float(row["final_price"])
        est = float(row["original_estimate"])
        desc = row["job_description"]

        features = extract_features(client, desc)
        if features is None:
            print(f"  EXTRACTION FAILED: {desc[:60]}")
            continue

        # Ground truth: manually labelled based on job type
        # labor_only=True means customer already has parts/materials; job is pure labor
        MANUAL_LABELS = {
            "Install 3":       True,   # "we supply" drawer glides
            "Move plants":     True,   # moving items = pure labor
            "Assemble 1 shel": True,   # customer has shelf/lights/flagpole
            "Diagnostic and":  True,   # diagnostics = pure labor
            "Small drywall":   False,  # contractor supplies drywall compound
            "Assemble two":    True,   # "ordered from Walmart" = customer-supplied
            "Assemble one":    True,   # customer owns furniture
            "Repair aluminum": False,  # screen + rollers = contractor supplies parts
            "Install supplied": True,  # "we supply shutters" explicit
            "Single small":    False,  # drywall patch = contractor supplies
            "Inspect then":    False,  # repair/replace = contractor supplies hardware
            "Front entry":     True,   # "we supply parts" explicit
            "3":               False,  # multi-trade repairs = contractor supplies
            "Minor fence":     False,  # fence panels = contractor supplies
        }
        lo = features["labor_only"]
        expected_labor_only = next(
            (v for k, v in MANUAL_LABELS.items() if desc.startswith(k)), None
        )
        if expected_labor_only is None:
            correct = "??"
        elif lo == expected_labor_only:
            correct = "OK"
            labor_only_correct += 1
        else:
            correct = "WRONG"

        marker = " <-- LABOR-ONLY CATCH" if lo and actual < est * 0.6 else ""
        print(
            f"${actual:>6.0f} ${est:>6.0f}  {str(lo):10} {features['task_count']:5} {features['complexity_tier']:8} "
            f"{str(features['has_area_measure']):5}  [{correct}] {desc[:60]}{marker}"
        )
        results.append({"row": row, "features": features, "labor_only_correct": lo == expected_labor_only})

    print(f"\n--- RESULTS ---")
    print(f"labor_only accuracy: {labor_only_correct}/{len(results)} ({100*labor_only_correct//len(results)}%)")
    print(f"Acceptance bar: >=12/14 (86%)")
    if labor_only_correct >= 12:
        print("PASS — proceed to full extraction on all 1,432 rows")
    else:
        print("FAIL — adjust the extraction prompt before training")
        print("Hint: add examples of 'we supply' vs 'contractor supplies all materials' to the prompt")

if __name__ == "__main__":
    main()
