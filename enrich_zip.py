"""
Enriches features.csv with geographic income signal by ZIP code.
Uses pgeocode for state lookup + 2023 ACS state median household incomes.
Saves to features_enriched.csv.

Usage:
    python3 enrich_zip.py
"""
import csv
import sys
import warnings
warnings.filterwarnings("ignore")

import pgeocode

INPUT_CSV  = "features.csv"
OUTPUT_CSV = "features_enriched.csv"

# 2023 ACS 1-year estimates, median household income by state
# Source: census.gov/library/publications/2024/demo/p60-282.html
STATE_MEDIAN_INCOME = {
    "Maryland":             101_710,
    "New Jersey":           101_050,
    "Massachusetts":         98_700,
    "Hawaii":                97_600,
    "Connecticut":           91_400,
    "California":            91_100,
    "Washington":            90_300,
    "Colorado":              87_900,
    "Virginia":              87_600,
    "Minnesota":             85_700,
    "New Hampshire":         84_900,
    "Utah":                  84_800,
    "Alaska":                83_000,
    "New York":              80_900,
    "Illinois":              78_000,
    "Oregon":                77_100,
    "Rhode Island":          76_400,
    "Delaware":              76_200,
    "Wisconsin":             74_500,
    "Georgia":               73_600,
    "Arizona":               72_600,
    "Nevada":                72_400,
    "Texas":                 72_300,
    "North Carolina":        71_500,
    "Florida":               70_900,
    "Pennsylvania":          70_600,
    "Michigan":              70_300,
    "Idaho":                 70_100,
    "Ohio":                  68_700,
    "Indiana":               68_400,
    "Tennessee":             66_600,
    "Montana":               66_000,
    "South Carolina":        65_300,
    "Vermont":               65_100,
    "Iowa":                  65_100,
    "Nebraska":              65_000,
    "Missouri":              64_800,
    "Kansas":                64_600,
    "Wyoming":               64_200,
    "North Dakota":          63_400,
    "South Dakota":          62_900,
    "Maine":                 62_900,
    "Oklahoma":              62_200,
    "Alabama":               59_600,
    "Kentucky":              59_300,
    "Louisiana":             58_300,
    "New Mexico":            58_200,
    "Arkansas":              57_600,
    "West Virginia":         55_900,
    "Mississippi":           53_600,
    "District of Columbia": 101_700,
    "Puerto Rico":           24_000,
    "US":                    75_149,  # national median fallback
}


def main():
    with open(INPUT_CSV) as f:
        rows = list(csv.DictReader(f))

    unique_zips = sorted({r["zip_code"] for r in rows})
    print(f"Loaded {len(rows)} rows | {len(unique_zips)} unique ZIPs")

    # Build ZIP → state map using pgeocode
    print("Looking up state for each ZIP via pgeocode...")
    nomi = pgeocode.Nominatim("us")
    zip_to_state = {}
    zip_to_latlon = {}

    batch = nomi.query_postal_code(unique_zips)
    for zip_code, (_, row) in zip(unique_zips, batch.iterrows()):
        state = row.get("state_name") if hasattr(row, "get") else row["state_name"]
        lat   = row.get("latitude")   if hasattr(row, "get") else row["latitude"]
        lon   = row.get("longitude")  if hasattr(row, "get") else row["longitude"]
        zip_to_state[zip_code]  = state if str(state) != "nan" else None
        zip_to_latlon[zip_code] = (
            round(float(lat), 4) if str(lat) != "nan" else None,
            round(float(lon), 4) if str(lon) != "nan" else None,
        )

    # Assign income
    direct = state_miss = national = 0
    enriched = []
    for row in rows:
        z     = row["zip_code"]
        state = zip_to_state.get(z)
        lat, lon = zip_to_latlon.get(z, (None, None))

        if state and state in STATE_MEDIAN_INCOME:
            income     = STATE_MEDIAN_INCOME[state]
            match_type = "state"
            direct    += 1
        elif state:
            income     = STATE_MEDIAN_INCOME["US"]
            match_type = "national"
            state_miss += 1
        else:
            income     = STATE_MEDIAN_INCOME["US"]
            match_type = "national"
            national  += 1

        enriched.append({
            **row,
            "state":              state or "",
            "latitude":           lat   or "",
            "longitude":          lon   or "",
            "state_median_income": income,
            "income_match_type":  match_type,
        })

    fieldnames = list(rows[0].keys()) + [
        "state", "latitude", "longitude",
        "state_median_income", "income_match_type",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    match_rate = 100 * direct // len(unique_zips)
    print(f"\n--- Results ---")
    print(f"  State matched:       {direct}/{len(unique_zips)} unique ZIPs ({match_rate}%)")
    print(f"  State name not in table: {state_miss}")
    print(f"  No state found:      {national}")
    print(f"\nSaved {len(enriched)} rows → {OUTPUT_CSV}")

    # Income distribution
    incomes = sorted(int(r["state_median_income"]) for r in enriched)
    n = len(incomes)
    print(f"\nState income distribution across rows:")
    print(f"  Min:    ${min(incomes):,}")
    print(f"  Median: ${incomes[n//2]:,}")
    print(f"  Max:    ${max(incomes):,}")

    # Unique states represented
    states = sorted({r["state"] for r in enriched if r["state"]})
    print(f"\n{len(states)} states in dataset: {', '.join(states[:10])}{'...' if len(states) > 10 else ''}")


if __name__ == "__main__":
    main()
