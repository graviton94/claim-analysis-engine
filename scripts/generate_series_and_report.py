import os
import json
from pathlib import Path
import pandas as pd

from core.storage import load_partitioned, generate_nested_series
from core.config import DATA_HUB_PATH, DATA_SERIES_PATH


def main():
    hub_path = Path(DATA_HUB_PATH)
    series_path = Path(DATA_SERIES_PATH)
    series_path.mkdir(parents=True, exist_ok=True)

    if not hub_path.exists():
        print(f"Hub path not found: {hub_path}")
        return

    # Load all partitioned data
    try:
        df = load_partitioned(path=hub_path)
    except Exception as e:
        print(f"Failed to load hub data: {e}")
        return

    if df.empty:
        print("No data in hub.")
        return

    # Generate series JSONs
    created = generate_nested_series(df, output_dir=series_path)
    print(f"Generated {created} series files.")

    # Pick a sample file
    sample_file = None
    for p in series_path.glob("*.json"):
        sample_file = p
        break

    report_path = Path("reports/phase1_refactor_nested.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if sample_file is None:
        content = "# Phase 1 Nested Series Report\n\nNo JSON files were generated."
        report_path.write_text(content, encoding="utf-8")
        print("Report written without sample (no series files).")
        return

    # Read sample JSON content
    sample_text = sample_file.read_text(encoding="utf-8")

    # Write report with sample
    content = (
        "# Phase 1 Nested Series Report\n\n"
        "This report includes a sample of the generated Nested Series JSON to verify schema compliance.\n\n"
        f"- Hub: {hub_path}\n"
        f"- Series: {series_path}\n"
        f"- Sample: {sample_file.name}\n\n"
        "## Sample JSON\n\n"
        f"```
{sample_text}
```\n"
    )
    report_path.write_text(content, encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
