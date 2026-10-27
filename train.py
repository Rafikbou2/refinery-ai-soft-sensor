"""Generate (or load) plant history, train the soft sensor and save it.

    python train.py                       # synthetic 180-day history
    python train.py --file data/my_unit.xlsx   # Excel or CSV with tags + lab columns
"""

import argparse
import json
from pathlib import Path

from softsensor import SoftSensor, generate_history
from softsensor.io import read_operating_file

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "history.csv"
MODEL = ROOT / "models" / "soft_sensor.joblib"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="Excel/CSV historian export with tag + lab columns")
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()

    if args.file:
        path = Path(args.file)
        history, report = read_operating_file(path.read_bytes(), path.name)
        print(f"Read {len(history)} rows from {path} ({report.dropped_rows} skipped)")
        if not report.has_lab:
            raise SystemExit("The file has no lab results (t95_lab / flash_point_lab): "
                             "nothing to train on.")
    else:
        history = generate_history(days=args.days)
        DATA.parent.mkdir(exist_ok=True)
        history.to_csv(DATA)
        print(f"Synthetic history written to {DATA} ({len(history)} rows)")

    sensor = SoftSensor().fit(history)
    sensor.save(MODEL)
    print(f"Model saved to {MODEL}")
    print(json.dumps(sensor.metrics, indent=2))


if __name__ == "__main__":
    main()
