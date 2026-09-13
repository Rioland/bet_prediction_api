#!/usr/bin/env python3
"""Train models from ingested history.

    python scripts/train_model.py
    python scripts/train_model.py --leagues 39,140
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import SessionLocal  # noqa: E402
from app.ml.pipeline import build_training_frame, retrain  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", help="comma-separated internal league ids")
    args = parser.parse_args()
    league_ids = [int(x) for x in args.leagues.split(",")] if args.leagues else None

    db = SessionLocal()
    try:
        frame = build_training_frame(db, league_ids)
        print(f"built {len(frame)} labelled rows")
        if frame.empty:
            print("Nothing to train on. Run scripts/ingest_history.py first.")
            return 1
        summary = retrain(db, league_ids)
    finally:
        db.close()

    print(json.dumps(summary, indent=2))
    for report in summary["targets"]:
        if not report["beats_baseline"]:
            print(f"\nWARNING: '{report['target']}' does not beat predicting base rates. "
                  "It has learned nothing useful - do not ship it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
