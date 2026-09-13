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

from app.config import MODEL_DIR  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.ml.model_store import publish_models  # noqa: E402
from app.ml.dataset import load_finished_matches  # noqa: E402
from app.ml.train import train_for_sport  # noqa: E402
from app.services.prediction_service import clear_model_cache  # noqa: E402
from app.sports.registry import available_sports, get_adapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sport", default="football", choices=available_sports(), help="sport to train"
    )
    args = parser.parse_args()
    adapter = get_adapter(args.sport)

    db = SessionLocal()
    try:
        rows = [
            m for m in load_finished_matches(db)
            if m.get("sport", "football") == adapter.name
        ]
        frame = adapter.build_dataset(rows)
        print(f"built {len(frame)} labelled {adapter.name} rows")
        if frame.empty:
            print("Nothing to train on. Backfill history first.")
            return 1
        summary = train_for_sport(adapter, frame)
        clear_model_cache()
        published = publish_models(MODEL_DIR, adapter.name)
        print(f"stored {len(published)} file(s) in the database")
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
