#!/usr/bin/env python3
"""Walk-forward backtest with a flat-stake value simulation.

    python scripts/backtest.py --folds 5 --edge 0.05
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal  # noqa: E402
from app.ml.backtest import calibration_table, value_simulation, walk_forward  # noqa: E402
from app.ml.dataset import load_finished_matches  # noqa: E402
from app.sports.registry import available_sports, get_adapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--edge", type=float, default=0.05)
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--sport", default="football", choices=available_sports())
    args = parser.parse_args()
    adapter = get_adapter(args.sport)

    db = SessionLocal()
    try:
        rows = [
            m for m in load_finished_matches(db)
            if m.get("sport", "football") == adapter.name
        ]
        frame = adapter.build_dataset(rows)
    finally:
        db.close()

    if frame.empty:
        print("No data. Run scripts/ingest_history.py first.")
        return 1

    result = walk_forward(
        frame, folds=args.folds, min_train=args.min_train,
        feature_columns=adapter.feature_columns,
    )
    predictions = result.pop("predictions")
    print(json.dumps(result, indent=2))

    print("\n--- calibration (home win) ---")
    print(calibration_table(predictions).to_string(index=False))

    print("\n--- flat-stake value simulation ---")
    print(json.dumps(value_simulation(predictions, edge_threshold=args.edge), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
