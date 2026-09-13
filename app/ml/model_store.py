"""Keep trained model files in the database so they survive a restart.

Training writes files to MODEL_DIR, which is all a long-lived server needs. On
an ephemeral filesystem that directory is gone after every restart, so after
training the files are also published to the model_artifacts table, and
inference restores any file it cannot find locally from there.

Both directions are best-effort: a database problem is logged and never turns
a successful training run, or a prediction that could be served, into an error.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import app.database as database
from app.models import ModelArtifact

logger = logging.getLogger(__name__)


def publish_models(model_dir: str | Path, sport: str) -> list[str]:
    """Copy one sport's model and report files from disk into the database."""
    from app.ml.train import report_filename

    directory = Path(model_dir)
    names = _sport_files(directory, sport)
    report = directory / report_filename(sport)
    if report.exists():
        names.append(report.name)

    published: list[str] = []
    db = database.SessionLocal()
    try:
        for name in names:
            data = (directory / name).read_bytes()
            row = db.get(ModelArtifact, name)
            if row is None:
                db.add(ModelArtifact(filename=name, data=data, size_bytes=len(data)))
            else:
                row.data, row.size_bytes, row.updated_at = data, len(data), datetime.utcnow()
            published.append(name)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Could not publish %s models to the database", sport)
        return []
    finally:
        db.close()
    return published


def _sport_files(directory: Path, sport: str) -> list[str]:
    from app.sports.registry import available_sports

    other_prefixes = tuple(f"{s}_" for s in available_sports() if s != "football")
    files = sorted(p.name for p in directory.glob("*.joblib"))
    if sport == "football":
        return [n for n in files if not n.startswith(other_prefixes)]
    return [n for n in files if n.startswith(f"{sport}_")]


def restore_model(filename: str, model_dir: str | Path) -> bool:
    """Write a stored model file back to disk. True if it is now present."""
    try:
        db = database.SessionLocal()
        try:
            row = db.get(ModelArtifact, filename)
            if row is None:
                return False
            destination = Path(model_dir) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Write then rename, so a concurrent reader never loads half a file.
            partial = destination.with_suffix(destination.suffix + ".partial")
            partial.write_bytes(row.data)
            partial.replace(destination)
        finally:
            db.close()
    except Exception:
        logger.exception("Could not restore %s from the database", filename)
        return False
    logger.info("Restored %s from the database", filename)
    return True
