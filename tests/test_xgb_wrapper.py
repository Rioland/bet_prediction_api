"""XGBoost with string labels.

XGBoost rejects non-integer targets, and sklearn's clone rejects estimators
whose constructor collects **kwargs. Both bite only at runtime, so both are
asserted here.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from app.ml.train import XGBOOST_AVAILABLE

pytestmark = pytest.mark.skipif(
    not XGBOOST_AVAILABLE, reason="XGBoost needs an OpenMP runtime that is not present"
)


@pytest.fixture()
def signal_data():
    rng = np.random.default_rng(0)
    X = rng.random((400, 5))
    y = np.where(X[:, 0] > 0.66, "H", np.where(X[:, 0] > 0.33, "D", "A"))
    return X, y


def test_raw_xgboost_rejects_string_labels(signal_data) -> None:
    """The reason the wrapper exists."""
    from xgboost import XGBClassifier

    X, y = signal_data
    with pytest.raises(Exception, match="Invalid classes"):
        XGBClassifier(n_estimators=5).fit(X, y)


def test_wrapper_accepts_string_labels(signal_data) -> None:
    from app.ml.xgb import StringLabelXGB

    X, y = signal_data
    model = StringLabelXGB(n_estimators=30, max_depth=3).fit(X, y)

    assert set(model.classes_) == {"A", "D", "H"}
    assert set(model.predict(X[:10])) <= {"A", "D", "H"}


def test_wrapper_is_clonable(signal_data) -> None:
    """CalibratedClassifierCV clones its base estimator on every fold."""
    from app.ml.xgb import StringLabelXGB

    original = StringLabelXGB(n_estimators=20, max_depth=3)
    copy = clone(original)
    assert copy.n_estimators == 20 and copy.max_depth == 3


def test_wrapper_calibrates(signal_data) -> None:
    from app.ml.xgb import StringLabelXGB

    X, y = signal_data
    calibrated = CalibratedClassifierCV(
        StringLabelXGB(n_estimators=40, max_depth=3), method="isotonic", cv=3
    ).fit(X, y)

    proba = calibrated.predict_proba(X[:20])
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(calibrated.predict(X[:20])) <= {"A", "D", "H"}


def test_probability_columns_line_up_with_classes(signal_data) -> None:
    """Callers zip classes_ against proba columns; a mismatch silently swaps outcomes."""
    from app.ml.xgb import StringLabelXGB

    X, y = signal_data
    model = StringLabelXGB(n_estimators=40, max_depth=3).fit(X, y)

    proba = model.predict_proba(X)
    argmax_labels = model.classes_[proba.argmax(axis=1)]
    assert list(argmax_labels) == list(model.predict(X))


def test_xgboost_joins_the_candidate_list() -> None:
    from app.ml.train import _candidates

    assert "xgboost" in _candidates()
