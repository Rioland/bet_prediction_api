"""XGBoost wrapped for string class labels.

Targets here are strings - "H"/"D"/"A", "over"/"under" - because they are read
by humans in API responses and stored that way on published tips. XGBoost
requires labels to be 0..n-1 integers and raises on anything else, so it cannot
join the candidate list directly.

This encodes on fit and decodes on predict, keeping ``classes_`` in the
original string space.

Constructor arguments are listed explicitly rather than collected with
**kwargs: sklearn's ``clone`` reads parameters off the instance by name and
refuses any estimator whose constructor does not set them, which
CalibratedClassifierCV relies on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class StringLabelXGB(ClassifierMixin, BaseEstimator):
    """XGBClassifier that accepts string targets."""

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        min_child_weight: int = 5,
        tree_method: str = "hist",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.tree_method = tree_method
        self.random_state = random_state

    def _build(self) -> Any:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            tree_method=self.tree_method,
            random_state=self.random_state,
            eval_metric="mlogloss",
        )

    def fit(self, X: Any, y: Any) -> "StringLabelXGB":
        self.encoder_ = LabelEncoder().fit(y)
        self.model_ = self._build()
        self.model_.fit(X, self.encoder_.transform(y))
        return self

    @property
    def classes_(self) -> np.ndarray:
        return self.encoder_.classes_

    def predict(self, X: Any) -> np.ndarray:
        return self.encoder_.inverse_transform(self.model_.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        # Columns follow the encoder's sorted classes, which is the order of
        # classes_ - so callers may zip the two safely.
        return self.model_.predict_proba(X)
