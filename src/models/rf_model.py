"""
Random Forest 模型
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class RandomForestModel:
    """Random Forest 二元分類器"""

    def __init__(self, cfg: dict):
        mc = cfg["models"]["random_forest"]
        self.params = {
            "n_estimators": mc.get("n_estimators", 300),
            "max_depth": mc.get("max_depth", 5),
            "min_samples_leaf": mc.get("min_samples_leaf", 3),
            "max_features": mc.get("max_features", "sqrt"),
            "class_weight": mc.get("class_weight", "balanced"),
            "oob_score": mc.get("oob_score", True),
            "n_jobs": mc.get("n_jobs", -1),
            "random_state": mc.get("random_state", 42),
        }
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "RandomForestModel":
        self.feature_names = X_train.columns.tolist()
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)

        train_auc = roc_auc_score(y_train, self.predict_proba(X_train))
        oob = self.model.oob_score_ if self.params.get("oob_score") else "N/A"
        logger.info(
            f"[RandomForest] 訓練集 AUC = {train_auc:.4f}  │  OOB Score = {oob}"
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def feature_importances(self) -> pd.Series:
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
