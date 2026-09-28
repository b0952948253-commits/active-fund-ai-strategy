"""
Logistic Regression 模型
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LogisticRegressionModel:
    """Logistic Regression 二元分類器"""

    def __init__(self, cfg: dict):
        mc = cfg["models"]["logistic_regression"]
        self.params = {
            "penalty": mc.get("penalty", "l2"),
            "C": mc.get("C", 0.1),
            "class_weight": mc.get("class_weight", "balanced"),
            "max_iter": mc.get("max_iter", 1000),
            "random_state": mc.get("random_state", 42),
            "solver": "liblinear" if mc.get("penalty") == "l1" else "lbfgs"
        }
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LogisticRegressionModel":
        
        self.feature_names = X_train.columns.tolist()
        
        # Logistic Regression needs scaled features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = LogisticRegression(**self.params)
        self.model.fit(X_train_scaled, y_train)
        
        train_auc = roc_auc_score(y_train, self.model.predict_proba(X_train_scaled)[:, 1])
        logger.info(f"[LogisticRegression] 訓練集 AUC = {train_auc:.4f}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """回傳正例機率"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def feature_importances(self) -> pd.Series:
        # For LR, absolute values of coefficients can be proxy for importance if features are standardized
        imp = np.abs(self.model.coef_[0])
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
