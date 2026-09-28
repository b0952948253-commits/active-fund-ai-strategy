"""
XGBoost 模型
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost 二元分類器（擊敗ETF預測）"""

    def __init__(self, cfg: dict):
        mc = cfg["models"]["xgboost"]
        self.params = {
            "objective": "binary:logistic",
            "n_estimators": mc.get("n_estimators", 300),
            "max_depth": mc.get("max_depth", 3),
            "learning_rate": mc.get("learning_rate", 0.05),
            "subsample": mc.get("subsample", 0.8),
            "colsample_bytree": mc.get("colsample_bytree", 0.8),
            "min_child_weight": mc.get("min_child_weight", 3),
            "eval_metric": "auc",
            "random_state": mc.get("random_state", 42),
            "verbosity": 0,
        }
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostModel":
        # 自動調整正負樣本權重
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale = n_neg / max(n_pos, 1)
        self.params["scale_pos_weight"] = scale
        logger.info(f"[XGBoost] scale_pos_weight = {scale:.2f}")

        self.feature_names = X_train.columns.tolist()
        self.model = xgb.XGBClassifier(**self.params)

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        train_auc = roc_auc_score(y_train, self.predict_proba(X_train))
        logger.info(f"[XGBoost] 訓練集 AUC = {train_auc:.4f}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """回傳正例（擊敗ETF）機率"""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def feature_importances(self) -> pd.Series:
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
