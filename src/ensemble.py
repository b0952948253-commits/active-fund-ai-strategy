"""
集成模型
========
支援兩種策略：
  1. soft_vote  — 加權機率平均
  2. stacking   — 以 Logistic Regression 作為元學習器
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class EnsembleModel:
    """三模型集成"""

    def __init__(
        self,
        models: dict,
        weights: Optional[dict] = None,
        method: str = "soft_vote",
    ):
        """
        Parameters
        ----------
        models  : {"xgboost": model, "random_forest": model, "lstm": model}
        weights : {"xgboost": 0.4, ...}  僅在 soft_vote 時使用
        method  : "soft_vote" | "stacking"
        """
        self.models = models
        self.method = method
        self.weights = weights or {k: 1 / len(models) for k in models}
        self.meta_model: Optional[LogisticRegression] = None

    def _get_base_probs(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """從各基模型取得預測機率"""
        probs = {}
        for name, model in self.models.items():
            p = model.predict_proba(X)
            # LSTM 可能有 NaN（前幾筆），替換為 0.5
            p = np.where(np.isnan(p), 0.5, p)
            probs[name] = p
        return probs

    def fit_stacking(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "EnsembleModel":
        """在驗證集上訓練 stacking 元學習器"""
        if self.method != "stacking":
            return self
        probs = self._get_base_probs(X_val)
        meta_X = np.column_stack(list(probs.values()))
        self.meta_model = LogisticRegression(C=0.1, random_state=42)
        self.meta_model.fit(meta_X, y_val.values)
        logger.info("[Ensemble] Stacking 元學習器訓練完成")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = self._get_base_probs(X)

        if self.method == "soft_vote":
            weighted = sum(
                probs[name] * self.weights.get(name, 0)
                for name in probs
            )
            total_w = sum(self.weights.get(k, 0) for k in probs)
            return weighted / max(total_w, 1e-8)

        elif self.method == "stacking":
            if self.meta_model is None:
                raise RuntimeError("請先執行 fit_stacking() 訓練元學習器。")
            meta_X = np.column_stack(list(probs.values()))
            return self.meta_model.predict_proba(meta_X)[:, 1]

        else:
            raise ValueError(f"未知集成方法：{self.method}")

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
