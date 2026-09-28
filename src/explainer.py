"""
SHAP 可解釋性模組
==================
- 特徵重要性圖（Summary Plot）
- 單一預測解釋（Force Plot）
- 特徵依賴圖（Dependence Plot）
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 非互動後端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """XGBoost / RandomForest SHAP 解釋器"""

    def __init__(self, model, X_background: pd.DataFrame, model_type: str = "tree"):
        """
        Parameters
        ----------
        model        : 訓練完成的 XGBoost 或 RF 模型物件（.model 屬性）
        X_background : 背景資料集（通常為訓練集）
        model_type   : "tree"（XGBoost/RF）
        """
        self.feature_names = X_background.columns.tolist()
        self.X_bg = X_background

        if model_type == "tree":
            underlying = getattr(model, "model", model)
            self.explainer = shap.TreeExplainer(underlying)
        else:
            raise NotImplementedError("目前僅支援 tree explainer。")

        logger.info("[SHAP] Explainer 初始化完成")

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        sv = self.explainer.shap_values(X)
        # 二元分類可能回傳 list；取正類
        if isinstance(sv, list):
            sv = sv[1]
        return sv

    def plot_summary(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """特徵重要性 Bar + Beeswarm 圖 (分離為兩張圖)"""
        sv = self.compute_shap_values(X)
        
        # 1. 自訂特徵重要性長條圖 (避免文字重疊)
        mean_abs = np.abs(sv).mean(axis=0)
        imp_df = pd.DataFrame({"Feature": self.feature_names, "Importance": mean_abs})
        imp_df = imp_df.sort_values("Importance", ascending=False).head(10)
        
        import seaborn as sns
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=imp_df, palette="viridis")
        plt.title("特徵重要性 (SHAP Mean |value|)")
        plt.xlabel("Mean |SHAP Value| (Impact on model output)")
        plt.ylabel("")
        plt.tight_layout()
        
        if save_path:
            bar_path = save_path.replace("shap_summary.png", "shap_importance.png")
            plt.savefig(bar_path, dpi=150, bbox_inches="tight")
            logger.info(f"[SHAP] 重要性長條圖已儲存至 {bar_path}")
        plt.close()

        # 2. SHAP 蜂巢分佈圖
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv, X,
            feature_names=self.feature_names,
            max_display=10,
            show=False
        )
        plt.title("SHAP 值分佈 (Beeswarm)")
        plt.tight_layout()
        
        if save_path:
            bee_path = save_path.replace("shap_summary.png", "shap_beeswarm.png")
            plt.savefig(bee_path, dpi=150, bbox_inches="tight")
            logger.info(f"[SHAP] Beeswarm圖已儲存至 {bee_path}")
        plt.close()

    def plot_dependence(
        self, X: pd.DataFrame, feature: str, save_path: Optional[str] = None
    ):
        """指定特徵的 SHAP 依賴圖"""
        sv = self.compute_shap_values(X)
        shap.dependence_plot(
            feature, sv, X,
            feature_names=self.feature_names,
            show=False
        )
        plt.title(f"SHAP 依賴圖：{feature}")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()

    def top_features(self, X: pd.DataFrame, n: int = 10) -> pd.Series:
        """回傳前 N 重要特徵（依平均 |SHAP|）"""
        sv = self.compute_shap_values(X)
        mean_abs = np.abs(sv).mean(axis=0)
        return (
            pd.Series(mean_abs, index=self.feature_names)
            .sort_values(ascending=False)
            .head(n)
        )
