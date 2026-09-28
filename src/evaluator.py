"""
模型評估模組
============
計算指標：
  - ROC-AUC
  - F1-Score（閾值 0.5）
  - Precision / Recall
  - Accuracy
  - Brier Score（概率校準）

金融回測：
  - 策略：機率 > 閾值時持有主動基金，否則持有現金 (避險)
  - 輸出：策略年化報酬、Sharpe Ratio、最大回撤
"""

import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 非互動後端，不開視窗
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

matplotlib.rcParams["font.family"] = "Microsoft JhengHei"
matplotlib.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 分類指標
# ──────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "AUC":       roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5,
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Brier":     brier_score_loss(y_true, y_prob),
    }
    return metrics


def evaluate_all(
    model_dict: dict,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    """
    評估所有模型，回傳比較 DataFrame。
    """
    results = {}
    for name, model in model_dict.items():
        proba = model.predict_proba(X_val)
        proba = np.where(np.isnan(proba), 0.5, proba)
        y_arr = y_val.values

        # 對齊 LSTM（可能少了前幾筆）
        if len(proba) != len(y_arr):
            min_len = min(len(proba), len(y_arr))
            proba = proba[-min_len:]
            y_arr = y_arr[-min_len:]

        results[name] = compute_metrics(y_arr, proba)

    df = pd.DataFrame(results).T
    df = df.sort_values("AUC", ascending=False)
    return df


# ──────────────────────────────────────────────
# 金融回測
# ──────────────────────────────────────────────

def backtest(
    model,
    X_val: pd.DataFrame,
    fund_ret: pd.Series,
    threshold: float = 0.5,
    model_name: str = "Model",
    force_entry_drawdown: Optional[float] = None,
) -> pd.DataFrame:
    """
    策略回測：機率 > threshold → 持有主動基金，否則持有現金 (0% 報酬)。

    Returns
    -------
    pd.DataFrame：週頻策略報酬 vs 現金 vs 主動基金
    """
    proba = model.predict_proba(X_val)
    proba = np.where(np.isnan(proba), 0.5, proba)

    # 對齊索引
    val_idx = X_val.index
    fund_r = fund_ret.reindex(val_idx)

    # 處理長度差異（LSTM）
    if len(proba) < len(val_idx):
        proba = np.concatenate([np.full(len(val_idx) - len(proba), 0.5), proba])

    signal = (proba >= threshold).astype(int)
    
    # 強制進場防護網 (深水區抄底)
    if force_entry_drawdown is not None and "max_drawdown_52w" in X_val.columns:
        force_idx = X_val["max_drawdown_52w"] <= force_entry_drawdown
        # 將對應位置改為 1 (pd.Series or np.array index)
        signal = pd.Series(signal, index=val_idx)
        signal[force_idx] = 1
        signal = signal.values
        logger.info(f"[{model_name}] 回測觸發強制進場 (抄底) 共 {force_idx.sum()} 週")

    cash_r = pd.Series(0.0, index=val_idx) # 現金報酬 0%
    strategy_ret = np.where(signal == 1, fund_r, cash_r)

    bt = pd.DataFrame({
        "主動基金報酬": fund_r.values,
        "現金(避險)報酬": cash_r.values,
        f"{model_name}_策略報酬": strategy_ret,
    }, index=val_idx)

    return bt


def compute_backtest_stats(bt: pd.DataFrame, col: str) -> dict:
    ret = bt[col].dropna()
    ann_ret = ret.mean() * 52
    ann_vol = ret.std() * np.sqrt(52)
    sharpe = ann_ret / max(ann_vol, 1e-8)
    cum = (1 + ret).cumprod()
    roll_max = cum.cummax()
    mdd = ((cum - roll_max) / roll_max).min()
    return {
        "年化報酬": f"{ann_ret:.2%}",
        "年化波動": f"{ann_vol:.2%}",
        "Sharpe":   f"{sharpe:.2f}",
        "最大回撤": f"{mdd:.2%}",
    }


# ──────────────────────────────────────────────
# 視覺化
# ──────────────────────────────────────────────

def plot_roc_curves(
    model_dict: dict,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]
    for (name, model), color in zip(model_dict.items(), colors):
        proba = model.predict_proba(X_val)
        proba = np.where(np.isnan(proba), 0.5, proba)
        y_arr = y_val.values
        if len(proba) < len(y_arr):
            proba = np.concatenate([np.full(len(y_arr) - len(proba), 0.5), proba])
        fpr, tpr, _ = roc_curve(y_arr, proba)
        auc = roc_auc_score(y_arr, proba) if len(set(y_arr)) > 1 else 0.5
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC 曲線比較")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"[Plot] ROC 儲存至 {save_path}")
    plt.close()


def plot_cumulative_returns(
    bt_dict: dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
):
    """
    bt_dict：{model_name: backtest_dataframe}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]
    color_idx = 0

    for model_name, bt in bt_dict.items():
        for col in bt.columns:
            cum = (1 + bt[col].dropna()).cumprod()
            ax.plot(cum.index, cum.values, label=col, color=colors[color_idx % len(colors)], lw=2)
            color_idx += 1
        break  # 只畫第一個 bt 的基準（避免重複）

    # 加上其他策略報酬
    for i, (model_name, bt) in enumerate(bt_dict.items()):
        strat_col = [c for c in bt.columns if "策略" in c]
        if strat_col:
            cum = (1 + bt[strat_col[0]].dropna()).cumprod()
            ax.plot(cum.index, cum.values, label=strat_col[0],
                    color=colors[(i + 2) % len(colors)], lw=2, linestyle="--")

    ax.set_title("驗證集累積報酬比較")
    ax.set_ylabel("累積報酬倍數")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"[Plot] 累積報酬儲存至 {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["預測:現金避險", "預測:繼續抱牢"],
        yticklabels=["實際:小於0", "實際:大於0"],
    )
    ax.set_title(f"{model_name} 混淆矩陣")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"[Plot] 混淡矩陣儲存至 {save_path}")
    plt.close()
