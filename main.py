# -*- coding: utf-8 -*-
"""
主流程入口 (Walk-Forward Validation 版)
==========
執行方式：
  python main.py
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# ── 路徑設定 ──
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_all_data
from src.feature_engineer import compute_features
from src.labeler import generate_labels, describe_label_distribution
from src.models.xgboost_model import XGBoostModel
from src.models.rf_model import RandomForestModel
from src.models.lr_model import LogisticRegressionModel
from src.ensemble import EnsembleModel
from src.evaluator import (
    evaluate_all,
    backtest,
    compute_backtest_stats,
    plot_roc_curves,
    plot_cumulative_returns,
    plot_confusion_matrix,
)
from src.explainer import SHAPExplainer

# ── 日誌設定 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "results" / "run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_dirs(cfg: dict):
    Path(cfg["output"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["figures_dir"]).mkdir(parents=True, exist_ok=True)


def prepare_dataset(cfg: dict):
    """載入資料 → 特徵 → 標籤 → 合併 → 清洗"""
    etf_nav, fund_nav, macro_df = load_all_data(cfg)

    features = compute_features(
        fund_nav=fund_nav,
        etf_nav=etf_nav,
        macro_df=macro_df,
        rf_annual=cfg["risk_free_rate"]["annual"],
        windows=cfg["features"]["rolling_windows"],
    )

    labels = generate_labels(
        fund_nav=fund_nav,
        etf_nav=etf_nav,
        forward_window=cfg["features"]["forward_window"],
        stop_loss_threshold=cfg["features"].get("stop_loss_threshold", -0.08),
        force_entry_drawdown=cfg["features"].get("force_entry_drawdown", -0.15),
    )

    df = features.join(labels, how="inner").dropna(subset=["label"])

    feature_cols = [c for c in df.columns if c != "label"]
    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    return df, etf_nav, fund_nav, macro_df


def main():
    print("=" * 60)
    print("  [系統] 主動基金 vs. ETF 超額報酬預測系統 (Walk-Forward版)")
    print("  [基金] 安聯台灣大壩基金-A類型-新臺幣")
    print("  [基準] 元大台灣50 ETF (0050.TW)")
    print("=" * 60)

    cfg = load_config("config.yaml")
    make_dirs(cfg)
    figs = cfg["output"]["figures_dir"]

    print("\n[步驟 1/5] 載入資料與特徵工程...")
    df, etf_nav, fund_nav, macro_df = prepare_dataset(cfg)
    
    if len(df) < 50:
        logger.error(f"資料筆數不足 ({len(df)})，請確認資料來源。")
        sys.exit(1)
        
    print(f"總資料筆數: {len(df)} 筆 ({df.index.min().date()} ~ {df.index.max().date()})")

    # 準備 TimeSeriesSplit
    n_splits = cfg.get("cv", {}).get("n_splits", 4)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print(f"\n[步驟 2/5] 模型訓練與 Walk-Forward 交叉驗證 ({n_splits} Splits)...")
    
    feature_cols = [c for c in df.columns if c != "label"]
    
    # 收集 OOS (Out-Of-Sample) 預測機率與真實標籤
    oos_predictions = {
        "LogisticRegression": [],
        "XGBoost": [],
        "RandomForest": [],
        "Ensemble": []
    }
    oos_indices = []
    oos_y_true = []
    
    last_models = {}

    for split_idx, (train_idx, val_idx) in enumerate(tscv.split(df)):
        print(f"\n--- Split {split_idx+1}/{n_splits} ---")
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]
        
        X_train, y_train = train[feature_cols], train["label"].astype(int)
        X_val, y_val = val[feature_cols], val["label"].astype(int)
        
        print(f"Train: {train.index.min().date()} ~ {train.index.max().date()} ({len(X_train)}筆)")
        print(f"Val:   {val.index.min().date()} ~ {val.index.max().date()} ({len(X_val)}筆)")
        
        # 訓練模型
        lr_model = LogisticRegressionModel(cfg).fit(X_train, y_train, X_val, y_val)
        xgb_model = XGBoostModel(cfg).fit(X_train, y_train, X_val, y_val)
        rf_model = RandomForestModel(cfg).fit(X_train, y_train, X_val, y_val)
        
        # 集成
        ens_cfg = cfg["ensemble"]
        ensemble = EnsembleModel(
            models={"logistic_regression": lr_model, "xgboost": xgb_model, "random_forest": rf_model},
            weights=ens_cfg.get("weights", {}),
            method=ens_cfg.get("method", "soft_vote"),
        )
        
        oos_indices.extend(val.index)
        oos_y_true.extend(y_val)
        
        oos_predictions["LogisticRegression"].extend(lr_model.predict_proba(X_val))
        oos_predictions["XGBoost"].extend(xgb_model.predict_proba(X_val))
        oos_predictions["RandomForest"].extend(rf_model.predict_proba(X_val))
        oos_predictions["Ensemble"].extend(ensemble.predict_proba(X_val))
        
        if split_idx == n_splits - 1:
            last_models = {
                "LogisticRegression": lr_model,
                "XGBoost": xgb_model,
                "RandomForest": rf_model,
                "Ensemble": ensemble
            }

    oos_y_true = pd.Series(oos_y_true, index=oos_indices)
    
    # Dummy Model 以相容 evaluator
    class OOSDummyModel:
        def __init__(self, probas):
            self.probas = np.array(probas)
        def predict_proba(self, X):
            return self.probas
        def predict(self, X, threshold=0.5):
            return (self.probas >= threshold).astype(int)

    oos_models = {name: OOSDummyModel(probs) for name, probs in oos_predictions.items()}
    X_val_combined = pd.DataFrame(index=oos_indices)

    print("\n[步驟 4/5] 彙總 Walk-Forward 樣本外評估...")
    results_df = evaluate_all(oos_models, X_val_combined, oos_y_true)
    print("\n" + "-" * 50)
    print("  Walk-Forward 總驗證集評估結果 (Out-Of-Sample)")
    print("-" * 50)
    print(results_df.to_string())
    results_df.to_csv(f"{cfg['output']['results_dir']}/metrics.csv")

    plot_roc_curves(oos_models, X_val_combined, oos_y_true, save_path=f"{figs}/roc_curves.png")

    best_model_name = results_df["AUC"].idxmax()
    best_p = oos_models[best_model_name].predict_proba(X_val_combined)
    plot_confusion_matrix(oos_y_true.values, best_p, best_model_name, f"{figs}/confusion_{best_model_name}.png")

    print("\n[步驟 5/5] 金融回測 (串聯所有 OOS 期間)...")
    fund_ret = fund_nav.pct_change().dropna()

    bt_results = {}
    for name, model in oos_models.items():
        bt = backtest(model, X_val_combined, fund_ret, model_name=name, force_entry_drawdown=cfg["features"].get("force_entry_drawdown", None))
        bt_results[name] = bt

        strat_col = [c for c in bt.columns if "策略" in c]
        if strat_col:
            stats = compute_backtest_stats(bt, strat_col[0])
            print(f"\n  [{name}] {strat_col[0]}")
            for k, v in stats.items():
                print(f"    {k}: {v}")

    first_bt = list(bt_results.values())[0]
    fund_stats = compute_backtest_stats(first_bt, "主動基金報酬")
    cash_stats = compute_backtest_stats(first_bt, "現金(避險)報酬")
    print(f"\n  [Buy-Hold] 安聯台灣大壩基金")
    for k, v in fund_stats.items():
        print(f"    {k}: {v}")
    print(f"\n  [避險資產] 現金 (0% 報酬)")
    for k, v in cash_stats.items():
        print(f"    {k}: {v}")

    plot_cumulative_returns(bt_results, save_path=f"{figs}/cumulative_returns.png")

    print("\n[SHAP] 可解釋性分析 (最後一個 Split)...")
    try:
        final_train_idx, final_val_idx = list(tscv.split(df))[-1]
        final_X_train = df.iloc[final_train_idx][feature_cols]
        final_X_val = df.iloc[final_val_idx][feature_cols]
        
        shap_exp = SHAPExplainer(last_models["XGBoost"], final_X_train)
        shap_exp.plot_summary(final_X_val, save_path=f"{figs}/shap_summary.png")
        top_feat = shap_exp.top_features(final_X_val, n=5)
        print("\n  TOP-5 重要特徵 (依 SHAP Mean |value|)：")
        print(top_feat.to_string())
    except Exception as e:
        logger.warning(f"SHAP 分析失敗: {e}")

    print("\n[完成] 全部流程結束！結果儲存於:", cfg["output"]["results_dir"])

if __name__ == "__main__":
    main()
