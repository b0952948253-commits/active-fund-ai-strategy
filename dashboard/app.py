"""
Streamlit 互動儀表板
====================
啟動方式：
  streamlit run dashboard/app.py

功能：
  - 側邊欄：模型選擇 / 閾值調整
  - 主頁面：
      1. NAV 走勢圖（基金 vs ETF）
      2. 滾動 Sharpe Ratio 比較
      3. 模型預測機率時序圖
      4. 驗證集指標表格
      5. 累積報酬回測圖
      6. 特徵重要性橫條圖
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

# ── 路徑 ──
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_all_data
from src.feature_engineer import compute_features
from src.labeler import generate_labels
from src.models.xgboost_model import XGBoostModel
from src.models.rf_model import RandomForestModel
from src.models.lr_model import LogisticRegressionModel
from src.ensemble import EnsembleModel
from src.evaluator import evaluate_all, backtest, compute_backtest_stats
from sklearn.impute import SimpleImputer

matplotlib.rcParams["font.family"] = "Microsoft JhengHei"

st.set_page_config(
    page_title="主動基金 vs ETF 預測系統",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────
# 樣式
# ─────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0f1117; }
  .metric-card {
    background: linear-gradient(135deg, #1e2130, #252840);
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid #2d3154;
  }
  .section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #7EB8F7;
    margin-bottom: 8px;
  }
  .stDataFrame thead { background-color: #1e2130 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 資料與模型快取
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="📥 載入資料中...")
def get_data():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    etf_nav, fund_nav, macro_df = load_all_data(cfg)
    features = compute_features(
        fund_nav, etf_nav, macro_df=macro_df,
        rf_annual=cfg["risk_free_rate"]["annual"],
        windows=cfg["features"]["rolling_windows"],
    )
    labels = generate_labels(
        fund_nav, etf_nav,
        forward_window=cfg["features"]["forward_window"],
        stop_loss_threshold=cfg["features"].get("stop_loss_threshold", -0.08),
        force_entry_drawdown=cfg["features"].get("force_entry_drawdown", -0.15),
    )
    df = features.join(labels, how="inner").dropna(subset=["label"])
    feat_cols = [c for c in df.columns if c != "label"]
    imputer = SimpleImputer(strategy="median")
    df[feat_cols] = imputer.fit_transform(df[feat_cols])
    return cfg, df, etf_nav, fund_nav, feat_cols


@st.cache_resource(show_spinner="🤖 訓練模型中（首次載入約需 1 分鐘）...")
def get_trained_models(_cfg, _X_train, _y_train, _X_val, _y_val):
    xgb = XGBoostModel(_cfg).fit(_X_train, _y_train, _X_val, _y_val)
    rf  = RandomForestModel(_cfg).fit(_X_train, _y_train, _X_val, _y_val)
    lr = LogisticRegressionModel(_cfg).fit(_X_train, _y_train, _X_val, _y_val)
    ens_cfg = _cfg["ensemble"]
    ens = EnsembleModel(
        {"xgboost": xgb, "random_forest": rf, "logistic_regression": lr},
        weights=ens_cfg.get("weights", {}),
        method=ens_cfg.get("method", "soft_vote"),
    )
    return {"XGBoost": xgb, "RandomForest": rf, "LogisticRegression": lr, "Ensemble": ens}


# ─────────────────────────────────────────────
# 主頁
# ─────────────────────────────────────────────

def main():
    # ── 標題 ──
    st.markdown("""
    <h1 style='text-align:center; color:#7EB8F7; margin-bottom:4px;'>
      📈 主動基金「絕對報酬」最大化與避險系統
    </h1>
    <p style='text-align:center; color:#8892b0; font-size:0.95rem;'>
      安聯台灣大壩基金-A類型-新臺幣　vs.　現金避險 (0% 報酬)
    </p>
    <hr style='border:1px solid #2d3154; margin:12px 0 20px;'>
    """, unsafe_allow_html=True)

    # ── 載入資料 ──
    cfg, df, etf_nav, fund_nav, feat_cols = get_data()

    # 針對 Dashboard 視覺化，抓取最後 3 年做為驗證集
    val_start = (df.index[-1] - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    train = df.loc[:val_start].iloc[:-1]
    val   = df.loc[val_start:]
    
    X_train, y_train = train[feat_cols], train["label"].astype(int)
    X_val,   y_val   = val[feat_cols],   val["label"].astype(int)

    # ── 訓練模型 ──
    models = get_trained_models(cfg, X_train, y_train, X_val, y_val)

    # ── 側邊欄 ──
    with st.sidebar:
        st.markdown("### ⚙️ 控制面板")
        selected_model = st.selectbox("選擇模型", list(models.keys()), index=3)
        threshold = st.slider("決策閾值", 0.3, 0.8, 0.5, 0.05)
        show_train = st.checkbox("顯示訓練期間", value=True)
        st.markdown("---")
        st.markdown(f"**模型展示用訓練期**：2009–{pd.to_datetime(val_start).year - 1}（{len(X_train)} 筆）")
        st.markdown(f"**模型展示用驗證期**：{pd.to_datetime(val_start).year}–2026（{len(X_val)} 筆）")
        st.markdown(f"**特徵數**：{len(feat_cols)}")

    # ── 版面 ──
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 基金走勢", "🎯 預測結果", "💰 回測績效", "🔍 特徵重要性"]
    )

    # ────────────────────────────────
    # Tab 1：走勢
    # ────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">NAV 走勢比較</div>', unsafe_allow_html=True)

        nav_range = (etf_nav if not show_train else etf_nav)
        fig = go.Figure()

        # 歸一化
        base_etf  = etf_nav.iloc[0]
        base_fund = fund_nav.iloc[0]
        etf_norm  = etf_nav / base_etf * 100
        fund_norm = fund_nav / base_fund * 100

        fig.add_trace(go.Scatter(
            x=etf_norm.index, y=etf_norm.values,
            name="元大台灣50 ETF", line=dict(color="#457B9D", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=fund_norm.index, y=fund_norm.values,
            name="安聯台灣大壩基金", line=dict(color="#E63946", width=2.5),
        ))
        fig.add_vline(
            x=pd.Timestamp(val_start), line_dash="dash",
            line_color="yellow", annotation_text="驗證期起",
            annotation_font_color="yellow",
        )
        fig.update_layout(
            template="plotly_dark", height=380,
            yaxis_title="相對績效（初始=100）",
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 滾動 Sharpe（12M）
        st.markdown('<div class="section-title">12 個月滾動 Sharpe Ratio</div>', unsafe_allow_html=True)
        rf_m = cfg["risk_free_rate"]["annual"] / 12
        fund_r  = fund_nav.pct_change()
        etf_r   = etf_nav.pct_change()
        sh_fund = ((fund_r - rf_m).rolling(12).mean() / fund_r.rolling(12).std() * np.sqrt(12)).dropna()
        sh_etf  = ((etf_r  - rf_m).rolling(12).mean() / etf_r.rolling(12).std()  * np.sqrt(12)).dropna()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sh_fund.index, y=sh_fund.values, name="基金 Sharpe",
                                   line=dict(color="#E63946", width=2)))
        fig2.add_trace(go.Scatter(x=sh_etf.index, y=sh_etf.values,  name="ETF Sharpe",
                                   line=dict(color="#457B9D", width=2)))
        fig2.add_hline(y=0, line_dash="dot", line_color="gray")
        fig2.update_layout(
            template="plotly_dark", height=300,
            yaxis_title="Sharpe Ratio",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ────────────────────────────────
    # Tab 2：預測
    # ────────────────────────────────
    with tab2:
        model = models[selected_model]
        proba = model.predict_proba(X_val)
        proba = np.where(np.isnan(proba), 0.5, proba)
        if len(proba) < len(y_val):
            proba = np.concatenate([np.full(len(y_val) - len(proba), 0.5), proba])

        # 指標卡
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
        auc = roc_auc_score(y_val, proba) if len(set(y_val)) > 1 else 0.5
        f1  = f1_score(y_val, (proba >= threshold).astype(int), zero_division=0)
        acc = accuracy_score(y_val, (proba >= threshold).astype(int))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{auc:.3f}")
        c2.metric("F1-Score", f"{f1:.3f}")
        c3.metric("Accuracy", f"{acc:.1%}")
        c4.metric("樣本數（驗證集）", len(y_val))

        # 預測機率時序
        st.markdown('<div class="section-title">預測機率時序圖（基金正報酬機率）</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=y_val.index, y=proba,
            name="預測機率", line=dict(color="#7EB8F7", width=2),
        ))
        fig3.add_trace(go.Scatter(
            x=y_val.index, y=y_val.values,
            name="實際標籤（0/1）", line=dict(color="#E9C46A", width=1.5, dash="dot"),
        ))
        fig3.add_hline(y=threshold, line_dash="dash", line_color="#2A9D8F",
                       annotation_text=f"閾值={threshold}", annotation_font_color="#2A9D8F")
        fig3.update_layout(
            template="plotly_dark", height=320,
            yaxis_title="機率 / 標籤",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # 所有模型評估表
        st.markdown('<div class="section-title">所有模型評估指標比較</div>', unsafe_allow_html=True)
        eval_df = evaluate_all(models, X_val, y_val)
        st.dataframe(eval_df.style.highlight_max(axis=0, subset=["AUC", "F1", "Accuracy"])
                              .highlight_min(axis=0, subset=["Brier"])
                              .format("{:.4f}"), use_container_width=True)

    # ────────────────────────────────
    # Tab 3：回測
    # ────────────────────────────────
    with tab3:
        fund_ret = fund_nav.pct_change().dropna()

        model = models[selected_model]
        bt = backtest(model, X_val, fund_ret, model_name=selected_model, threshold=threshold, force_entry_drawdown=cfg["features"].get("force_entry_drawdown", None))

        # 累積報酬
        fig4 = go.Figure()
        colors_bt = {"主動基金報酬": "#E63946", "現金(避險)報酬": "#457B9D"}
        for col in bt.columns:
            cum = (1 + bt[col].dropna()).cumprod()
            color = colors_bt.get(col, "#2A9D8F")
            dash  = "solid" if col != f"{selected_model}_策略報酬" else "dash"
            fig4.add_trace(go.Scatter(
                x=cum.index, y=cum.values,
                name=col, line=dict(color=color, width=2, dash=dash),
            ))
        fig4.update_layout(
            template="plotly_dark", height=350,
            yaxis_title="累積報酬倍數",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig4, use_container_width=True)

        # 統計表
        st.markdown('<div class="section-title">回測統計摘要</div>', unsafe_allow_html=True)
        stat_rows = []
        for col in bt.columns:
            stats = compute_backtest_stats(bt, col)
            stats["策略"] = col
            stat_rows.append(stats)
        stat_df = pd.DataFrame(stat_rows).set_index("策略")
        st.dataframe(stat_df, use_container_width=True)

    # ────────────────────────────────
    # Tab 4：特徵重要性
    # ────────────────────────────────
    with tab4:
        st.markdown('<div class="section-title">XGBoost 特徵重要性 Top 15</div>', unsafe_allow_html=True)
        fi = models["XGBoost"].feature_importances.head(15).sort_values()
        fig5 = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation="h",
            marker=dict(color=fi.values, colorscale="Viridis"),
        ))
        fig5.update_layout(
            template="plotly_dark", height=450,
            xaxis_title="Importance Score",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown('<div class="section-title">Random Forest 特徵重要性 Top 15</div>', unsafe_allow_html=True)
        fi_rf = models["RandomForest"].feature_importances.head(15).sort_values()
        fig6 = go.Figure(go.Bar(
            x=fi_rf.values, y=fi_rf.index, orientation="h",
            marker=dict(color=fi_rf.values, colorscale="Plasma"),
        ))
        fig6.update_layout(
            template="plotly_dark", height=450,
            xaxis_title="Importance Score",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig6, use_container_width=True)


if __name__ == "__main__":
    main()
