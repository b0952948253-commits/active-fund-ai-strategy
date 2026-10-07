![QR Code](https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=https://github.com/b0952948253-commits/active-fund-ai-strategy)

# 主動基金 AI 避險系統
# Active Fund AI Hedge Strategy

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made with](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest%20%7C%20LogisticRegression-green)]()

> 利用機器學習與總體經濟指標，對台灣主動型股票基金進行自動化絕對報酬預測與避險的量化研究系統。

---

## 📌 研究背景

台灣優質主動型基金（如安聯台灣大壩基金）長期年化報酬率達 **28%**，選股能力突出。  
然而在系統性熊市（2022 年 Fed 暴力升息）中，基金最大回撤高達 **-41.82%**。

**核心問題：** AI 能否預判「深水區」來臨，自動切換至現金避險，控制下行風險？

---

## 🏗️ 系統架構

```
active_fund_absolute/
├── main.py                  # 主程式 (訓練 + 回測 + SHAP)
├── config.yaml              # 全域設定
├── indicator_scanner.py     # 總經指標自動篩選器
├── generate_report.py       # Word 論文報告生成器
├── requirements.txt
├── src/
│   ├── data_loader.py       # 多來源資料載入 (Yahoo / FRED)
│   ├── feature_engineer.py  # 特徵工程 (滾動窗口 + 總經動能)
│   ├── labeler.py           # 標籤生成 (8% 止損 + 13週前瞻)
│   ├── models/
│   │   ├── xgboost_model.py
│   │   ├── random_forest_model.py
│   │   └── logistic_regression_model.py
│   ├── ensemble.py          # 集成模型 (Soft Voting)
│   ├── evaluator.py         # 回測 + 分類指標
│   └── explainer.py         # SHAP 可解釋性分析
├── dashboard/
│   └── app.py               # Streamlit 互動儀表板
└── results/
    └── figures/             # 自動生成的圖表
```

---

## 🔬 研究方法與模型迭代

### 版本演進

| 版本 | 特徵 | 資料頻率 | Out-of-Sample AUC | 最大回撤 |
|------|------|----------|-------------------|---------|
| V1 | VIX only | 月 | ~0.53 | -41.82% |
| V2 | VIX + 深水區強制進場 | 月→週 | ~0.78 | -41.82% |
| **V3（最終）** | **5大總經指標 + 取消強制進場** | **週** | **~0.60** | **-28.45%** |

### Walk-Forward Validation 架構

```
Split 1: Train 2009~2012 → Val 2012~2016
Split 2: Train 2009~2016 → Val 2016~2019
Split 3: Train 2009~2019 → Val 2019~2022
Split 4: Train 2009~2022 → Val 2022~2026
```

---

## 📊 最終實證結果

| 策略 | 年化報酬 | 年化波動 | Sharpe | **最大回撤** |
|------|---------|---------|--------|------------|
| LogisticRegression | 11.18% | 13.94% | 0.80 | **-28.45%** |
| XGBoost | 11.01% | 16.14% | 0.68 | -47.96% |
| RandomForest | 16.35% | 17.84% | 0.92 | -41.89% |
| Ensemble | 11.08% | 14.96% | 0.74 | **-28.45%** |
| **Buy & Hold** | **28.11%** | **22.00%** | **1.28** | -41.82% |

> ✅ LogisticRegression + Ensemble 成功將最大回撤從 **-41.82% → -28.45%**（縮減 13.37%）

---

## 🧠 SHAP 特徵重要性

AI 決策的前五大總經訊號：

1. 🥇 **DXY_mom_26w** — 美元指數 26 週動能（台股外資撤離訊號）
2. 🥈 **VIX_mom_26w** — 恐慌指數動能（市場情緒加速惡化）
3. 🥉 **volatility_52w** — 基金自身波動率
4. 🏅 **TNX_mean_26w** — 美債 10 年期殖利率（估值壓制）
5.  **sharpe_52w** — 基金風險調整後報酬趨勢

---

## 🚀 快速開始

### 安裝相依套件

```bash
pip install -r requirements.txt
```

### 執行完整訓練 + 回測

```bash
python main.py
```

### 啟動互動儀表板

```bash
streamlit run dashboard/app.py
```

### 總經指標篩選分析

```bash
python indicator_scanner.py
```

---

## ⚙️ 設定檔說明 (`config.yaml`)

```yaml
features:
  rolling_windows: [26, 52]     # 特徵滾動窗口（週）
  forward_window: 13            # 預測前瞻週數
  stop_loss_threshold: -0.08    # 單月止損線 -8%
  force_entry_drawdown: null    # 深水區強制進場（null = 停用）
```

---

## 📦 資料來源

| 資料 | 來源 | 說明 |
|------|------|------|
| 0050.TW (ETF) | Yahoo Finance | 基準指數 |
| 主動基金 NAV | MoneyDJ / CSV | 安聯台灣大壩基金 |
| VIX | Yahoo Finance `^VIX` | 恐慌指數 |
| DXY | Yahoo Finance `DX-Y.NYB` | 美元指數 |
| TNX | Yahoo Finance `^TNX` | 美債 10 年殖利率 |
| USD/TWD | Yahoo Finance `TWD=X` | 台幣匯率 |
| T10Y2Y | FRED | 長短債利差 |
| High Yield Spread | FRED `BAMLH0A0HYM2` | 高收益債利差 |

---

## 🖥️ 儀表板功能

- 📅 即時總經環境監控（DXY / VIX / TNX）
- 🤖 AI 避險訊號即時機率輸出
- 📈 歷史回測績效視覺化
- 🔍 可調整決策閾值 (threshold)

---

## 📄 授權

MIT License © 2026
