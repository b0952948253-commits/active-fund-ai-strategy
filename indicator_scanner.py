import os
import yaml
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def fetch_yfinance_macro(start, end):
    tickers = {
        "VIX": "^VIX",
        "VVIX": "^VVIX",
        "SKEW": "^SKEW",
        "DXY": "DX-Y.NYB",
        "USD_TWD": "TWD=X",
        "TNX": "^TNX",
        "SOX": "^SOX"
    }
    dfs = []
    for name, ticker in tickers.items():
        logging.info(f"Fetching {name} ({ticker})")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            logging.warning(f"Empty df for {name}")
            continue
        
        # 處理 multi-index 欄位 (yfinance 新版)
        if isinstance(df.columns, pd.MultiIndex):
            close_col = ("Close", ticker) if ("Close", ticker) in df.columns else df.columns[df.columns.get_level_values(0) == "Close"][0]
            close = df[close_col]
        else:
            close = df["Close"]
            
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        weekly = close.resample("W-FRI").last().dropna()
        weekly.name = name
        dfs.append(weekly)
    return pd.concat(dfs, axis=1)

def fetch_fred_macro(start, end):
    tickers = {
        "T10Y2Y": "T10Y2Y",
        "HighYieldSpread": "BAMLH0A0HYM2"
    }
    dfs = []
    for name, ticker in tickers.items():
        logging.info(f"Fetching FRED {name} ({ticker})")
        try:
            df = web.DataReader(ticker, "fred", start, end)
            weekly = df.resample("W-FRI").last().dropna()
            weekly.columns = [name]
            dfs.append(weekly)
        except Exception as e:
            logging.error(f"Failed to fetch {ticker}: {e}")
    if dfs:
        return pd.concat(dfs, axis=1)
    return pd.DataFrame()

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    from src.data_loader import load_all_data
    etf_nav, fund_nav, vix_df = load_all_data(cfg)
    
    start = cfg["dates"]["data_start"]
    end = fund_nav.index.max().strftime("%Y-%m-%d")
    
    yf_df = fetch_yfinance_macro(start, end)
    fred_df = fetch_fred_macro(start, end)
    
    macro_df = pd.concat([yf_df, fred_df], axis=1)
    
    # 建立目標變數
    # 1. 基金 4 週與 13 週未來報酬
    fund_ret = fund_nav.pct_change()
    future_4w = (fund_nav.shift(-4) / fund_nav) - 1
    future_13w = (fund_nav.shift(-13) / fund_nav) - 1
    
    # 2. 未來 13 週是否跌入深水區 (MDD < -15%)
    def _mdd(x):
        if len(x) < 2: return np.nan
        cum_max = np.maximum.accumulate(x)
        return ((x - cum_max) / cum_max).min()
    
    # 計算每週開始往後 13 週內的最大回撤
    # 我們可以先對原本數列做前向 rolling，再 shift，確保沒 Look-ahead
    future_mdd_13w = fund_nav.shift(-13).rolling(13).apply(_mdd, raw=True)
    is_future_deep_water = (future_mdd_13w <= -0.15).astype(int)
    
    # 合併
    target_df = pd.DataFrame({
        "future_4w_ret": future_4w,
        "future_13w_ret": future_13w,
        "future_mdd_13w": future_mdd_13w,
        "is_future_deep_water": is_future_deep_water
    })
    
    df = pd.concat([macro_df, target_df], axis=1).dropna()
    
    # 計算特徵 (絕對值與 4 週動能)
    features = pd.DataFrame(index=df.index)
    for col in macro_df.columns:
        if col in df.columns:
            features[col] = df[col]
            features[f"{col}_mom_4w"] = df[col] - df[col].shift(4)
            
    features = features.dropna()
    df = df.loc[features.index]
    
    logging.info(f"Data prepared! Shape: {features.shape}")
    
    # 相關性分析
    corr = features.corrwith(df["future_13w_ret"]).sort_values(ascending=False)
    print("\n[未來 13 週報酬] 相關係數 (Top 5 & Bottom 5):")
    print("Top 5 (正相關, 數值越高未來報酬越好):")
    print(corr.head(5))
    print("\nBottom 5 (負相關, 數值越高未來報酬越差):")
    print(corr.tail(5))
    
    # RandomForest Feature Importance for Deep Water
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    y = df["is_future_deep_water"]
    print(f"\n深水區 (MDD < -15%) 發生次數: {y.sum()} / {len(y)}")
    
    if y.sum() > 0:
        rf.fit(features, y)
        importances = pd.Series(rf.feature_importances_, index=features.columns).sort_values(ascending=False)
        print("\n[未來 13 週深水區預測] Random Forest 特徵重要性 (Top 10):")
        print(importances.head(10))
        
        # 繪製長條圖
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.head(10).values, y=importances.head(10).index)
        plt.title("Top 10 Macro Indicators for Predicting Deep Water (MDD < -15%)")
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/macro_importance.png")
        logging.info("Saved macro importance plot to results/macro_importance.png")
    else:
        logging.warning("No deep water events found in aligned data.")

if __name__ == "__main__":
    main()
