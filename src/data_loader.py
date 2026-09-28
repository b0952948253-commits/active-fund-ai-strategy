"""
資料載入模組
============
支援資料來源：
  1. yfinance       — 0050.TW ETF（主要）
  2. MoneyDJ 爬蟲   — 安聯台灣大壩基金（主要）
  3. CSV 備援       — 若自動抓取失敗，讀取使用者提供的 CSV
  4. 合成模擬資料   — 開發測試用

CSV 備援格式（data/raw/active_fund_nav.csv）：
    Date,NAV
    2020-01-02,15.32
    2020-02-03,15.11
    ...
"""

import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# ETF 資料（yfinance）
# ──────────────────────────────────────────────

def fetch_etf_weekly(ticker: str, start: str, end: str) -> pd.Series:
    """
    從 Yahoo Finance 下載 ETF 週頻收盤價。

    Returns
    -------
    pd.Series  index=每週五日期, values=調整後收盤價
    """
    logger.info(f"[ETF] 下載 {ticker}，區間 {start} ~ {end}")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"yfinance 無法取得 {ticker} 資料，請確認網路連線。")

    # 轉週頻（取每週五最後一個交易日）
    close = df["Close"]
    # yfinance 新版可能回傳 DataFrame（多欄位），壓縮成 Series
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    weekly = close.resample("W-FRI").last().dropna()
    weekly.name = "etf_nav"
    logger.info(f"[ETF] 成功取得 {len(weekly)} 筆週頻資料")
    return weekly


import pandas_datareader.data as web

def fetch_macro_weekly(start: str, end: str) -> pd.DataFrame:
    """
    從 Yahoo Finance 與 FRED 下載 總經指標。
    """
    # Yahoo Finance 指標
    yf_tickers = {
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
        "USD_TWD": "TWD=X",
        "TNX": "^TNX"
    }
    
    dfs = []
    for name, ticker in yf_tickers.items():
        logger.info(f"[{name}] 下載 {ticker}，區間 {start} ~ {end}")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            continue
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

    # VIX 的 High 也保留 (相容舊版)
    vix_df = yf.download("^VIX", start=start, end=end, progress=False)
    if not vix_df.empty:
        if isinstance(vix_df.columns, pd.MultiIndex):
            high_col = ("High", "^VIX") if ("High", "^VIX") in vix_df.columns else vix_df.columns[vix_df.columns.get_level_values(0) == "High"][0]
            vix_high = vix_df[high_col]
        else:
            vix_high = vix_df["High"]
        if isinstance(vix_high, pd.DataFrame):
            vix_high = vix_high.squeeze()
        vix_high_w = vix_high.resample("W-FRI").max().dropna()
        vix_high_w.name = "vix_high"
        dfs.append(vix_high_w)

    # FRED 指標
    fred_tickers = {
        "T10Y2Y": "T10Y2Y",
        "HighYieldSpread": "BAMLH0A0HYM2"
    }
    for name, ticker in fred_tickers.items():
        logger.info(f"[{name}] 下載 FRED {ticker}，區間 {start} ~ {end}")
        try:
            df = web.DataReader(ticker, "fred", start, end)
            weekly = df.resample("W-FRI").last().dropna()
            weekly.columns = [name]
            dfs.append(weekly)
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")

    macro_df = pd.concat(dfs, axis=1)
    logger.info(f"[Macro] 成功取得 {len(macro_df)} 筆週頻總經資料")
    return macro_df

# ──────────────────────────────────────────────
# 主動基金資料（MoneyDJ 爬蟲）
# ──────────────────────────────────────────────

MONEYDJ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
    "Referer": "https://www.moneydj.com/",
}


def _search_moneydj_fund_code(fund_name: str) -> str | None:
    """透過 MoneyDJ 搜尋基金代碼"""
    url = "https://www.moneydj.com/funddj/yp/yp010.djhtm"
    params = {"a": fund_name}
    try:
        resp = requests.get(url, params=params, headers=MONEYDJ_HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # 尋找基金連結（格式：/funddj/yp/yp011000.djhtm?a=XXXXXX）
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "yp011000" in href and "a=" in href:
                code = href.split("a=")[-1].split("&")[0].strip()
                if code:
                    logger.info(f"[MoneyDJ] 搜尋到基金代碼: {code}")
                    return code
    except Exception as e:
        logger.warning(f"[MoneyDJ] 搜尋失敗: {e}")
    return None


def _fetch_moneydj_nav(fund_code: str, start: str, end: str) -> pd.Series | None:
    """
    爬取 MoneyDJ 基金歷史淨值。
    URL 格式: https://www.moneydj.com/funddj/yp/yp011000.djhtm?a={fund_code}
    """
    url = f"https://www.moneydj.com/funddj/yp/yp011000.djhtm"
    params = {"a": fund_code}
    try:
        resp = requests.get(url, params=params, headers=MONEYDJ_HEADERS, timeout=20)
        resp.encoding = "utf-8"
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # 找包含淨值的表格
        tables = pd.read_html(io.StringIO(str(soup)), thousands=",", flavor="lxml")
        for tbl in tables:
            if tbl.shape[1] >= 2:
                # 試著找日期列和淨值列
                for col in tbl.columns:
                    col_str = str(tbl[col].iloc[0]) if len(tbl) > 0 else ""
                    if "/" in col_str or "-" in col_str:
                        date_col = col
                        val_col = tbl.columns[tbl.columns.get_loc(col) + 1]
                        try:
                            dates = pd.to_datetime(tbl[date_col], errors="coerce")
                            vals = pd.to_numeric(tbl[val_col], errors="coerce")
                            s = pd.Series(vals.values, index=dates).dropna()
                            s = s[s.index.notna()]
                            if len(s) > 10:
                                # 轉週頻
                                s = s.sort_index().resample("W-FRI").last().dropna()
                                s = s.loc[start:end]
                                s.name = "fund_nav"
                                logger.info(f"[MoneyDJ] 成功解析 {len(s)} 筆淨值資料")
                                return s
                        except Exception:
                            continue
    except Exception as e:
        logger.warning(f"[MoneyDJ] 爬取失敗（代碼={fund_code}）: {e}")
    return None


# ──────────────────────────────────────────────
# CSV 備援
# ──────────────────────────────────────────────

def load_fund_from_csv(csv_path: str, start: str, end: str) -> pd.Series:
    """
    從 CSV 讀取基金 NAV。
    期望欄位：Date（日期）, NAV（淨值）
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"\n[備援 CSV] 找不到檔案：{csv_path}\n"
            "請手動下載基金淨值後存成 CSV，格式範例：\n"
            "  Date,NAV\n"
            "  2020-01-31,15.32\n"
            "  2020-02-28,14.98\n"
            "資料來源建議：MoneyDJ / CMoney / 投信公司官網"
        )
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    s = df["NAV"].sort_index()
    # 轉週頻
    s = s.resample("W-FRI").last().dropna()
    s = s.loc[start:end]
    s.name = "fund_nav"
    logger.info(f"[CSV] 讀取 {len(s)} 筆週頻 NAV")
    return s


# ──────────────────────────────────────────────
# 合成模擬資料（開發 / 測試用）
# ──────────────────────────────────────────────

def _generate_synthetic_fund(etf_series: pd.Series, seed: int = 42) -> pd.Series:
    """
    以 ETF 報酬為基礎，疊加主動管理 alpha 與 tracking error，
    模擬真實主動基金 NAV 走勢。
    （僅供系統測試，不代表真實績效）
    """
    rng = np.random.default_rng(seed)

    # 確保是 1D Series
    if isinstance(etf_series, pd.DataFrame):
        etf_series = etf_series.squeeze()

    etf_ret = etf_series.pct_change().dropna()
    etf_vals = etf_ret.values.astype(float).flatten()  # 確保 numpy 1D array

    # 參數：年化 alpha 0.5%、TE 4%（月化）
    monthly_alpha = 0.005 / 12
    monthly_te = 0.04 / np.sqrt(12)
    beta = 0.95

    noise = rng.normal(0, monthly_te, len(etf_vals))
    fund_ret_vals = beta * etf_vals + monthly_alpha + noise
    fund_nav = (1 + pd.Series(fund_ret_vals, index=etf_ret.index)).cumprod() * 15.0
    fund_nav.name = "fund_nav"
    logger.warning(
        "[合成資料] 使用模擬 NAV（非真實資料）。"
        "請提供真實 CSV 以取得正確分析結果。"
    )
    return fund_nav


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def load_all_data(cfg: dict) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    載入 ETF、主動基金週頻 NAV 與總經資料。

    Returns
    -------
    etf_nav  : pd.Series
    fund_nav : pd.Series
    macro_df : pd.DataFrame
    """
    start = cfg["dates"]["data_start"]
    
    # 如果設定檔沒有 val_end (如 Walk-Forward 模式)，預設抓到今天
    from datetime import datetime
    end = cfg["dates"].get("val_end", datetime.today().strftime("%Y-%m-%d"))

    # 1. ETF
    etf_nav = fetch_etf_weekly(cfg["etf"]["ticker"], start, end)

    # 2. 主動基金（三層備援）
    fund_nav = None
    fund_cfg = cfg["active_fund"]

    # 嘗試 MoneyDJ（直接用設定的代碼）
    fund_code = fund_cfg.get("moneydj_code", "")
    if fund_code:
        logger.info(f"[主動基金] 嘗試 MoneyDJ 代碼: {fund_code}")
        fund_nav = _fetch_moneydj_nav(fund_code, start, end)
        time.sleep(1)

    # 若失敗，搜尋代碼
    if fund_nav is None:
        logger.info("[主動基金] 嘗試以基金名稱搜尋 MoneyDJ 代碼...")
        found_code = _search_moneydj_fund_code(fund_cfg["name"])
        if found_code:
            fund_nav = _fetch_moneydj_nav(found_code, start, end)

    # 若失敗，嘗試 CSV
    if fund_nav is None:
        csv_path = fund_cfg.get("csv_fallback", "data/raw/active_fund_nav.csv")
        try:
            logger.info(f"[主動基金] 嘗試讀取備援 CSV: {csv_path}")
            fund_nav = load_fund_from_csv(csv_path, start, end)
        except FileNotFoundError as e:
            logger.warning(str(e))

    # 最終備援：合成資料
    if fund_nav is None:
        logger.warning("[主動基金] 所有資料來源失敗，改用合成模擬資料（測試用）。")
        fund_nav = _generate_synthetic_fund(etf_nav, seed=42)

    # 對齊索引
    macro_df = fetch_macro_weekly(start, end)

    common_idx = etf_nav.index.intersection(fund_nav.index).intersection(macro_df.index)
    etf_nav = etf_nav.loc[common_idx]
    fund_nav = fund_nav.loc[common_idx]
    macro_df = macro_df.loc[common_idx]

    logger.info(f"資料對齊完成，共 {len(common_idx)} 個週頻時間點")
    return etf_nav, fund_nav, macro_df
