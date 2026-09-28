"""
特徵工程模組
============
計算以下滾動特徵（支援多個時間視窗）：

  績效類：
    - excess_ret       超額報酬（vs 無風險利率）
    - sharpe           Sharpe Ratio
    - sortino          Sortino Ratio
    - calmar           Calmar Ratio
    - ann_ret          年化報酬率

  風險類：
    - max_drawdown     最大回撤
    - volatility       年化波動率
    - downside_vol     下行風險

  相對 ETF 類：
    - alpha            Jensen's Alpha（vs ETF）
    - beta             市場敏感度（vs ETF）
    - ir               資訊比率（Information Ratio）
    - tracking_error   追蹤誤差
    - active_ret_cumul 累積主動報酬

  基金特性：
    - ret_momentum     報酬動能（近N月 vs 遠N月）
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 輔助函數
# ──────────────────────────────────────────────

def _rolling_drawdown(nav: pd.Series, window: int) -> pd.Series:
    """計算滾動最大回撤"""
    def _mdd(x):
        if len(x) < 2:
            return np.nan
        cum_max = np.maximum.accumulate(x)
        dd = (x - cum_max) / cum_max
        return dd.min()
    return nav.rolling(window).apply(_mdd, raw=True)


def _rolling_downside_std(excess_ret: pd.Series, window: int) -> pd.Series:
    """下行標準差（只計算負超額報酬）"""
    def _ds(x):
        neg = x[x < 0]
        return neg.std() if len(neg) > 1 else np.nan
    return excess_ret.rolling(window).apply(_ds, raw=True)


def _rolling_alpha_beta(fund_ret: pd.Series, etf_ret: pd.Series, window: int):
    """
    滾動 OLS 回歸：fund_ret = alpha + beta * etf_ret + e
    回傳 (alpha_series, beta_series)
    """
    alphas, betas = [], []
    for i in range(len(fund_ret)):
        if i < window - 1:
            alphas.append(np.nan)
            betas.append(np.nan)
        else:
            y = fund_ret.iloc[i - window + 1: i + 1].values
            x = etf_ret.iloc[i - window + 1: i + 1].values
            if np.std(x) < 1e-8:
                alphas.append(np.nan)
                betas.append(np.nan)
                continue
            # OLS: β = cov(x,y)/var(x), α = mean(y) - β*mean(x)
            beta_val = np.cov(x, y)[0, 1] / np.var(x)
            alpha_val = np.mean(y) - beta_val * np.mean(x)
            # 年化 alpha（月頻 × 12）
            alphas.append(alpha_val * 12)
            betas.append(beta_val)

    alpha_s = pd.Series(alphas, index=fund_ret.index)
    beta_s = pd.Series(betas, index=fund_ret.index)
    return alpha_s, beta_s


# ──────────────────────────────────────────────
# 主函數
# ──────────────────────────────────────────────

def compute_features(
    fund_nav: pd.Series,
    etf_nav: pd.Series,
    macro_df: pd.DataFrame = None,
    rf_annual: float = 0.015,
    windows: list[int] = (26, 52),
) -> pd.DataFrame:
    """
    計算全部特徵。

    Parameters
    ----------
    fund_nav   : 主動基金月頻 NAV
    etf_nav    : ETF 月頻 NAV
    rf_annual  : 年化無風險利率
    windows    : 滾動視窗（月）列表

    Returns
    -------
    pd.DataFrame，index = 月底日期，columns = 特徵名稱
    """
    rf_weekly = rf_annual / 52

    # 月報酬率
    fund_ret = fund_nav.pct_change()
    etf_ret = etf_nav.pct_change()
    excess_ret = fund_ret - rf_weekly
    active_ret = fund_ret - etf_ret  # 主動報酬（相對 ETF）

    feature_frames = []

    for w in windows:
        tag = f"{w}w"
        sqrt_w = np.sqrt(w)
        sqrt_52 = np.sqrt(52)

        # ── 績效 ──
        ann_ret = fund_ret.rolling(w).mean() * 52
        volatility = fund_ret.rolling(w).std() * sqrt_52
        excess_mean = excess_ret.rolling(w).mean()

        sharpe = (excess_mean / (fund_ret.rolling(w).std() + 1e-8)) * sqrt_52

        downside_s = _rolling_downside_std(excess_ret, w)
        sortino = (excess_mean / (downside_s + 1e-8)) * sqrt_52

        mdd = _rolling_drawdown(fund_nav, w)
        calmar = np.where(
            mdd.abs() > 1e-8, ann_ret / mdd.abs(), np.nan
        )
        calmar = pd.Series(calmar, index=fund_nav.index)

        # ── Alpha / Beta ──
        alpha, beta = _rolling_alpha_beta(fund_ret, etf_ret, w)

        # ── 相對 ETF ──
        ir = (active_ret.rolling(w).mean() / (active_ret.rolling(w).std() + 1e-8)) * sqrt_52
        te = active_ret.rolling(w).std() * sqrt_52
        active_cumul = active_ret.rolling(w).sum()

        # ── 動能 ──
        # 近半 vs 遠半 動能差
        half = max(w // 2, 1)
        momentum = fund_ret.rolling(half).sum() - fund_ret.shift(half).rolling(half).sum()

        # 深水區標記
        is_deep_water = (mdd <= -0.15).astype(float)

        # 整合到 DataFrame
        df_w = pd.DataFrame({
            f"ann_ret_{tag}":         ann_ret,
            f"volatility_{tag}":      volatility,
            f"downside_vol_{tag}":    downside_s,
            f"sharpe_{tag}":          sharpe,
            f"sortino_{tag}":         sortino,
            f"calmar_{tag}":          calmar,
            f"max_drawdown_{tag}":    mdd,
            f"is_deep_water_{tag}":   is_deep_water,
            f"alpha_{tag}":           alpha,
            f"beta_{tag}":            beta,
            f"ir_{tag}":              ir,
            f"tracking_error_{tag}":  te,
            f"active_cumul_{tag}":    active_cumul,
            f"momentum_{tag}":        momentum,
        })
        
        # ── 總經與情緒特徵 (Macro) ──
        if macro_df is not None and not macro_df.empty:
            for col in macro_df.columns:
                if col == "vix_high":
                    df_w[f"vix_high_max_{tag}"] = macro_df[col].rolling(w).max()
                else:
                    df_w[f"{col}_mean_{tag}"] = macro_df[col].rolling(w).mean()
                    df_w[f"{col}_mom_{tag}"] = macro_df[col].rolling(half).mean() - macro_df[col].shift(half).rolling(half).mean()

        feature_frames.append(df_w)

    features = pd.concat(feature_frames, axis=1)

    # 移除全 NaN 的行（前 max(windows) 筆沒有有效特徵）
    features = features.dropna(how="all")

    # 無限值替換
    features = features.replace([np.inf, -np.inf], np.nan)
    
    # 向前/向後填補剩餘缺失值
    features = features.ffill().bfill()
    
    # 特徵降維 (針對絕對報酬與下行風險)
    selected_features = [
        "max_drawdown_26w",
        "max_drawdown_52w",
        "downside_vol_52w",
        "ann_ret_26w",
        "volatility_52w",
        "momentum_26w",
        "sharpe_52w",
        "beta_52w",
        "VIX_mean_26w",
        "VIX_mom_26w",
        "vix_high_max_26w",
        "HighYieldSpread_mean_26w",
        "HighYieldSpread_mom_26w",
        "T10Y2Y_mean_26w",
        "T10Y2Y_mom_26w",
        "USD_TWD_mean_26w",
        "USD_TWD_mom_26w",
        "TNX_mean_26w",
        "DXY_mean_26w",
        "DXY_mom_26w",
        "is_deep_water_26w",
        "is_deep_water_52w"
    ]
    
    # 確保特徵存在
    available_features = [f for f in selected_features if f in features.columns]
    features = features[available_features]

    logger.info(
        f"特徵工程完成：縮減至 {features.shape[1]} 個核心特徵，"
        f"{len(features)} 個有效時間點"
    )
    return features
