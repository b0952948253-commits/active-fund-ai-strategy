"""
標籤生成模組
============
預測目標（二元分類）：
  1 = 主動基金未來 N 週累積報酬大於 0，且期間內任一4週跌幅未超過止損線；或深水區強制進場
  0 = 未來報酬小於等於 0，或期間內發生暴跌（觸發止損）

注意：嚴格防止 Look-ahead Bias
  特徵 X_t 使用 t 時點「過去」資料
  標籤 y_t 使用 t 時點「未來」N 個月報酬
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_labels(
    fund_nav: pd.Series,
    etf_nav: pd.Series,
    forward_window: int = 13,
    stop_loss_threshold: float = -0.08,
    force_entry_drawdown: float = -0.15,
) -> pd.Series:
    """
    生成二元分類標籤。

    Parameters
    ----------
    fund_nav       : 主動基金月頻 NAV
    etf_nav        : ETF 月頻 NAV
    forward_window : 前瞻視窗（月），預設 3 個月

    Returns
    -------
    pd.Series（0 或 1），index = 月底日期
      注意：最後 forward_window 個時間點標籤為 NaN（無法計算未來報酬）
    """
    fund_ret = fund_nav.pct_change()
    etf_ret = etf_nav.pct_change()

    # 未來 N 個月累積報酬（shift(-N) 再 rolling(N).sum()）
    # 等效：對報酬做前向滾動加總
    future_fund = fund_ret.shift(-forward_window).rolling(forward_window).sum()
    future_etf  = etf_ret.shift(-forward_window).rolling(forward_window).sum()

    # y_t = sum(r_fund[t+1 ... t+N]) > 0
    future_fund = (fund_nav.shift(-forward_window) / fund_nav - 1)
    
    # 計算未來 N 週內的單月(4週)最低報酬
    rolling_4w_ret = fund_nav / fund_nav.shift(4) - 1
    min_4w_return = rolling_4w_ret.rolling(forward_window).min().shift(-forward_window)

    # 條件 1: 總報酬 > 0
    # 條件 2: 過程沒有觸發止損
    label = ((future_fund > 0) & (min_4w_return >= stop_loss_threshold)).astype(float)
    
    # 計算當前 52 週 MDD (用來標記深水區)
    def _mdd(x):
        if len(x) < 2: return np.nan
        cum_max = np.maximum.accumulate(x)
        return ((x - cum_max) / cum_max).min()
    current_mdd = fund_nav.rolling(52).apply(_mdd, raw=True)
    
    # 覆寫：如果當下處於深水區，強制標籤為 1 (進場抄底)
    label = np.where(current_mdd <= force_entry_drawdown, 1.0, label)
    label = pd.Series(label, index=fund_nav.index)
    
    # 只要 future_fund 是 NaN，代表時間不夠算未來報酬，就設為 NaN
    label = label.where(future_fund.notna(), other=np.nan)
    label.name = "label"

    n_pos = int(label.sum())
    n_total = label.notna().sum()
    logger.info(
        f"標籤生成完成（前瞻={forward_window}M）："
        f"共 {n_total} 個標籤，"
        f"正例（繼續抱牢）= {n_pos} ({n_pos/max(n_total,1)*100:.1f}%)，"
        f"負例（現金避險）= {n_total - n_pos} ({(n_total-n_pos)/max(n_total,1)*100:.1f}%)"
    )
    return label


def describe_label_distribution(label: pd.Series, train_idx, val_idx):
    """輸出訓練集與驗證集的標籤分佈統計"""
    for split_name, idx in [("訓練集", train_idx), ("驗證集", val_idx)]:
        y = label.loc[idx].dropna()
        pos = int(y.sum())
        neg = len(y) - pos
        print(
            f"  {split_name}: 共 {len(y)} 筆 │ "
            f"正例（1）= {pos} ({pos/max(len(y),1)*100:.1f}%) │ "
            f"負例（0）= {neg} ({neg/max(len(y),1)*100:.1f}%)"
        )
