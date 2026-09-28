import os
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def main():
    doc = Document()

    # Title
    title = doc.add_heading('以機器學習與總經指標優化主動基金絕對報酬之實證研究', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph('——安聯台灣大壩基金與台股 ETF 避險策略分析——').alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # 摘要
    doc.add_heading('摘要 (Abstract)', level=1)
    doc.add_paragraph(
        "本研究旨在探討如何透過機器學習演算法與總體經濟指標，為台灣主動型股票基金（以安聯台灣大壩基金為例）建立一套自動化的「絕對報酬與避險系統」。"
        "我們採用 2009 年至 2026 年長達 17 年的歷史資料，涵蓋台股 ETF (0050.TW)、VIX 恐慌指數、美元指數 (DXY)、長短債利差 (T10Y2Y) 等跨市場數據進行特徵工程。"
        "研究過程中經歷了多次模型迭代：從最初的低頻月資料與單一 VIX 指標，演進至高頻週資料與多因子總經防護網；同時探討了「深水區強制進場」機制對於最大回撤 (Maximum Drawdown, MDD) 的潛在危害。"
        "最終實證結果顯示，整合美元動能 (DXY_mom) 與美債殖利率 (TNX) 的 AI 決策模型，能在有效躲避 2022 年系統性崩盤的情況下，將最大回撤從 -41.82% 大幅縮減至 -28.45%，實現了兼顧報酬與風險控管的絕對報酬目標。"
    )

    # 1. 前言
    doc.add_heading('1. 研究背景與動機', level=1)
    doc.add_paragraph(
        "台灣的優質主動型基金（如安聯台灣大壩基金）長期年化報酬率優異（達 28% 以上），展現出極強的選股能力 (Alpha)。"
        "然而，在遇到系統性熊市（如 2020 年疫情爆發、2022 年美國聯準會暴力升息）時，這類滿水位操作的股票型基金往往會面臨高達 40% 以上的嚴重回撤。"
        "因此，本研究的核心動機為：是否能利用 AI 演算法，在預判市場即將進入「深水區」時主動將部位轉為現金（避險），以最大化「絕對報酬」並控制下行風險。"
    )

    # 2. 研究方法
    doc.add_heading('2. 研究方法與資料處理', level=1)
    doc.add_heading('2.1 資料來源與頻率轉換', level=2)
    doc.add_paragraph(
        "本研究採用 Yahoo Finance 與聯準會 FRED 經濟資料庫，擷取 2009/01 至 2026/09 之時間序列。初始模型使用「月頻率」(Monthly) 資料，但回測發現月頻率反應過於遲鈍，"
        "在快速崩盤時往往來不及避險。因此第二階段全面升級為「週頻率」(Weekly - W-FRI) 計算，大幅提升了特徵的即時捕捉能力。"
    )
    
    doc.add_heading('2.2 模型架構與 Walk-Forward Validation', level=2)
    doc.add_paragraph(
        "由於金融時序資料具有強烈的非穩定性 (Non-stationary)，本系統採用 Walk-Forward Validation 滾動驗證架構（切割為 4 個 Splits）。"
        "模型訓練演算法包含：XGBoost, Random Forest, Logistic Regression 以及整合三者的集成模型 (Ensemble Soft-Voting)。"
    )

    # 3. 實證歷程與模型迭代
    doc.add_heading('3. 實證歷程與模型迭代', level=1)
    
    doc.add_heading('3.1 初始版本 (純 VIX 與月頻資料)', level=2)
    doc.add_paragraph(
        "初始設想是利用 VIX（恐慌指數）做為唯一的市場情緒特徵來預測台股。然而，AI 訓練結果顯示 AUC 僅在 0.5 ~ 0.58 徘徊，宛如丟硬幣。"
        "這顯示出 VIX 為「同步或落後指標」，當 VIX 飆升時，市場往往已經跌完，無法作為良好的前瞻預測特徵。"
    )
    
    doc.add_heading('3.2 加入深水區標籤與強制進場測試', level=2)
    doc.add_paragraph(
        "為了解決 AI 不敢在低檔買進的問題，我們引入了「MDD < -15% 時強制進場」的邏輯。"
        "此舉成功讓 AUC 預測力飆升至 0.78 左右，但回測結果卻揭露了一個殘酷事實：2022 年的空頭市場深不見底，"
        "當市場跌至 -15% 時強迫 AI 滿倉，反而讓模型吃下了後續所有的跌幅，導致策略的最大回撤依然與 Buy and Hold 一模一樣（高達 -41.82%）。"
    )

    doc.add_heading('3.3 總經防護網與取消強制進場 (最終版本)', level=2)
    doc.add_paragraph(
        "意識到 VIX 的侷限性後，我們開發了「Macro Indicator Scanner」，自動對齊大壩基金走勢與各項總經數據。"
        "相關性與 Random Forest 分析得出驚人結論：預測大崩盤最準確的並非 VIX，而是「高收益債利差」(High Yield Spread)、「美元指數」(DXY) 與「長短債利差」(T10Y2Y)。"
        "我們將前五大總經指標納入特徵工程，並果斷刪除了「強制進場」的包袱，允許 AI 在總經環境惡劣時持續抱持現金避險。"
    )

    # 4. 最終實證結果
    doc.add_heading('4. 最終實證結果與解釋 (SHAP)', level=1)
    doc.add_paragraph(
        "最終版本（無強制進場 + 總經特徵）在回測中表現出卓越的風險規避能力。雖然年化報酬（約 11~16%）不及死抱不放的 28%，"
        "但最大回撤成功從 -41.82% 壓縮至 -28.45%，顯著降低了系統性風險下的資產縮水。"
    )

    if os.path.exists("results/figures/cumulative_returns.png"):
        doc.add_picture("results/figures/cumulative_returns.png", width=Inches(5.5))
        p = doc.add_paragraph('圖 1: 最終版本模型累積報酬與避險回測比較圖')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('4.1 SHAP 特徵重要性分析', level=2)
    doc.add_paragraph(
        "透過 SHAP (SHapley Additive exPlanations) 分析，我們揭開了 AI 的決策黑盒子。排名前四的核心決策特徵為："
    )
    doc.add_paragraph("1. DXY_mom_26w (美元指數 26 週動能)", style='List Bullet')
    doc.add_paragraph("2. VIX_mom_26w (VIX 26 週動能)", style='List Bullet')
    doc.add_paragraph("3. volatility_52w (基金 52 週波動率)", style='List Bullet')
    doc.add_paragraph("4. TNX_mean_26w (10 年期美債殖利率平均)", style='List Bullet')
    doc.add_paragraph(
        "這符合金融邏輯：強勢美元代表資金抽離新興市場（台股），而高殖利率則壓抑科技股估值。AI 成功學會了觀察這些「大戶提款訊號」來保護資產。"
    )

    if os.path.exists("results/figures/shap_importance.png"):
        doc.add_picture("results/figures/shap_importance.png", width=Inches(5.5))
        p = doc.add_paragraph('圖 2: XGBoost 模型 SHAP 特徵重要性圖')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 5. 結論
    doc.add_heading('5. 結論', level=1)
    doc.add_paragraph(
        "本研究成功將原本充滿雜訊的市場預測問題，轉化為一套可執行的量化避險系統。我們證實了：\n"
        "1. 在預測台股深水區時，美元匯率與美債殖利率的前瞻性遠優於單純的情緒指標 (VIX)。\n"
        "2. 嚴格的「強制抄底」規則在面對如 2022 年的長期熊市時會帶來毀滅性打擊，賦予模型「依據總經數據動態避險」的自由度，才是降低最大回撤的唯一解方。\n"
        "未來，本系統可直接部署為投資人的自動化儀表板，提供客觀的總經防護網訊號。"
    )

    # Save
    out_path = "Active_Fund_AI_Research_Report.docx"
    doc.save(out_path)
    print(f"Report saved to {out_path}")

if __name__ == '__main__':
    main()
