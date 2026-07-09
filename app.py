import html as _html
import io
import os
import pickle
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.step4_analyze import analyze_tables
from src.step1_upload import parse_csv, parse_excel
from src.models import AIAnalysisResult, DetectedTable
from src.step3_format import UNIT_VOCAB, _is_agg_label
from src.step5_generate import (
    find_latent_tables,
    derive_latent_tables,
    LatentTableGroup,
    group_latent_proposals,
)
from src.models import DerivedLatentTable, IntegrationRecommendation

load_dotenv()

# ---------------------------------------------------------------------------
# ページ設定 & CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="テーブル抽出アプリ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    /* ── Badges ── */
    .step-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-left: 6px;
    }
    .badge-detail     { background:#d4edda; color:#155724; }
    .badge-summary    { background:#cce5ff; color:#004085; }
    .badge-master     { background:#fff3cd; color:#856404; }
    .badge-integrated { background:#e2d9f3; color:#3d1a78; }
    .badge-ref        { background:#f8d7da; color:#721c24; }

    /* ── General ── */
    .stDataFrame { font-size: 0.85rem; }
    div[data-testid="stExpander"] { border: 1px solid #dee2e6; border-radius: 8px; margin-bottom: 8px; }

    /* ── Fullscreen dataframe fills modal ── */
    [data-testid="stFullScreenFrame"] [data-testid="stDataFrame"] > div:first-child {
        height: calc(100vh - 100px) !important;
        max-height: none !important;
    }

    /* ── Hide Streamlit default header / footer (keep #MainMenu for theme toggle) ── */
    header[data-testid="stHeader"] {
        height: 0 !important;
        overflow: hidden !important;
        background: transparent !important;
    }
    /* ツールバー・Deploy ボタン・3点メニューを完全非表示 */
    [data-testid="stToolbar"],
    [data-testid="stToolbarActions"],
    [data-testid="stToolbarAction"],
    [data-testid="stDeployButton"],
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    button[kind="deployButton"],
    button[title*="Deploy"],
    button[aria-label*="Deploy"],
    .stDeployButton,
    #MainMenu {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    footer { visibility: hidden !important; }

    /* ── Remove default top padding (both old and new Streamlit selectors) ── */
    .block-container,
    [data-testid="stMainBlockContainer"] { padding-top: 0 !important; }

    /* ── Disable scroll anchoring so buildFixedHeader's padding-top doesn't
       cause the browser to auto-scroll back when adding header space ── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main { overflow-anchor: none !important; }

    /* Hidden splitter iframes: keep JS alive but remove from flex layout so
       they do not contribute gap spacing below the fixed header.
       position:absolute takes them out of the normal flow while allowing
       the iframe JS to keep running (unlike display:none). */
    .element-container:has(iframe[height="42"]),
    div[data-testid="stCustomComponentV1"]:has(iframe[height="42"]) {
        position: absolute !important;
        height: 0 !important;
        min-height: 0 !important;
        overflow: hidden !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* ── Progress dots ── */
    .app-progress-wrap { position:relative; margin-top:0.4rem; padding:4px 0; }
    .app-progress-bg-line, .app-progress-fill-line {
        position:absolute; top:50%; height:2px;
        border-radius:999px; transform:translateY(-50%); pointer-events:none;
    }
    .app-progress-bg-line  { background:rgba(127,255,212,0.14); }
    .app-progress-fill-line { background:rgba(127,255,212,0.55); }
    .app-progress-dots {
        position:relative; display:flex; height:18px; align-items:center; z-index:1;
    }
    .app-pd-slot { flex:1; display:flex; justify-content:center; align-items:center; }
    .app-pd-dot  { border-radius:50%; flex-shrink:0; }
    .app-pd-dot.pd-done   { width:8px; height:8px; background:rgba(127,255,212,0.65); }
    .app-pd-dot.pd-curr   { width:12px; height:12px;
                             background:#7FFFD4;
                             animation:dot-glow 2.8s ease-in-out infinite; }
    .app-pd-dot.pd-future { width:8px; height:8px;
                             background:rgba(127,255,212,0.13);
                             border:1.5px solid rgba(127,255,212,0.28); }
    @keyframes dot-glow {
        0%,100% { background:rgba(127,255,212,0.55); box-shadow:0 0 3px 1px rgba(127,255,212,0.15); }
        50%      { background:rgba(127,255,212,1.0);  box-shadow:0 0 7px 2px rgba(127,255,212,0.45); }
    }

    /* ── Styles for the JS-injected fixed header (id=_appFixedHdr) ── */
    #_appFixedHdr {
        position: fixed  !important;
        top: 0           !important;
        left: 0          !important;
        right: 0         !important;
        z-index: 99999   !important;
        background: rgb(14, 17, 23) !important;
        padding: 0.5rem 2rem 0.25rem !important;
        border-bottom: 1px solid #2d333b !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5) !important;
        box-sizing: border-box !important;
        color: #fafafa !important;
    }
    #_appFixedHdr p, #_appFixedHdr span, #_appFixedHdr label,
    #_appFixedHdr h1, #_appFixedHdr h2, #_appFixedHdr h3 {
        color: #fafafa !important;
    }
    /* Save ボタンスタイル（JS で portal に直接 position:absolute で配置される） */
    #_appFixedHdr [data-testid="stDownloadButton"] button,
    #_appFixedHdr .hdr-save-wrap button {
        padding: 3px 12px !important;
        min-height: 1.9rem !important;
        white-space: nowrap !important;
    }
    /* ── ヘッダー内ステップタブ: ピル型・半透明モダンデザイン ── */
    #_appFixedHdr div[data-testid="stHorizontalBlock"] button {
        padding: 3px 12px !important;
        border-radius: 999px !important;
        min-height: 1.9rem !important;
        letter-spacing: 0.01em !important;
        transition: background 0.15s, border-color 0.15s, color 0.15s !important;
    }
    /* 現在のステップ（primary）: 半透明グラス風 */
    #_appFixedHdr button[data-testid="stBaseButton-primary"] {
        background: rgba(127,255,212,0.18) !important;
        border: 1.5px solid rgba(127,255,212,0.75) !important;
        color: #7FFFD4 !important;
        font-weight: 700 !important;
    }
    /* 完了 / アクセス可能ステップ（secondary）*/
    #_appFixedHdr button[data-testid="stBaseButton-secondary"] {
        background: rgba(127,255,212,0.1) !important;
        border-color: rgba(127,255,212,0.45) !important;
        color: rgba(127,255,212,0.85) !important;
    }
    #_appFixedHdr button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(127,255,212,0.2) !important;
        border-color: rgba(127,255,212,0.75) !important;
        color: #7FFFD4 !important;
    }
    /* 未解放ステップ（disabled）: 背景は同じ半透明、文字色だけ暗くして区別 */
    /* button[disabled] だけでなく内部の span/p にも直接適用（#_appFixedHdr span の白が勝つため）*/
    #_appFixedHdr button[disabled],
    #_appFixedHdr button:disabled {
        background: rgba(127,255,212,0.07) !important;
        border-color: rgba(127,255,212,0.25) !important;
        color: rgba(170,170,170,0.8) !important;
        cursor: default !important;
        opacity: 1 !important;
    }
    #_appFixedHdr button[disabled] *,
    #_appFixedHdr button:disabled * {
        color: rgba(170,170,170,0.8) !important;
    }

    /* ── コンテンツエリアのボタン ──
       テキスト色は var(--text-color) でダーク/ライト自動対応。
       ダーク: 白文字 × アクアマリンボーダー、ライト: 黒文字 × アクアマリンボーダー。 */
    button[data-testid="stBaseButton-primary"] {
        background: rgba(127,255,212,0.18) !important;
        color: var(--text-color) !important;
        border: 1.5px solid rgba(127,255,212,0.6) !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        backdrop-filter: blur(6px) !important;
        transition: background 0.15s ease, border-color 0.15s ease, transform 0.15s ease !important;
        padding: 0.45rem 1.5rem !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background: rgba(127,255,212,0.32) !important;
        border-color: rgba(127,255,212,0.9) !important;
        transform: translateY(-1px) !important;
    }
    button[data-testid="stBaseButton-secondary"] {
        background: rgba(127,255,212,0.07) !important;
        border: 1.5px solid rgba(127,255,212,0.4) !important;
        color: var(--text-color) !important;
        border-radius: 999px !important;
        font-weight: 500 !important;
        backdrop-filter: blur(6px) !important;
        transition: background 0.15s ease, border-color 0.15s ease, transform 0.15s ease !important;
        padding: 0.45rem 1.2rem !important;
        opacity: 0.8 !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(127,255,212,0.16) !important;
        border-color: rgba(127,255,212,0.7) !important;
        opacity: 1 !important;
        transform: translateY(-1px) !important;
    }

    /* ── Hide the splitter JS iframe (height=42 is our unique marker) ── */
    .element-container:has(iframe[height="42"]),
    div[data-testid="stCustomComponentV1"]:has(iframe[height="42"]) {
        height: 0 !important;
        min-height: 0 !important;
        overflow: hidden !important;
        padding: 0 !important;
        margin: 0 !important;
    }

</style>
<script>
(function () {
    var HIDE_SELECTORS = [
        '[data-testid="stToolbar"]',
        '[data-testid="stToolbarActions"]',
        '[data-testid="stToolbarAction"]',
        '[data-testid="stDeployButton"]',
        '[data-testid="stStatusWidget"]',
        '#MainMenu',
        'button[kind="deployButton"]',
        'button[title*="Deploy"]',
        'button[aria-label*="Deploy"]',
    ];
    var _HIDE_CSS = 'display:none!important;visibility:hidden!important;width:0!important;height:0!important;overflow:hidden!important;pointer-events:none!important;';
    function hideToolbar() {
        HIDE_SELECTORS.forEach(function (sel) {
            document.querySelectorAll(sel).forEach(function (el) {
                el.style.cssText = _HIDE_CSS;
            });
        });
    }
    hideToolbar();
    var obs = new MutationObserver(hideToolbar);
    obs.observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Steps インポート
# ---------------------------------------------------------------------------

from steps.shared import (
    STEP_LABELS, _SCROLL_TO_TOP_JS,
    _init, _go_to, _render_header, _inject_splitter_js,
)
from steps.step1_upload import step1
from steps.step2_detect import step2
from steps.step3_format import step_format
from steps.step4_analyze import step3
from steps.step5_generate import step4
from steps.step6_select import step5
from steps.step7_export import step6


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------


def main():
    _init()

    # _scroll_to_top フラグを保存してから pop（後でステップ描画後に発火するため）
    scroll_to_top = st.session_state.pop("_scroll_to_top", False)

    with st.container():
        _render_header()

    # stepの関数より前にポータル同期を発火し、Step 3のAI呼び出しのような長時間処理中でも
    # （main()の末尾の_inject_splitter_js()より前にst.spinnerで中間状態が送られる場合も）
    # ヘッダーが即座に更新されるようにする。
    _inject_splitter_js()

    step = st.session_state.step
    if step == 1:
        step1()
    elif step == 2:
        step2()
    elif step == 3:
        step_format()
    elif step == 4:
        step3()
    elif step == 5:
        step4()
    elif step == 6:
        step5()
    elif step == 7:
        step6()
    _inject_splitter_js()

    # ステップコンテンツ全描画後にスクロールトップを発火する。
    # コンテンツ描画前に発火すると、その後の DOM 追加でブラウザが位置を戻してしまう。
    if scroll_to_top:
        components.html(_SCROLL_TO_TOP_JS, height=42)


if __name__ == "__main__":
    main()
