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

from src.relation_analyzer import analyze_tables
from src.excel_parser import parse_csv, parse_excel
from src.models import AIAnalysisResult, DetectedTable
from src.table_formatter import UNIT_VOCAB, _is_agg_label
from src.latent_table_detector import (
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
    /* Deploy button — exhaustive cross-platform selectors */
    [data-testid="stDeployButton"],
    [data-testid="stToolbar"] [data-testid="stDeployButton"],
    [data-testid="stToolbar"] > button,
    button[kind="deployButton"],
    button[title*="Deploy"],
    button[aria-label*="Deploy"],
    .stDeployButton,
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    #MainMenu {
        visibility: visible !important;
        position: fixed !important;
        top: 22px !important;
        right: 10px !important;
        z-index: 10001 !important;
    }
    /* ダークヘッダー上では常に白アイコン */
    #MainMenu button { color: rgba(255,255,255,0.75) !important; }
    #MainMenu button:hover { color: #ffffff !important; }
    #MainMenu button svg { fill: rgba(255,255,255,0.75) !important; }
    #MainMenu button:hover svg { fill: #ffffff !important; }
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
    var DEPLOY_SELECTORS = [
        '[data-testid="stDeployButton"]',
        'button[kind="deployButton"]',
        'button[title*="Deploy"]',
        'button[aria-label*="Deploy"]',
    ];
    function hideDeployBtn() {
        DEPLOY_SELECTORS.forEach(function (sel) {
            document.querySelectorAll(sel).forEach(function (el) {
                el.style.cssText = 'display:none!important;visibility:hidden!important;width:0!important;height:0!important;overflow:hidden!important;';
            });
        });
        // Also hide toolbar buttons that are not inside #MainMenu
        var toolbar = document.querySelector('[data-testid="stToolbar"]');
        if (toolbar) {
            toolbar.querySelectorAll('button').forEach(function (btn) {
                if (!btn.closest('#MainMenu')) {
                    btn.style.cssText = 'display:none!important;visibility:hidden!important;width:0!important;height:0!important;overflow:hidden!important;';
                }
            });
        }
    }
    hideDeployBtn();
    var obs = new MutationObserver(hideDeployBtn);
    obs.observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state の初期化
# ---------------------------------------------------------------------------

STEP_LABELS = [
    "ファイル選択",
    "テーブル検出",
    "テーブル整形",
    "テーブル関係分析",
    "新規テーブル案生成",
    "テーブル選択",
    "エクスポート",
]


def _init():
    defaults = {
        "step": 1,
        "run_mode": "manual",  # "manual" | "semiauto" | "fullauto"
        "auto_processing": False,  # True only during the forward auto-run pass
        "auto_completed": False,   # True when auto-run naturally reached its stop point
        "source_mode": None,  # "new_file" | "project" — how data was loaded
        "file_content": None,
        "filename": None,
        "file_ext": None,
        "detected_tables": [],
        "sheet_names": [],
        "ai_analysis": None,
        "integration_decisions": {},  # rec_id -> bool
        "master_decisions": {},  # dim_master_{rec_id} -> bool
        "derived_decisions": {},  # dlt_id -> bool (legacy; superseded by latent_group_decisions)
        "latent_group_decisions": {},  # group_key -> bool
        "latent_auto_int_decisions": {},  # auto_ir rec_id -> bool
        "final_tables": {},  # table_id -> dict
        "selected_ids": set(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


def _go_to(step: int, stop_auto: bool = True) -> None:
    """ナビゲーションコールバック — 次のレンダリング前に実行されるため、
    クリック後の最初のrerunで常に正しいアクティブタブがヘッダーに表示される。"""
    if stop_auto:
        st.session_state.auto_processing = False
    st.session_state.step = step
    st.session_state._scroll_to_top = True


def _build_and_go_step6() -> None:
    _build_final_tables()
    st.session_state.step = 6
    st.session_state._scroll_to_top = True


# ---------------------------------------------------------------------------
# プロジェクトの保存 / 読み込み
# ---------------------------------------------------------------------------

_PROJECT_VERSION = "1.0"
_PROJECT_DIR = Path(__file__).parent / "project"

_SAVE_KEYS = [
    "filename",
    "file_ext",
    "run_mode",
    "detected_tables",
    "sheet_names",
    "ai_analysis",
    "integration_decisions",
    "master_decisions",
    "derived_decisions",
    "latent_group_decisions",
    "latent_auto_int_decisions",
    "final_tables",
    "selected_ids",
    "step",
]


def _serialize_project() -> bytes:
    """現在のsession stateを.tepプロジェクトblobにPickle化する。"""
    payload = {
        "__tep_version__": _PROJECT_VERSION,
        "__saved_at__": datetime.now().isoformat(timespec="seconds"),
    }
    for k in _SAVE_KEYS:
        payload[k] = st.session_state.get(k)
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _save_project_to_disk() -> str:
    """現在のセッションを_PROJECT_DIRに保存する。ステータスメッセージを返す。"""
    try:
        _PROJECT_DIR.mkdir(parents=True, exist_ok=True)
        fname = st.session_state.get("filename", "project")
        stem = Path(fname).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = _PROJECT_DIR / f"{stem}_{ts}.tep"
        out_path.write_bytes(_serialize_project())
        return f"✅ 保存しました: `{out_path.name}`"
    except Exception as e:
        return f"❌ 保存に失敗しました: {e}"


def _load_project(raw: bytes) -> str:
    """.tepblobからsession stateを復元する。ステータスメッセージを返す。"""
    try:
        payload = pickle.loads(raw)
    except Exception as e:
        return f"❌ ファイルの読み込みに失敗しました: {e}"

    ver = payload.get("__tep_version__")
    if ver != _PROJECT_VERSION:
        return f"❌ バージョン不一致 (保存: {ver}, 現在: {_PROJECT_VERSION})"

    for k in _SAVE_KEYS:
        if k in payload:
            st.session_state[k] = payload[k]

    # 復元時はauto_processingをオフにし、ソースを記録する
    st.session_state["auto_processing"] = False
    st.session_state["source_mode"] = "project"
    return f"✅ プロジェクトを復元しました（保存日時: {payload.get('__saved_at__', '不明')}）"


# ---------------------------------------------------------------------------
# 左右リサイズ可能スプリッター
# ---------------------------------------------------------------------------


def _splitter_marker(split_id: str) -> None:
    """JSが直後のst.columns()を特定するために使う不可視マーカー。"""
    st.markdown(
        f'<div class="split-init-marker" data-split-id="{split_id}"></div>',
        unsafe_allow_html=True,
    )


def _inject_splitter_js() -> None:
    """ページ上のすべてのsplit-init-markerにドラッグリサイズ機能を注入する。

    height=42を一意のタグとして使用し、CSSルールがiframeラッパーを0pxに
    折りたたむ一方でスクリプトが動き続けるようにする。
    """
    components.html(
        """
        <script>
        (function () {
            var pdoc = window.parent.document;
            var win  = window.parent;

            /* ── One-time global mouse listeners ── */
            if (!win._splitDragState) {
                win._splitDragState = null;

                pdoc.addEventListener('mousemove', function (e) {
                    var s = win._splitDragState;
                    if (!s) return;
                    var w = s.container.getBoundingClientRect().width;
                    if (!w) return;
                    var pct = Math.max(15, Math.min(82,
                        s.startLeft + (e.clientX - s.startX) / w * 100));
                    s.leftCol.style.flex = '0 0 ' + pct.toFixed(1) + '%';
                    localStorage.setItem('split-' + s.id, pct.toFixed(1));
                });

                pdoc.addEventListener('mouseup', function () {
                    var s = win._splitDragState;
                    if (!s) return;
                    s.handle.style.background  = '#2d333b';
                    s.handle.style.borderColor = '#444';
                    s.dots.style.color         = '#666';
                    pdoc.body.style.userSelect = '';
                    pdoc.body.style.cursor     = '';
                    win._splitDragState = null;
                });
            }

            /* ── Per-splitter setup ── */
            function setup(hBlock, id) {
                if (hBlock.dataset.splitDone) return;
                hBlock.dataset.splitDone = '1';

                var cols = Array.from(
                    hBlock.querySelectorAll(':scope > [data-testid="stColumn"]'));
                if (cols.length < 2) return;

                var L = cols[0], R = cols[1];
                var saved = parseFloat(localStorage.getItem('split-' + id) || '50');

                /* Insert drag handle between L and R */
                var handle = pdoc.createElement('div');
                handle.style.cssText =
                    'flex:0 0 10px;min-width:10px;width:10px;padding:0;cursor:col-resize;' +
                    'background:#2d333b;border:1px solid #444;border-radius:4px;' +
                    'display:flex;align-items:center;justify-content:center;' +
                    'transition:background 0.15s,border-color 0.15s;box-sizing:border-box;';

                var dots = pdoc.createElement('span');
                dots.textContent = '⋮';
                dots.style.cssText =
                    'color:#666;font-size:18px;line-height:1;pointer-events:none;' +
                    'user-select:none;display:block';
                handle.appendChild(dots);
                hBlock.insertBefore(handle, R);

                /* flex layout */
                hBlock.style.display   = 'flex';
                hBlock.style.flexWrap  = 'nowrap';
                hBlock.style.alignItems = 'stretch';
                L.style.flex    = '0 0 ' + saved + '%';
                L.style.minWidth = '0';
                L.style.overflow = 'hidden';

                R.style.flex     = '1 1 0';
                R.style.minWidth = '0';
                R.style.overflow = 'hidden';

                /* hover */
                handle.addEventListener('mouseenter', function () {
                    if (win._splitDragState) return;
                    handle.style.background  = 'rgba(127,255,212,0.15)';
                    handle.style.borderColor = '#7FFFD4';
                    dots.style.color         = '#7FFFD4';
                });
                handle.addEventListener('mouseleave', function () {
                    if (win._splitDragState) return;
                    handle.style.background  = '#2d333b';
                    handle.style.borderColor = '#444';
                    dots.style.color         = '#666';
                });

                /* drag start */
                handle.addEventListener('mousedown', function (e) {
                    var cur = parseFloat(L.style.flex.split(' ').pop()) || saved;
                    win._splitDragState = {
                        container: hBlock, leftCol: L,
                        handle: handle, dots: dots, id: id,
                        startX: e.clientX, startLeft: cur
                    };
                    handle.style.background  = 'rgba(127,255,212,0.25)';
                    handle.style.borderColor = '#7FFFD4';
                    dots.style.color         = '#7FFFD4';
                    pdoc.body.style.userSelect = 'none';
                    pdoc.body.style.cursor     = 'col-resize';
                    e.preventDefault();
                });
            }

            /* ── Body-level portal header ──
               position:fixed on a direct child of <body> is ALWAYS viewport-relative,
               unaffected by Streamlit's overflow/transform/will-change on inner elements.
               Strategy:
                 1. Create a <div id="_appFixedHdr"> appended to pdoc.body.
                 2. Copy innerHTML from the Streamlit-rendered srcHdr into it.
                 3. Forward button-clicks from the portal to the real Streamlit buttons
                    in srcHdr so Streamlit state updates still fire.
                 4. Collapse srcHdr to zero-height so it occupies no space.
                 5. Add padding-top to rootBlock so step content starts below the header. */
            function buildFixedHeader() {
                /* Use sentinel to find the exact header stVerticalBlock.
                   rootBlock.firstElementChild would target the wrong element when
                   st.markdown(style) runs before main() and creates a sibling. */
                var sentinel = pdoc.querySelector('.app-hdr-sentinel');
                if (!sentinel) return; /* not rendered yet */

                var srcHdr = sentinel.closest('[data-testid="stVerticalBlock"]');
                if (!srcHdr) return;

                /* rootBlock = the parent stVerticalBlock of srcHdr */
                var rootBlock = srcHdr.parentElement
                    ? srcHdr.parentElement.closest('[data-testid="stVerticalBlock"]')
                    : null;
                if (!rootBlock) return;

                /* ── 1. Create portal div at body level (once) ── */
                var FID = '_appFixedHdr';
                var portal = pdoc.getElementById(FID);
                if (!portal) {
                    portal = pdoc.createElement('div');
                    portal.id = FID;
                    pdoc.body.appendChild(portal);
                }

                /* ── 2. Sync visual content ── */
                portal.innerHTML = srcHdr.innerHTML;
                /* Remove duplicate IDs to prevent CSS/JS selector conflicts */
                Array.from(portal.querySelectorAll('[id]')).forEach(function (el) {
                    el.removeAttribute('id');
                });

                /* ── 2b. Move Save button to position:absolute at top-right ──
                   Search every <button> in the portal for the Save button by text,
                   then move its outermost Streamlit wrapper to an absolutely-positioned
                   child div so it renders next to the ⋮ menu. ── */
                (function() {
                    var allBtns = portal.querySelectorAll('button');
                    var saveBtn = null;
                    for (var i = 0; i < allBtns.length; i++) {
                        var t = allBtns[i].textContent;
                        if (t.indexOf('Save') !== -1) { saveBtn = allBtns[i]; break; }
                    }
                    if (!saveBtn) return;
                    /* Walk up to the first div with data-testid (the Streamlit wrapper) */
                    var saveEl = saveBtn.parentElement;
                    while (saveEl && saveEl !== portal) {
                        if (saveEl.hasAttribute('data-testid')) break;
                        saveEl = saveEl.parentElement;
                    }
                    if (!saveEl || saveEl === portal) saveEl = saveBtn.parentElement;
                    var saveWrap = pdoc.createElement('div');
                    saveWrap.className = 'hdr-save-wrap';
                    saveWrap.style.cssText =
                        'position:absolute;top:14px;right:52px;z-index:2;';
                    portal.appendChild(saveWrap);
                    saveWrap.appendChild(saveEl);
                })();

                /* ── 3. Forward clicks to real Streamlit buttons in srcHdr ── */
                portal.onclick = function (e) {
                    var btn = e.target && e.target.closest('button');
                    if (!btn) return;
                    var txt = btn.textContent.trim();
                    var realBtns = Array.from(srcHdr.querySelectorAll('button'));
                    for (var i = 0; i < realBtns.length; i++) {
                        if (realBtns[i].textContent.trim() === txt) {
                            /* Use dispatchEvent so React's synthetic event system
                               receives a properly formed MouseEvent even when the
                               target lives off-screen at position: fixed / left:-9999px.
                               composed:true lets it cross any shadow-DOM boundaries. */
                            realBtns[i].dispatchEvent(
                                new MouseEvent('click', {
                                    bubbles: true, cancelable: true,
                                    view: window.parent, composed: true
                                })
                            );
                            break;
                        }
                    }
                    e.preventDefault();
                };

                /* ── 4. Remove srcHdr from flex flow.
                   position:fixed is enough to take it out of flow entirely —
                   the element stays off-screen at -9999px and its buttons
                   remain full-size and interactable so JS .click() reliably
                   triggers Streamlit's React event system.
                   IMPORTANT: do NOT set width/height/overflow/visibility here —
                   those collapse the button hit-areas and break programmatic
                   click dispatch on the hidden real buttons. ── */
                srcHdr.style.setProperty('position', 'fixed',   'important');
                srcHdr.style.setProperty('left',     '-9999px', 'important');
                srcHdr.style.setProperty('top',      '-9999px', 'important');

                /* ── 4b. Walk every ancestor of rootBlock up to <body> and zero
                   out padding-top + margin-top.  This works regardless of which
                   CSS class or data-testid Streamlit uses for the main container
                   in any given version — no selector guessing required. ── */
                var anc = rootBlock.parentElement;
                while (anc && anc !== pdoc.body && anc !== pdoc.documentElement) {
                    anc.style.setProperty('padding-top', '0', 'important');
                    anc.style.setProperty('margin-top',  '0', 'important');
                    anc = anc.parentElement;
                }
                rootBlock.style.setProperty('margin-top', '0', 'important');

                /* ── 5. Push rootBlock content below the portal header ── */
                var ph = portal.getBoundingClientRect().height;
                if (ph > 0) rootBlock.style.setProperty('padding-top', ph + 'px', 'important');
            }

            /* ── Progress dots: width is set via inline style from Python; no JS needed ── */
            function animateProgress() { /* no-op: replaced by CSS dot animation */ }

            /* ── Light mode: 背景を薄グレーに ──
               インラインスタイルではなく <style> タグを head に注入する。
               Streamlit が rerun で inline style を書き戻しても、
               head の <style> はカスケードで上位に残り続ける。 */
            function applyGreyBg() {
                var grey   = '#f0f2f5';
                var sid    = '_appGreyBgStyle';
                var bg     = getComputedStyle(pdoc.documentElement)
                                 .getPropertyValue('--background-color')
                                 .trim().replace(/\s+/g, '');
                var isLight = bg === '#ffffff' || bg === '#fff' ||
                              bg === 'rgb(255,255,255)';

                if (isLight) {
                    pdoc.body.classList.add('app-light-mode');
                    if (!pdoc.getElementById(sid)) {
                        var s = pdoc.createElement('style');
                        s.id  = sid;
                        s.textContent =
                            '.stApp,[data-testid="stAppViewContainer"],' +
                            '[data-testid="stMainBlockContainer"]{' +
                            'background-color:' + grey + '!important;}';
                        pdoc.head.appendChild(s);
                    }
                } else {
                    pdoc.body.classList.remove('app-light-mode');
                    var old = pdoc.getElementById(sid);
                    if (old) old.remove();
                }
            }

            /* ── Find markers and wire up ── */
            function init() {
                buildFixedHeader();
                animateProgress();
                applyGreyBg();
                var markers = pdoc.querySelectorAll(
                    '.split-init-marker:not([data-split-done])');
                markers.forEach(function (marker) {
                    marker.setAttribute('data-split-done', '1');
                    var id = marker.getAttribute('data-split-id');
                    var vb = marker.closest('[data-testid="stVerticalBlock"]');
                    if (!vb) return;
                    var wrap = marker;
                    while (wrap.parentElement && wrap.parentElement !== vb) {
                        wrap = wrap.parentElement;
                    }
                    var next = wrap.nextElementSibling;
                    while (next) {
                        var hb = next.querySelector('[data-testid="stHorizontalBlock"]');
                        if (hb) { setup(hb, id); break; }
                        next = next.nextElementSibling;
                    }
                });
            }

            [0, 80, 250, 600, 1400].forEach(function (ms) {
                setTimeout(init, ms);
            });

            /* ── MutationObserver: keep portal in sync whenever srcHdr changes ──
               Without this, navigating while a long step (e.g. Step 3 AI call) is
               running leaves the portal showing the previous step until the next
               _inject_splitter_js() call fires at the very end of main().
               Debounced at 50 ms so rapid DOM mutations batch into one sync. */
            function connectObserver() {
                var sentinel = pdoc.querySelector('.app-hdr-sentinel');
                if (!sentinel) return;
                var node = sentinel.closest('[data-testid="stVerticalBlock"]');
                if (!node || win._hdrObservedNode === node) return;
                if (win._hdrObserver) win._hdrObserver.disconnect();
                win._hdrObservedNode = node;
                win._hdrObserver = new MutationObserver(function () {
                    clearTimeout(win._hdrSyncTimer);
                    win._hdrSyncTimer = setTimeout(function () {
                        buildFixedHeader();
                        animateProgress();
                    }, 50);
                });
                win._hdrObserver.observe(node, {
                    childList: true, subtree: true,
                    attributes: true, characterData: true
                });
            }
            [0, 80, 250, 600, 1400].forEach(function (ms) {
                setTimeout(connectObserver, ms);
            });
        })();
        </script>
        """,
        height=42,
    )


# ---------------------------------------------------------------------------
# ヘッダー & ステップインジケーター
# ---------------------------------------------------------------------------


def _can_navigate_to(target: int) -> bool:
    """対象ステップに必要なデータが揃っている場合True を返す。"""
    s = st.session_state
    if target == 1:
        return True
    if target == 2:
        return bool(s.get("file_content")) or bool(s.get("detected_tables"))
    if target == 3:
        return bool(s.get("detected_tables"))
    if target == 4:
        return bool(s.get("detected_tables"))
    if target == 5:
        return s.get("ai_analysis") is not None
    if target == 6:
        return bool(s.get("final_tables"))
    if target == 7:
        return bool(s.get("selected_ids"))
    return False


@st.dialog("💾 プロジェクトを保存")
def _save_dialog():
    default_name = Path(st.session_state.get("filename", "project")).stem
    save_name = st.text_input(
        "ファイル名",
        value=default_name,
        placeholder="ファイル名を入力",
        help=".tep 拡張子は自動で付加されます",
    )
    st.caption("保存された `.tep` ファイルは Step 1「プロジェクト読込」で再開できます。")
    file_name = f"{save_name.strip() or default_name}.tep"
    st.download_button(
        f"⬇️ {file_name} としてダウンロード",
        data=_serialize_project(),
        file_name=file_name,
        mime="application/octet-stream",
        use_container_width=True,
        type="primary",
    )


def _render_header():
    # st.container() で専用 stVerticalBlock を作成し、JS の sentinel.closest() が
    # ヘッダー専用ブロックを確実に特定できるようにする（ページ全体の外側ブロックを
    # 誤ってキャプチャしてコンテンツが被るのを防ぐ）。
    with st.container():
        # センチネル: JSがヘッダーのstVerticalBlockを確実に特定するために使用
        st.markdown(
            '<span class="app-hdr-sentinel" style="display:none"></span>',
            unsafe_allow_html=True,
        )
        st.title("📊 Table Extractor (開発中)")
        st.caption("Excel / CSV ファイルから分析対象とするテーブルを抽出します。")
        if st.session_state.get("filename") and bool(st.session_state.get("detected_tables")):
            if st.button(
                "💾 Save",
                key="hdr_save_btn",
                help="現在の解析状態を .tep ファイルとしてダウンロードします。再開時はStep 1でアップロードしてください。",
            ):
                _save_dialog()

        current = st.session_state.step
        cols = st.columns(len(STEP_LABELS))

        for i, (col, label) in enumerate(zip(cols, STEP_LABELS), 1):
            with col:
                is_current = i == current
                is_done = i < current
                accessible = _can_navigate_to(i)

                if is_current:
                    # 現在のステップ — 緑色のボーダーでハイライト（ボタンではなくHTML div）
                    st.markdown(
                        f'<div style="'
                        f"background:rgba(127,255,212,0.18);"
                        f"border:1.5px solid rgba(127,255,212,0.75);"
                        f"border-radius:999px;"
                        f"text-align:center;"
                        f"padding:3px 12px;"
                        f"font-weight:700;"
                        f"letter-spacing:0.01em;"
                        f"color:#7FFFD4;"
                        f"min-height:1.9rem;"
                        f"display:flex;"
                        f"align-items:center;"
                        f"justify-content:center;"
                        f"box-sizing:border-box;"
                        f'">▶ {label}</div>',
                        unsafe_allow_html=True,
                    )
                elif is_done:
                    # 完了済みのステップ — クリック可能; 戻る際にauto_processingを停止
                    col.button(
                        f"✅ {label}",
                        key=f"nav_{i}",
                        on_click=_go_to,
                        args=(i,),
                        use_container_width=True,
                    )
                elif accessible:
                    # データが利用可能な将来のステップ — クリック可能
                    col.button(
                        f"○ {label}",
                        key=f"nav_{i}",
                        on_click=_go_to,
                        args=(i,),
                        use_container_width=True,
                    )
                else:
                    # 将来のステップ、データ未準備 — 無効
                    st.button(
                        f"○ {label}",
                        key=f"nav_{i}",
                        use_container_width=True,
                        disabled=True,
                    )

        n_steps = len(STEP_LABELS)
        slot_half = 100.0 / (2 * n_steps)          # half a slot width in %
        fill_w = 2 * (current - 1) * slot_half      # fill ends at current dot center
        bg_style   = f"left:{slot_half:.3f}%;right:{slot_half:.3f}%"
        fill_style = f"left:{slot_half:.3f}%;width:{fill_w:.3f}%"
        dots_html = "".join(
            f'<div class="app-pd-slot"><div class="app-pd-dot '
            f'{"pd-done" if i < current else "pd-curr" if i == current else "pd-future"}'
            f'"></div></div>'
            for i in range(1, n_steps + 1)
        )
        st.markdown(
            f'<div class="app-progress-wrap">'
            f'<div class="app-progress-bg-line" style="{bg_style}"></div>'
            f'<div class="app-progress-fill-line" style="{fill_style}"></div>'
            f'<div class="app-progress-dots">{dots_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 自動処理バナーはコンテナ外（固定ヘッダーのクローン対象外）に配置し
    # プログレスバーとの重なりを防ぐ。
    # run_mode で表示有無を判定することで、タブ移動後も消えないようにする。
    _mode = st.session_state.get("run_mode", "manual")
    _auto = st.session_state.get("auto_processing", False)
    _completed = st.session_state.get("auto_completed", False)
    if _mode in ("semiauto", "fullauto") and (_auto or _completed):
        _mode_label = "セミオート" if _mode == "semiauto" else "フルオート"
        if _auto:
            _step_label = (
                STEP_LABELS[current - 1] if 1 <= current <= len(STEP_LABELS) else ""
            )
            st.info(
                f"⚙️ **{_mode_label} 実行中** — "
                f"ステップ {current} / {len(STEP_LABELS)}「{_step_label}」を処理中...",
                icon=None,
            )
        else:
            st.success(
                f"✅ **{_mode_label} 完了** — 確認・選択後、手動で続けてください",
                icon=None,
            )


# ---------------------------------------------------------------------------
# Step 1 — ファイルアップロード
# ---------------------------------------------------------------------------


def _check_api_config() -> tuple:
    """環境変数からAPI認証情報を検証する。

    (is_valid: bool, label: str, error_msg: str, hint_code: str) を返す。
    """
    api_type = os.getenv("OPENAI_API_TYPE", "openai").strip().lower()

    if api_type == "azure":
        missing = []
        if not os.getenv("AZURE_OPENAI_API_KEY", "").strip():
            missing.append("AZURE_OPENAI_API_KEY")
        if not os.getenv("AZURE_OPENAI_ENDPOINT", "").strip():
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip():
            missing.append("AZURE_OPENAI_DEPLOYMENT")
        if missing:
            hint = "\n".join(f"{k}=..." for k in missing)
            return (
                False,
                "",
                f"Azure OpenAI の設定が不足しています: {', '.join(missing)}",
                hint,
            )
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        return True, f"Azure OpenAI / {deployment}", "", ""
    else:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            return False, "", "OPENAI_API_KEY が未設定です。", "OPENAI_API_KEY=sk-..."
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        return True, f"OpenAI / {model}", "", ""


def step1():
    st.header("📂 ステップ 1 : ファイルを選択")

    is_valid, label, error_msg, hint_code = _check_api_config()
    if not is_valid:
        st.error(
            f"⚠️ **{error_msg}**  "
            "プロジェクトルートの `.env` ファイルを確認してください。"
        )
        st.code(hint_code, language="bash")
        return

    st.success(f"✅ API キー確認済み ({label})")

    tab_new, tab_load = st.tabs(["📂 新規ファイル", "📁 プロジェクト読込"])

    # .tepファイルからセッションが復元された際にプロジェクト読込タブを自動選択する。
    # st.tabsはrerun時に常に最初のタブをデフォルトにするため、JSでボタンをクリックする。
    # 複数のリトライ遅延を使ってStreamlitのReactレンダリングのタイミング差異に対応する。
    if st.session_state.get("source_mode") == "project":
        components.html(
            """<script>
            (function () {
                var pdoc = window.parent.document;
                var done = false;
                function tryClick() {
                    if (done) return;
                    var lists = pdoc.querySelectorAll('[data-testid="stTabsTabList"]');
                    if (!lists.length) return;
                    var tabs = lists[0].querySelectorAll('button[role="tab"]');
                    if (tabs.length < 2) return;
                    done = true;
                    tabs[1].click();
                }
                [0, 50, 120, 250, 500, 1000].forEach(function(ms) {
                    setTimeout(tryClick, ms);
                });
            })();
            </script>""",
            height=1,
        )

    # ── Tab 1: 新規ファイル ───────────────────────────────────────────────────
    with tab_new:
        # 後のステップから戻ってきたとき、現在読み込み中のファイルを表示
        if (
            st.session_state.get("filename")
            and st.session_state.get("source_mode") == "new_file"
        ):
            st.info(
                f"📄 現在読み込み中: **{st.session_state.filename}**  "
                f"（{len(st.session_state.get('detected_tables', []))} テーブル検出済み）  \n"
                "新しいファイルをアップロードすると現在のデータは破棄されます。"
            )
            st.divider()

        # モード選択
        st.markdown("#### 実行モード")
        mode_labels = {
            "manual": "マニュアル  —  各ステップを手動で確認しながら進む",
            "semiauto": "セミオート  —  新規テーブル案生成まで自動実行し、確認・選択のみ手動で行う",
            "fullauto": "フルオート  —  推奨テーブルを自動選択してエクスポートまで完全自動実行",
        }
        mode_keys = list(mode_labels.keys())
        selected_mode = st.radio(
            "モードを選択してください",
            options=mode_keys,
            format_func=lambda k: mode_labels[k],
            index=mode_keys.index(st.session_state.run_mode),
            label_visibility="collapsed",
        )
        st.session_state.run_mode = selected_mode

        st.divider()

        if (
            st.session_state.get("filename")
            and st.session_state.get("detected_tables")
            and st.session_state.get("source_mode") == "new_file"
        ):
            st.button(
                "次へ：テーブル検出の結果を確認 →",
                type="primary",
                use_container_width=True,
                on_click=_go_to,
                args=(2,),
                key="s1_next_new",
            )
            st.divider()

        uploaded = st.file_uploader(
            "Excel または CSV ファイルを選択してください",
            type=["xlsx", "xlsm", "xls", "csv"],
            help="複数シート・複数テーブルを含む Excel ファイルに対応しています",
        )

        if uploaded:
            content = uploaded.getvalue()
            size_kb = len(content) / 1024
            ext = Path(uploaded.name).suffix.lower()

            c1, c2 = st.columns([3, 1])
            c1.markdown(
                f"<div style='font-size:0.75rem;color:#888;margin-bottom:2px'>ファイル名</div>"
                f"<div style='font-size:0.95rem;font-weight:600;text-align:left'>{uploaded.name}</div>",
                unsafe_allow_html=True,
            )
            c2.markdown(
                f"<div style='font-size:0.75rem;color:#888;margin-bottom:2px;text-align:right'>サイズ</div>"
                f"<div style='font-size:0.95rem;font-weight:600;text-align:right'>{size_kb:.1f} KB</div>",
                unsafe_allow_html=True,
            )

            btn_labels = {
                "manual": "テーブル検出を開始",
                "semiauto": "セミオートで開始",
                "fullauto": "フルオートで開始",
            }

            def _start_file(c, name, e, mode):
                st.session_state.file_content = c
                st.session_state.filename = name
                st.session_state.file_ext = e
                st.session_state.detected_tables = []
                st.session_state.ai_analysis = None
                st.session_state.final_tables = {}
                st.session_state.selected_ids = set()
                st.session_state.auto_processing = mode != "manual"
                st.session_state.auto_completed = False
                st.session_state.source_mode = "new_file"
                st.session_state.step = 2

            st.button(
                btn_labels[selected_mode],
                type="primary",
                use_container_width=True,
                on_click=_start_file,
                args=(content, uploaded.name, ext, selected_mode),
            )

    # ── Tab 2: プロジェクト読込 ────────────────────────────────────────────────
    with tab_load:
        # 後のステップから戻ってきたとき、現在読み込み済みのプロジェクトを表示
        if (
            st.session_state.get("filename")
            and st.session_state.get("source_mode") == "project"
        ):
            st.info(
                f"📁 読み込み済みプロジェクト: **{st.session_state.filename}**  "
                f"（{len(st.session_state.get('detected_tables', []))} テーブル / "
                f"Step {st.session_state.get('step', 1)} まで完了）  \n"
                "別のプロジェクトを読み込むと現在のデータは上書きされます。"
            )
            if st.session_state.get("detected_tables"):
                st.button(
                    "次へ：テーブル検出の結果を確認 →",
                    type="primary",
                    use_container_width=True,
                    on_click=_go_to,
                    args=(2,),
                    key="s1_next_proj",
                )
            st.divider()

        st.markdown("#### 保存済みプロジェクトを読み込む")
        st.caption(
            "以前の解析セッションで保存した `.tep` ファイルを選択すると、"
            "テーブル検出・AI 分析・統合設定などの結果をそのまま復元し、"
            "保存時のステップから再開できます。"
        )

        proj_file = st.file_uploader(
            ".tep プロジェクトファイルを選択",
            type=["tep"],
            key="project_uploader",
        )

        if proj_file is not None:
            msg = _load_project(proj_file.getvalue())
            if msg.startswith("✅"):
                st.success(msg)
                restored_step = st.session_state.get("step", 1)
                fname = st.session_state.get("filename", "（不明）")
                saved_at_label = ""
                try:
                    payload = pickle.loads(proj_file.getvalue())
                    saved_at_label = payload.get("__saved_at__", "")
                except Exception:
                    pass

                c1, c2, c3 = st.columns(3)
                c1.metric("元ファイル", fname)
                c2.metric("保存時ステップ", f"Step {restored_step}")
                c3.metric("保存日時", saved_at_label or "不明")

                if st.button(
                    "▶ このプロジェクトで再開", type="primary", use_container_width=True
                ):
                    st.rerun()
            else:
                st.error(msg)


# ---------------------------------------------------------------------------
# Step 2 — テーブル検出
# ---------------------------------------------------------------------------


def _get_original_df(t: "DetectedTable") -> "Optional[pd.DataFrame]":
    """ステップ2表示用: 整形処理適用前の生 DataFrame を返す。

    優先順位: 多段ヘッダー統合前 → ffill 前 → 集計除去前 → 最終 df
    """
    for candidate in [
        t.raw_df,
        getattr(t, "pre_fill_df", None),
        t.pre_agg_df,
        t.df,
    ]:
        if candidate is not None and not candidate.empty:
            return candidate
    return None


def step2():
    st.header("🔍 ステップ 2 : テーブル検出")

    if not st.session_state.detected_tables:
        with st.spinner("ファイルを解析中..."):
            try:
                ext = st.session_state.file_ext
                if ext == ".csv":
                    tables, sheets = parse_csv(
                        st.session_state.file_content, st.session_state.filename
                    )
                else:
                    tables, sheets = parse_excel(
                        st.session_state.file_content, st.session_state.filename
                    )
                st.session_state.detected_tables = tables
                st.session_state.sheet_names = sheets
                st.rerun()  # re-render header so Save button appears
            except Exception as e:
                st.error(f"❌ 解析エラー: {e}")
                return

    tables: List[DetectedTable] = st.session_state.detected_tables
    sheets: List[str] = st.session_state.sheet_names

    # 初回の前進パス中のみ自動的に次のステップへ進む（UI出力より前に実行）
    if st.session_state.auto_processing:
        st.session_state.step = 3
        st.rerun()

    st.success(
        f"✅ **{len(sheets)} シート** から **{len(tables)} テーブル** を検出しました"
    )

    # シートでグループ化
    by_sheet: Dict[str, List[DetectedTable]] = {}
    for t in tables:
        by_sheet.setdefault(t.sheet_name, []).append(t)

    # ── ツリービュー ───────────────────────────────────────────────────────
    tree_lines = [f"📁 {st.session_state.filename}"]
    for i, sheet in enumerate(sheets):
        sh_tables = by_sheet.get(sheet, [])
        cnt_str = f"{len(sh_tables)} テーブル" if sh_tables else "テーブルなし"
        is_last_sh = i == len(sheets) - 1
        sh_pfx = "└── " if is_last_sh else "├── "
        tree_lines.append(f"{sh_pfx}📋 {sheet}  ({cnt_str})")
        ch_pfx = "    " if is_last_sh else "│   "
        for j, t in enumerate(sh_tables):
            is_last_t = j == len(sh_tables) - 1
            t_pfx = ch_pfx + ("└── " if is_last_t else "├── ")
            dims = f"{t.row_count}行×{t.col_count}列"
            title_part = f"  [{t.title}]" if t.title else ""
            tree_lines.append(f"{t_pfx}📊 {t.table_id}  {dims}{title_part}")
    st.code("\n".join(tree_lines), language=None)

    # ── シートごとのexpander（デフォルトで折りたたみ） ─────────────────────
    for sheet in sheets:
        sheet_tables = by_sheet.get(sheet, [])
        cnt_str = f"{len(sheet_tables)} テーブル" if sheet_tables else "テーブルなし"
        label = f"📋  {sheet}  （{cnt_str}）"
        with st.expander(label, expanded=False):
            if not sheet_tables:
                st.info("このシートにはテーブルが検出されませんでした")
                continue
            for t in sheet_tables:
                title_str = f"  🏷️ `{t.title}`" if t.title else ""
                st.markdown(
                    f"**`{t.table_id}`**{title_str}  —  {t.row_count} 行 × {t.col_count} 列"
                    f"  （行 {t.start_row}〜{t.end_row}, 列 {t.start_col}〜{t.end_col}）"
                )
                orig = _get_original_df(t)
                if orig is not None:
                    st.dataframe(
                        orig.astype(str),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.divider()

    c1, c2 = st.columns([1, 4])
    with c1:
        st.button("← 戻る", on_click=_go_to, args=(1,))
    with c2:
        if not tables:
            st.warning("テーブルが検出されませんでした。別のファイルをお試しください。")
        else:
            st.button(
                "次へ：テーブル整形を確認 →",
                type="primary",
                use_container_width=True,
                on_click=_go_to,
                args=(3,),
            )


# ---------------------------------------------------------------------------
# Step 3 — テーブル整形
# ---------------------------------------------------------------------------


_TH_STYLE = (
    "position:sticky;top:0;z-index:2;"
    "background-color:var(--background-color,#0f1117);"
    "background-image:linear-gradient(rgba(66,153,225,0.20),rgba(66,153,225,0.20));"
    "color:var(--text-color,#fafafa);"
    "padding:6px 12px;"
    "text-align:left;"
    "border-bottom:2px solid rgba(66,153,225,0.5);"
    "white-space:nowrap;"
    "font-weight:600;"
    "font-size:13px;"
)
_TD_STYLE = (
    "padding:4px 12px;"
    "font-size:13px;"
    "border-bottom:1px solid rgba(255,255,255,0.06);"
)
_TD_CENTER_STYLE = (
    "padding:4px 10px;"
    "font-size:13px;"
    "text-align:center;"
    "white-space:nowrap;"
    "border-bottom:1px solid rgba(255,255,255,0.06);"
)


def _df_to_html(
    df: pd.DataFrame,
    max_height: Optional[int] = None,
    highlight_row_count: int = 0,
    highlight_row_indices: Optional[set] = None,
    highlight_col_names: Optional[set] = None,
    unit_col_names: Optional[set] = None,
    green_col_names: Optional[set] = None,
) -> str:
    """DataFrameをモダンなスタイルのHTMLテーブルに変換する。
    max_height を指定すると縦スクロール可能なコンテナで包む。
    highlight_row_count > 0 の場合、先頭 N 行を赤色強調表示する。
    highlight_row_indices: 赤色強調する行の位置インデックス集合。
    highlight_col_names: オレンジ色ヘッダーで示す除去列名集合。
    unit_col_names: 紫色ヘッダーで示す単位付加列名集合。
    green_col_names: 緑色ヘッダーで示す前方補完列名集合。"""
    col_names = list(df.columns)
    orange_pos: set = {
        j for j, c in enumerate(col_names)
        if highlight_col_names and str(c) in highlight_col_names
    }
    purple_pos: set = {
        j for j, c in enumerate(col_names)
        if unit_col_names and str(c) in unit_col_names
    }
    green_pos: set = {
        j for j, c in enumerate(col_names)
        if green_col_names and str(c) in green_col_names
    }

    def _th(j: int, c: str) -> str:
        label = _html.escape(str(c))
        if j in orange_pos:
            return (
                f"<th style='{_TH_STYLE}"
                f"background-image:linear-gradient(rgba(255,140,0,0.25),rgba(255,140,0,0.25));"
                f"color:rgba(200,100,0,0.9);border-bottom:2px solid rgba(255,140,0,0.5)'>"
                f"{label}</th>"
            )
        if j in purple_pos:
            return (
                f"<th style='{_TH_STYLE}"
                f"background-image:linear-gradient(rgba(124,58,237,0.35),rgba(124,58,237,0.35));"
                f"color:rgba(221,214,254,1.0);border-bottom:2px solid rgba(167,139,250,0.7)'>"
                f"{label}</th>"
            )
        if j in green_pos:
            return (
                f"<th style='{_TH_STYLE}"
                f"background-image:linear-gradient(rgba(16,185,129,0.25),rgba(16,185,129,0.25));"
                f"color:rgba(16,185,129,1.0);border-bottom:2px solid rgba(16,185,129,0.5)'>"
                f"{label}</th>"
            )
        return f"<th style='{_TH_STYLE}'>{label}</th>"

    headers = "".join(_th(j, c) for j, c in enumerate(col_names))
    rows_parts = []
    for i, (_, row) in enumerate(df.iterrows()):
        is_red = (i < highlight_row_count) or (
            highlight_row_indices is not None and i in highlight_row_indices
        )
        if is_red:
            cells = "".join(
                f"<td style='{_TD_STYLE}"
                f"{'border-left:3px solid rgba(220,50,50,0.55);' if j == 0 else ''}"
                f"color:rgba(220,50,50,0.9);'>"
                f"{_html.escape(str(v))}</td>"
                for j, v in enumerate(row)
            )
            rows_parts.append(
                f"<tr style='background:rgba(239,68,68,0.10);'>{cells}</tr>"
            )
        else:
            cells = "".join(
                f"<td style='{_TD_STYLE}'>{_html.escape(str(v))}</td>"
                for v in row
            )
            rows_parts.append(f"<tr>{cells}</tr>")
    rows = "".join(rows_parts)
    scroll_style = (
        f"overflow-x:auto;overflow-y:auto;max-height:{max_height}px"
        if max_height else "overflow-x:auto"
    )
    return (
        f"<div style='{scroll_style}'>"
        "<table style='border-collapse:separate;border-spacing:0;width:100%'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table></div>"
    )


def _render_merge_detail_body(t: "DetectedTable") -> None:
    """列名対応表 + before/after プレビュー（expander なし）。"""
    raw = t.raw_df
    fmt = t.df
    n_residue = len(raw) - len(fmt)
    st.caption(f"ヘッダー行を統合し、残留ヘッダー {n_residue} 行をデータから除去しました")

    before_cols = list(raw.columns)
    after_cols = list(fmt.columns)
    max_len = max(len(before_cols), len(after_cols))
    col_diff_df = pd.DataFrame(
        {
            "整形前（第1行のみ）": before_cols + [""] * (max_len - len(before_cols)),
            "整形後（多段マージ）": after_cols + [""] * (max_len - len(after_cols)),
        }
    ).assign(
        変化=lambda df: df.apply(
            lambda r: "→" if r["整形前（第1行のみ）"] != r["整形後（多段マージ）"] else "=",
            axis=1,
        )
    )[["整形前（第1行のみ）", "変化", "整形後（多段マージ）"]]

    st.markdown("**列名の変化**")
    rows_html = "".join(
        "<tr>"
        + f"<td style='{_TD_STYLE}'>{_html.escape(str(r['整形前（第1行のみ）']))}</td>"
        + f"<td style='{_TD_CENTER_STYLE}'>{_html.escape(str(r['変化']))}</td>"
        + f"<td style='{_TD_STYLE}'>{_html.escape(str(r['整形後（多段マージ）']))}</td>"
        + "</tr>"
        for _, r in col_diff_df.iterrows()
    )
    headers_html = (
        f"<th style='{_TH_STYLE}'>整形前（第1行のみ）</th>"
        f"<th style='{_TH_STYLE}text-align:center;'>変化</th>"
        f"<th style='{_TH_STYLE}'>整形後（多段マージ）</th>"
    )
    st.markdown(
        "<div style='overflow-x:auto'>"
        "<table style='border-collapse:collapse'>"
        f"<thead><tr>{headers_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )

    raw_col_set = {str(c) for c in raw.columns}
    unit_cols = {str(c) for c in fmt.columns if "[" in str(c) and str(c) not in raw_col_set}
    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown(f"**整形前**（全件 / 赤色 {n_residue} 行が除去対象）")
        st.markdown(
            _df_to_html(raw.astype(str), max_height=340, highlight_row_count=n_residue),
            unsafe_allow_html=True,
        )
    with col_a:
        _after_hint = "紫列 = 単位付加" if unit_cols else ""
        st.markdown(f"**整形後**（全件{' / ' + _after_hint if _after_hint else ''}）")
        st.markdown(
            _df_to_html(fmt.astype(str), max_height=340, unit_col_names=unit_cols or None),
            unsafe_allow_html=True,
        )


def _merge_detail_body_html(t: "DetectedTable") -> str:
    """列名対応表 + before/after プレビューをHTML文字列で返す（ネスト details 用）。"""
    raw = t.raw_df
    fmt = t.df
    n_residue = len(raw) - len(fmt)

    before_cols = list(raw.columns)
    after_cols = list(fmt.columns)
    max_len = max(len(before_cols), len(after_cols))

    rows_html = ""
    for i in range(max_len):
        bc = _html.escape(before_cols[i]) if i < len(before_cols) else ""
        ac = _html.escape(after_cols[i]) if i < len(after_cols) else ""
        arrow = "→" if bc != ac else "="
        rows_html += (
            f"<tr>"
            f"<td style='{_TD_STYLE}'>{bc}</td>"
            f"<td style='{_TD_CENTER_STYLE}'>{arrow}</td>"
            f"<td style='{_TD_STYLE}'>{ac}</td>"
            f"</tr>"
        )
    headers_html = (
        f"<th style='{_TH_STYLE}'>整形前（第1行のみ）</th>"
        f"<th style='{_TH_STYLE}text-align:center;'>変化</th>"
        f"<th style='{_TH_STYLE}'>整形後（多段マージ）</th>"
    )

    PREVIEW_ROWS = 10
    hl = min(n_residue, PREVIEW_ROWS)
    raw_col_set = {str(c) for c in raw.columns}
    unit_cols = {str(c) for c in fmt.columns if "[" in str(c) and str(c) not in raw_col_set}
    before_tbl = _df_to_html(raw.head(PREVIEW_ROWS).astype(str), highlight_row_count=hl)
    after_tbl  = _df_to_html(
        fmt.head(PREVIEW_ROWS).astype(str), unit_col_names=unit_cols or None
    )
    _after_hint = " / 紫列 = 単位付加" if unit_cols else ""

    return (
        f"<p style='font-size:0.83em;opacity:0.65;margin:0 0 0.6rem'>"
        f"ヘッダー行を統合し、残留ヘッダー {n_residue} 行をデータから除去しました</p>"
        f"<p style='font-weight:600;margin:0 0 0.3rem'>列名の変化</p>"
        f"<div style='overflow-x:auto'>"
        f"<table style='border-collapse:collapse'>"
        f"<thead><tr>{headers_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>"
        f"<div style='display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.8rem'>"
        f"<div style='flex:1;min-width:280px'>"
        f"<p style='font-weight:600;margin:0 0 0.3rem'>整形前（赤色 {n_residue} 行が除去対象 / 先頭 {PREVIEW_ROWS} 行）</p>"
        f"{before_tbl}</div>"
        f"<div style='flex:1;min-width:280px'>"
        f"<p style='font-weight:600;margin:0 0 0.3rem'>整形後（先頭 {PREVIEW_ROWS} 行{_after_hint}）</p>"
        f"{after_tbl}</div>"
        f"</div>"
    )


_MHD_CSS = """
<style>
/* ── Level-2: その他の同様処理 ── */
details.mhd-l2 {
    border: 1px solid rgba(127,255,212,0.4);
    border-radius: 8px;
    margin: 0.6rem 0 1.2rem;
    background: rgba(127,255,212,0.04);
    overflow: hidden;
}
details.mhd-l2 > summary {
    padding: 0.65rem 1rem;
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-weight: 700;
    font-size: 0.97rem;
    user-select: none;
    transition: background 0.15s;
    color: rgba(127,255,212,0.9);
}
details.mhd-l2 > summary:hover { background: rgba(127,255,212,0.1); }
details.mhd-l2 > summary::-webkit-details-marker { display: none; }
details.mhd-l2 > summary::before {
    content: "▶";
    font-size: 0.62em;
    opacity: 0.75;
    display: inline-block;
    transition: transform 0.18s ease;
    flex-shrink: 0;
}
details.mhd-l2[open] > summary::before { transform: rotate(90deg); }
details.mhd-l2 > .mhd-body {
    padding: 0.6rem 0.8rem;
    border-top: 1px solid rgba(127,255,212,0.2);
}
/* ── Level-3: 個別テーブル ── */
details.mhd-l3 {
    border: 1px solid rgba(127,255,212,0.22);
    border-radius: 6px;
    margin: 0.35rem 0;
    background: rgba(0,0,0,0.12);
    overflow: hidden;
}
details.mhd-l3 > summary {
    padding: 0.5rem 0.85rem;
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-weight: 500;
    font-size: 0.88rem;
    user-select: none;
    transition: background 0.15s;
}
details.mhd-l3 > summary:hover { background: rgba(127,255,212,0.07); }
details.mhd-l3 > summary::-webkit-details-marker { display: none; }
details.mhd-l3 > summary::before {
    content: "▶";
    font-size: 0.55em;
    opacity: 0.55;
    display: inline-block;
    transition: transform 0.18s ease;
    flex-shrink: 0;
}
details.mhd-l3[open] > summary::before { transform: rotate(90deg); }
details.mhd-l3 > .mhd-body {
    padding: 0.55rem 0.85rem;
    border-top: 1px solid rgba(127,255,212,0.15);
}
</style>
"""


def _make_details_html(label: str, body_html: str, open: bool = False, level: int = 2) -> str:
    cls = f"mhd-l{level}"
    open_attr = " open" if open else ""
    return (
        f"<details class='{cls}'{open_attr}>"
        f"<summary>{label}</summary>"
        f"<div class='mhd-body'>{body_html}</div>"
        f"</details>"
    )


def _render_header_merge_detail(
    t: "DetectedTable",
    rest: "Optional[List[DetectedTable]]" = None,
) -> None:
    """代表テーブルの詳細 expander（Streamlit）。
    rest がある場合はネスト HTML details でその他を表示する。"""
    title_str = f"  🏷️ `{t.title}`" if t.title else ""
    with st.expander(
        f"**`{t.table_id}`**{title_str}  —  シート: {t.sheet_name}",
        expanded=True,
    ):
        _render_merge_detail_body(t)

        if rest:
            # レベル3: 各テーブルを <details> で包む
            inner_html = ""
            for r in rest:
                r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                lbl = (
                    f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                    f" — シート: {_html.escape(r.sheet_name)}"
                )
                inner_html += _make_details_html(
                    lbl, _merge_detail_body_html(r), open=False, level=3
                )
            # レベル2: 「その他の同様処理」<details>
            outer_html = _MHD_CSS + _make_details_html(
                f"その他の同様処理（{len(rest)} 件）",
                inner_html,
                open=False,
                level=2,
            )
            st.markdown(outer_html, unsafe_allow_html=True)


def _render_fill_cols_body(t: "DetectedTable") -> None:
    """グルーピング列 ffill の詳細（Streamlit ウィジェット版）。"""
    pre = t.pre_fill_df
    post = t.df
    cols = getattr(t, "filled_cols", [])

    badges = " ".join(
        f"<span style='background:rgba(16,185,129,0.2);color:rgba(16,185,129,1);border:1px solid rgba(16,185,129,0.4);"
        f"border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600'>{_html.escape(c)}</span>"
        for c in cols
    )
    st.markdown(
        f"<p style='margin:4px 0 10px'>空白補完した列: {badges}</p>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**補完前**（オレンジ列 = 空白補完の対象）")
        if pre is not None:
            st.markdown(
                _df_to_html(pre, max_height=340, highlight_col_names=set(cols)),
                unsafe_allow_html=True,
            )
    with col_b:
        st.markdown("**補完後**（緑列 = 補完済み）")
        if post is not None:
            st.markdown(
                _df_to_html(post, max_height=340, green_col_names=set(cols)),
                unsafe_allow_html=True,
            )


def _render_fill_cols_body_html(t: "DetectedTable") -> str:
    """グルーピング列 ffill の詳細（HTML 文字列版）。"""
    pre = t.pre_fill_df
    post = t.df
    cols = getattr(t, "filled_cols", [])

    badges = " ".join(
        f"<span style='background:rgba(16,185,129,0.2);color:rgba(16,185,129,1);border:1px solid rgba(16,185,129,0.4);"
        f"border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600'>{_html.escape(c)}</span>"
        for c in cols
    )
    badge_html = f"<p style='margin:4px 0 10px'>空白補完した列: {badges}</p>"

    pre_html = _df_to_html(pre, max_height=340, highlight_col_names=set(cols)) if pre is not None else ""
    post_html = _df_to_html(post, max_height=340, green_col_names=set(cols)) if post is not None else ""

    return (
        badge_html
        + "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px'>"
        + f"<div><p style='margin:0 0 6px;font-weight:600'>補完前（オレンジ列 = 空白補完の対象）</p>{pre_html}</div>"
        + f"<div><p style='margin:0 0 6px;font-weight:600'>補完後（緑列 = 補完済み）</p>{post_html}</div>"
        + "</div>"
    )


def _render_stack_body(t: "DetectedTable") -> None:
    """クロス集計→縦持ち変換の詳細（Streamlit ウィジェット版）。"""
    info = t.stack_info
    wide = t.df
    long_df = t.stacked_df
    if not info or wide is None or long_df is None:
        return

    label_cols = info.get("label_cols", [])
    time_cols  = info.get("time_cols", [])
    var_name   = info.get("var_name", "期間")
    value_name = info.get("value_name", "値")
    year_ctx   = info.get("year_context")

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    label_html = " ".join(_badge(c, "156,163,175") for c in label_cols) or "（なし）"
    shown_time = time_cols[:6]
    rest_count = len(time_cols) - len(shown_time)
    time_html  = " ".join(_badge(c, "56,189,248") for c in shown_time)
    if rest_count > 0:
        time_html += f" <span style='font-size:12px;opacity:0.7'>...他 {rest_count} 列</span>"

    meta_lines = [
        f"ラベル列: {label_html}",
        f"時系列列: {time_html}（計 {len(time_cols)} 列）",
        f"縦持ち後の列構成: ラベル列 → <b>{_html.escape(var_name)}</b> → <b>{_html.escape(value_name)}</b>",
    ]
    if year_ctx:
        meta_lines.append(f"年コンテキスト（タイトル/ファイル名から補完）: <b>{year_ctx}年</b>")

    st.markdown(
        "<div style='margin:4px 0 12px;line-height:2'>" +
        "<br>".join(meta_lines) + "</div>",
        unsafe_allow_html=True,
    )

    time_col_set  = set(time_cols)
    new_col_set   = {var_name, value_name}
    if year_ctx and info.get("time_kind") == "month":
        new_col_set.add("年")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**変換前**（横持ち / {len(wide.columns)} 列 / オレンジ列 = 時系列列）")
        st.markdown(
            _df_to_html(wide, max_height=340, highlight_col_names=time_col_set),
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(f"**変換後**（縦持ち / {len(long_df.columns)} 列 × {len(long_df)} 行 / 緑列 = 変換で生まれた列）")
        st.markdown(
            _df_to_html(long_df, max_height=340, green_col_names=new_col_set),
            unsafe_allow_html=True,
        )


def _render_stack_body_html(t: "DetectedTable") -> str:
    """クロス集計→縦持ち変換の詳細（HTML 文字列版）。"""
    info = t.stack_info
    wide = t.df
    long_df = t.stacked_df
    if not info or wide is None or long_df is None:
        return ""

    label_cols = info.get("label_cols", [])
    time_cols  = info.get("time_cols", [])
    var_name   = info.get("var_name", "期間")
    value_name = info.get("value_name", "値")
    year_ctx   = info.get("year_context")

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    label_html = " ".join(_badge(c, "156,163,175") for c in label_cols) or "（なし）"
    shown_time = time_cols[:6]
    rest_count = len(time_cols) - len(shown_time)
    time_html  = " ".join(_badge(c, "56,189,248") for c in shown_time)
    if rest_count > 0:
        time_html += f" <span style='font-size:12px;opacity:0.7'>...他 {rest_count} 列</span>"

    year_line = f"<br>年コンテキスト: <b>{year_ctx}年</b>" if year_ctx else ""
    meta_html = (
        f"<div style='margin:4px 0 12px;line-height:2'>"
        f"ラベル列: {label_html}<br>"
        f"時系列列: {time_html}（計 {len(time_cols)} 列）<br>"
        f"縦持ち後の列構成: ラベル列 → <b>{_html.escape(var_name)}</b> → <b>{_html.escape(value_name)}</b>"
        f"{year_line}</div>"
    )

    time_col_set = set(time_cols)
    new_col_set  = {var_name, value_name}
    if year_ctx and info.get("time_kind") == "month":
        new_col_set.add("年")

    pre_html  = _df_to_html(wide, max_height=340, highlight_col_names=time_col_set)
    post_html = _df_to_html(long_df, max_height=340, green_col_names=new_col_set)
    grid_html = (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px'>"
        f"<div><p style='margin:0 0 6px;font-weight:600'>変換前（横持ち / {len(wide.columns)} 列 / オレンジ列 = 時系列列）</p>{pre_html}</div>"
        f"<div><p style='margin:0 0 6px;font-weight:600'>変換後（縦持ち / {len(long_df.columns)} 列 × {len(long_df)} 行 / 緑列 = 変換で生まれた列）</p>{post_html}</div>"
        "</div>"
    )
    return meta_html + grid_html


def _render_agg_removal_body(t: "DetectedTable") -> None:
    """集計除去の詳細（Streamlit ウィジェット版、expander なし）。"""
    pre = t.pre_agg_df
    post = t.df

    removed_rows = t.agg_rows_removed
    removed_cols = t.agg_cols_removed

    parts = []
    if removed_rows:
        parts.append(f"集計行 **{len(removed_rows)}** 行")
    if removed_cols:
        parts.append(f"集計列 **{len(removed_cols)}** 列")
    st.caption("、".join(parts) + " を除去しました")

    # 除去した列
    if removed_cols:
        st.markdown("**除去した集計列**")
        st.markdown(
            " &nbsp;".join(
                f"<code style='background:rgba(255,180,100,0.15);"
                f"border:1px solid rgba(255,180,100,0.4);border-radius:4px;"
                f"padding:1px 6px'>{_html.escape(c)}</code>"
                for c in removed_cols
            ),
            unsafe_allow_html=True,
        )

    if removed_rows:
        n_removed = len(removed_rows)
        st.markdown(f"**除去した集計行**（{n_removed} 件）")
        rows_html = "".join(
            "<tr>"
            + "".join(
                (
                    f"<td style='{_TD_STYLE}'>"
                    f"<span style='background:rgba(255,140,0,0.22);border:1px solid rgba(255,140,0,0.45);"
                    f"border-radius:3px;padding:1px 5px;font-weight:600'>"
                    f"{_html.escape(str(v))}</span></td>"
                ) if (
                    "__trigger_col__" in row_info
                    and k == row_info["__trigger_col__"]
                ) or (
                    "__trigger_col__" not in row_info
                    and _is_agg_label(str(v))
                ) else (
                    f"<td style='{_TD_STYLE}'>{_html.escape(str(v))}</td>"
                )
                for k, v in row_info.items() if k != "__trigger_col__"
            )
            + "</tr>"
            for row_info in removed_rows
        )
        headers_html = "".join(
            f"<th style='{_TH_STYLE}'>{_html.escape(c)}</th>"
            for c in removed_rows[0].keys() if c != "__trigger_col__"
        )
        row_max_h = 300 if n_removed > 10 else None
        scroll_style = (
            f"overflow-x:auto;overflow-y:auto;max-height:{row_max_h}px"
            if row_max_h else "overflow-x:auto"
        )
        st.markdown(
            f"<div style='{scroll_style}'>"
            "<table style='border-collapse:collapse'>"
            f"<thead><tr>{headers_html}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table></div>",
            unsafe_allow_html=True,
        )

    # before / after プレビュー（代表テーブルは全件スクロール）
    st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
    removed_positions = set(getattr(t, "agg_rows_removed_positions", []))
    n_removed_rows = len(removed_rows)
    _before_hints = []
    if n_removed_rows:
        _before_hints.append(f"赤色 {n_removed_rows} 行が除去対象")
    if removed_cols:
        _before_hints.append("オレンジ列が除去対象")
    _before_label = "全件 / " + "・".join(_before_hints) if _before_hints else "全件"
    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown(f"**除去前**（{_before_label}）")
        st.markdown(
            _df_to_html(
                pre.astype(str),
                max_height=340,
                highlight_row_indices=removed_positions,
                highlight_col_names=set(removed_cols) if removed_cols else None,
            ),
            unsafe_allow_html=True,
        )
    with col_a:
        st.markdown("**除去後**（全件）")
        st.markdown(_df_to_html(post.astype(str), max_height=340), unsafe_allow_html=True)


def _render_agg_removal_body_html(t: "DetectedTable") -> str:
    """集計除去の詳細を HTML 文字列で返す（ネスト details 用）。"""
    pre = t.pre_agg_df
    post = t.df
    removed_rows = t.agg_rows_removed
    removed_cols = t.agg_cols_removed

    parts = []
    if removed_rows:
        parts.append(f"集計行 {len(removed_rows)} 行")
    if removed_cols:
        parts.append(f"集計列 {len(removed_cols)} 列")
    caption = "、".join(parts) + " を除去しました"

    cols_html = ""
    if removed_cols:
        badges = " ".join(
            f"<code style='background:rgba(255,180,100,0.15);"
            f"border:1px solid rgba(255,180,100,0.4);border-radius:4px;"
            f"padding:1px 5px;font-size:0.82em'>{_html.escape(c)}</code>"
            for c in removed_cols
        )
        cols_html = f"<p style='margin:0.4em 0 0.2em'><b>除去した集計列:</b> {badges}</p>"

    rows_html_block = ""
    if removed_rows:
        n_removed = len(removed_rows)
        rows_html = "".join(
            "<tr>"
            + "".join(
                (
                    f"<td style='{_TD_STYLE}'>"
                    f"<span style='background:rgba(255,140,0,0.22);border:1px solid rgba(255,140,0,0.45);"
                    f"border-radius:3px;padding:1px 5px;font-weight:600'>"
                    f"{_html.escape(str(v))}</span></td>"
                ) if (
                    "__trigger_col__" in row_info
                    and k == row_info["__trigger_col__"]
                ) or (
                    "__trigger_col__" not in row_info
                    and _is_agg_label(str(v))
                ) else (
                    f"<td style='{_TD_STYLE}'>{_html.escape(str(v))}</td>"
                )
                for k, v in row_info.items() if k != "__trigger_col__"
            )
            + "</tr>"
            for row_info in removed_rows
        )
        headers_html = "".join(
            f"<th style='{_TH_STYLE}'>{_html.escape(c)}</th>"
            for c in removed_rows[0].keys() if c != "__trigger_col__"
        )
        row_scroll = (
            f"overflow-x:auto;overflow-y:auto;max-height:340px"
            if n_removed > 10 else "overflow-x:auto"
        )
        rows_html_block = (
            f"<p style='margin:0.6em 0 0.2em'><b>除去した集計行（{n_removed} 件）:</b></p>"
            f"<div style='{row_scroll}'>"
            "<table style='border-collapse:collapse'>"
            f"<thead><tr>{headers_html}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table></div>"
        )

    PREVIEW = 10
    removed_positions = set(getattr(t, "agg_rows_removed_positions", []))
    preview_removed = {p for p in removed_positions if p < PREVIEW}
    before_tbl = _df_to_html(
        pre.head(PREVIEW).astype(str),
        highlight_row_indices=preview_removed,
        highlight_col_names=set(removed_cols) if removed_cols else None,
    )
    after_tbl = _df_to_html(post.head(PREVIEW).astype(str))
    n_removed_label = len(removed_rows)
    _hints = []
    if n_removed_label:
        _hints.append(f"赤色 {n_removed_label} 行が除去対象")
    if removed_cols:
        _hints.append("オレンジ列が除去対象")
    _before_lbl = f"先頭 {PREVIEW} 行" + (" / " + "・".join(_hints) if _hints else "")
    preview_html = (
        f"<div style='display:flex;gap:1rem;flex-wrap:wrap;margin-top:1.2rem'>"
        f"<div style='flex:1;min-width:280px'>"
        f"<p style='font-weight:600;margin:0 0 0.3rem'>除去前（{_before_lbl}）</p>{before_tbl}</div>"
        f"<div style='flex:1;min-width:280px'>"
        f"<p style='font-weight:600;margin:0 0 0.3rem'>除去後（先頭 {PREVIEW} 行）</p>{after_tbl}</div>"
        f"</div>"
    )

    return (
        f"<p style='font-size:0.82em;opacity:0.7;margin:0 0 0.4em'>{_html.escape(caption)}</p>"
        f"{cols_html}{rows_html_block}{preview_html}"
    )


def step_format():
    st.header("🔧 ステップ 3 : テーブル整形")

    tables: List[DetectedTable] = st.session_state.detected_tables

    if st.session_state.auto_processing:
        st.session_state.step = 4
        st.rerun()

    formatted = [t for t in tables if t.raw_df is not None]
    unformatted = [t for t in tables if t.raw_df is None]
    agg_removed = [t for t in tables if t.pre_agg_df is not None]
    fill_applied = [t for t in tables if getattr(t, "filled_cols", [])]

    stacked_all = [t for t in tables if getattr(t, "stacked_df", None) is not None]
    nothing_done = not formatted and not agg_removed and not fill_applied and not stacked_all
    if nothing_done:
        st.info("全テーブルに対して整形処理はありませんでした。")
    else:
        first_section = True

        # ── ① 多段ヘッダーの検出と解決機能 ──────────────────────────────
        if formatted:
            if not first_section:
                st.divider()
            first_section = False
            st.subheader(f"🔗 多段ヘッダーの検出と解決機能（対象：{len(formatted)}テーブル）")
            st.success(
                f"**{len(formatted)}** テーブルで多段ヘッダーを統合しました"
                f"（整形なし: {len(unformatted)} テーブル）"
            )
            rest = formatted[1:] if len(formatted) > 1 else None
            _render_header_merge_detail(formatted[0], rest=rest)

        # ── ② グルーピング列の前方補完（視覚結合セル対応） ─────────────────
        if fill_applied:
            if not first_section:
                st.divider()
            first_section = False
            total_filled = sum(len(getattr(t, "filled_cols", [])) for t in fill_applied)
            st.subheader(f"↕️ グルーピング列の空白補完機能（対象：{len(fill_applied)}テーブル）")
            st.success(
                f"**{len(fill_applied)}** テーブルのグルーピング列の空白を上の値で埋めました  "
                f"（列: {total_filled} 件）"
            )
            rep_f = fill_applied[0]
            rep_f_title = f"  🏷️ `{rep_f.title}`" if rep_f.title else ""
            with st.expander(
                f"**`{rep_f.table_id}`**{rep_f_title}  —  シート: {rep_f.sheet_name}",
                expanded=True,
            ):
                _render_fill_cols_body(rep_f)

                rest_fill = fill_applied[1:]
                if rest_fill:
                    inner_html = ""
                    for r in rest_fill:
                        r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                        lbl = (
                            f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                            f" — シート: {_html.escape(r.sheet_name)}"
                        )
                        inner_html += _make_details_html(
                            lbl, _render_fill_cols_body_html(r), open=False, level=3
                        )
                    outer_html = _MHD_CSS + _make_details_html(
                        f"その他の同様処理（{len(rest_fill)} 件）",
                        inner_html,
                        open=False,
                        level=2,
                    )
                    st.markdown(outer_html, unsafe_allow_html=True)

        # ── ③ 集計行の検出・削除・メタデータ保存機能 ──────────────────────
        if agg_removed:
            if not first_section:
                st.divider()
            total_rows = sum(len(t.agg_rows_removed) for t in agg_removed)
            total_cols = sum(len(t.agg_cols_removed) for t in agg_removed)
            st.subheader(f"🗑️ 集計行の検出・削除・メタデータ保存機能（対象：{len(agg_removed)}テーブル）")
            st.success(
                f"**{len(agg_removed)}** テーブルで集計行・集計列を除去しました  "
                f"（行: {total_rows} 件、列: {total_cols} 件）"
            )

            # 代表テーブル（Streamlit expander）
            rep = agg_removed[0]
            rep_title = f"  🏷️ `{rep.title}`" if rep.title else ""
            with st.expander(
                f"**`{rep.table_id}`**{rep_title}  —  シート: {rep.sheet_name}",
                expanded=True,
            ):
                _render_agg_removal_body(rep)

                # その他を MHD_CSS + <details> でネスト表示
                rest_agg = agg_removed[1:]
                if rest_agg:
                    inner_html = ""
                    for r in rest_agg:
                        r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                        lbl = (
                            f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                            f" — シート: {_html.escape(r.sheet_name)}"
                        )
                        inner_html += _make_details_html(
                            lbl, _render_agg_removal_body_html(r), open=False, level=3
                        )
                    outer_html = _MHD_CSS + _make_details_html(
                        f"その他の同様処理（{len(rest_agg)} 件）",
                        inner_html,
                        open=False,
                        level=2,
                    )
                    st.markdown(outer_html, unsafe_allow_html=True)

        # ── ④ クロス集計形式の検出と縦持ち変換機能 ──────────────────────────
        stacked = [t for t in tables if getattr(t, "stacked_df", None) is not None]
        if stacked:
            if not first_section:
                st.divider()
            st.subheader(f"📐 クロス集計形式の検出と縦持ち変換機能（対象：{len(stacked)}テーブル）")
            total_time_cols = sum(len(getattr(t, "stack_info", {}).get("time_cols", [])) for t in stacked)
            st.success(
                f"**{len(stacked)}** テーブルで横持ち時系列列を検出し、縦持ちに変換しました  "
                f"（時系列列: 計 {total_time_cols} 列）"
            )
            rep_s = stacked[0]
            rep_s_title = f"  🏷️ `{rep_s.title}`" if rep_s.title else ""
            with st.expander(
                f"**`{rep_s.table_id}`**{rep_s_title}  —  シート: {rep_s.sheet_name}",
                expanded=True,
            ):
                _render_stack_body(rep_s)

                rest_stack = stacked[1:]
                if rest_stack:
                    inner_html = ""
                    for r in rest_stack:
                        r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                        lbl = (
                            f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                            f" — シート: {_html.escape(r.sheet_name)}"
                        )
                        inner_html += _make_details_html(
                            lbl, _render_stack_body_html(r), open=False, level=3
                        )
                    outer_html = _MHD_CSS + _make_details_html(
                        f"その他の同様処理（{len(rest_stack)} 件）",
                        inner_html,
                        open=False,
                        level=2,
                    )
                    st.markdown(outer_html, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 4])
    with c1:
        st.button("← 戻る", on_click=_go_to, args=(2,))
    with c2:
        st.button(
            "次へ：テーブル関係分析を開始 →",
            type="primary",
            use_container_width=True,
            on_click=_go_to,
            args=(4,),
        )


# ---------------------------------------------------------------------------
# Step 4 — テーブル関係分析
# ---------------------------------------------------------------------------


def step3():
    st.header("🧠 ステップ 4 : テーブル関係分析")

    if st.session_state.ai_analysis is None:
        with st.spinner("テーブルを分析中です..."):
            try:
                result = analyze_tables(
                    st.session_state.detected_tables,
                    st.session_state.sheet_names,
                )
                st.session_state.ai_analysis = result
            except Exception as e:
                st.error(f"❌ テーブル関係分析エラー: {e}")
                return

    analysis: AIAnalysisResult = st.session_state.ai_analysis

    # 初回の前進パス中のみ自動的に次のステップへ進む
    if st.session_state.auto_processing:
        st.session_state.step = 5
        st.rerun()

    # サマリーバナー
    st.info(f"📊 **分析サマリー**: {analysis.summary}")

    # シート分類
    with st.expander("📋 シート分類", expanded=True):
        for sc in analysis.sheet_classifications:
            icon = "📊" if sc.is_data_sheet else "📝"
            badge = "データシート" if sc.is_data_sheet else "説明 / 補足シート"
            st.markdown(f"{icon} **{sc.sheet_name}** → {badge}")
            if sc.description:
                st.caption(sc.description)

    # 階層概要
    with st.expander("🔗 テーブル階層・種別", expanded=True):
        detail_t = [
            ta for ta in analysis.table_analyses if ta.granularity_level == "detail"
        ]
        summary_t = [
            ta for ta in analysis.table_analyses if ta.granularity_level == "summary"
        ]
        master_t = [ta for ta in analysis.table_analyses if ta.is_master_table]
        other_t = [
            ta
            for ta in analysis.table_analyses
            if ta.granularity_level not in ("detail", "summary")
            and not ta.is_master_table
        ]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**🔍 詳細データ**")
            for ta in detail_t:
                star = "⭐ " if ta.is_minimum_granularity_candidate else ""
                st.markdown(f"- {star}{ta.display_name}")
        with c2:
            st.markdown("**📈 集計データ**")
            for ta in summary_t:
                st.markdown(f"- {ta.display_name}")
        with c3:
            st.markdown("**📚 マスタ**")
            for ta in master_t:
                st.markdown(f"- {ta.display_name}")
        with c4:
            st.markdown("**📄 その他**")
            for ta in other_t:
                st.markdown(f"- {ta.display_name}")

    # 最小粒度
    min_cands = [
        ta for ta in analysis.table_analyses if ta.is_minimum_granularity_candidate
    ]
    if min_cands:
        with st.expander(
            f"⭐ 最小粒度データ候補（{len(min_cands)} 件）", expanded=True
        ):
            for ta in min_cands:
                st.markdown(f"**{ta.display_name}** &nbsp; `{ta.table_id}`")
                st.markdown(f"> {ta.description}")
                if ta.has_external_info and ta.external_info_description:
                    st.warning(f"⚠️ 外だし情報あり: {ta.external_info_description}")
                st.caption(f"💡 {ta.reasoning}")
                if ta.parent_table_ids:
                    st.caption(f"📤 上位テーブル: {', '.join(ta.parent_table_ids)}")
                st.divider()

    # マスタテーブル
    if analysis.master_tables:
        with st.expander(
            f"📚 マスタテーブル（{len(analysis.master_tables)} 件）", expanded=False
        ):
            for mt in analysis.master_tables:
                st.markdown(f"**{mt.table_id}**  —  キー列: `{mt.key_column}`")
                st.caption(mt.description)
                if mt.referenced_by:
                    st.caption(f"参照元: {', '.join(mt.referenced_by)}")
                st.divider()

    # 統合推奨プレビュー
    if analysis.integration_recommendations:
        with st.expander(
            f"🔀 統合推奨（{len(analysis.integration_recommendations)} 件） — 次のステップで確認します",
            expanded=False,
        ):
            for ir in analysis.integration_recommendations:
                st.markdown(f"**{ir.group_name}**  →  対象: {', '.join(ir.table_ids)}")
                st.caption(ir.reasoning)
    else:
        st.info("統合推奨はありません")

    # マスタ生成推奨プレビュー（軸ごと）
    _s3_tables_dict = {t.table_id: t for t in st.session_state.detected_tables}
    _s3_master_previews = _collect_unique_master_specs(
        analysis.integration_recommendations, _s3_tables_dict
    )
    if _s3_master_previews:
        with st.expander(
            f"🗂️ マスタ自動生成推奨（{len(_s3_master_previews)} 件） — 次のステップで確認します",
            expanded=False,
        ):
            for ir, spec in _s3_master_previews:
                child_col = spec["child_col"]
                parent_col = spec["parent_col"]
                st.markdown(
                    f"**{child_col} × {parent_col} マスタ**"
                    f"  ←  上位集計テーブル `{spec['parent_id']}` の階層から生成"
                )
                st.caption(
                    f"統合テーブルを「{child_col}」で結合すると「{parent_col}」単位の再集計が可能になります。"
                )

    c1, c2 = st.columns([1, 4])
    with c1:
        st.button("← 戻る", on_click=_go_to, args=(3,))
    with c2:
        st.button(
            "次へ：新規テーブル案生成 →",
            type="primary",
            use_container_width=True,
            on_click=_go_to,
            args=(5,),
        )


# ---------------------------------------------------------------------------
# Step 4 — 統合レビュー
# ---------------------------------------------------------------------------


def _ir_column_signature(ir, tables_dict) -> frozenset:
    """「類似」統合推奨を検出するために使用する列シグネチャ。

    同じ識別列を持ち、かつソーステーブルのスキーマが同一である場合、
    2つのIRは類似とみなされる。
    """
    col_names = getattr(ir, "new_column_names", []) or [ir.new_column_name]
    for tid in ir.table_ids:
        t = tables_dict.get(tid)
        if t is not None and t.effective_df is not None:
            return frozenset(col_names + list(t.effective_df.columns))
    return frozenset(col_names)


def _detect_redundant_axes(col_names: list, multi_vals: dict, ir, tables_dict: dict) -> set:
    """統合で追加する新カラムのうち、既存データから導出可能な冗長な軸を検出する。

    新カラム値に含まれる4桁年（YYYY）が、各テーブルの既存カラムのいずれかに
    すでに含まれている場合、そのカラムは冗長と判定する。

    Returns: 冗長と判定した col_names のインデックス集合
    """
    import re as _r
    redundant: set = set()
    for ci, col_name in enumerate(col_names):
        # 全テーブルについて「新カラム値の年が既存カラムに存在するか」を確認
        all_found = True
        any_table = False
        for tid in ir.table_ids:
            t = tables_dict.get(tid)
            if t is None or t.effective_df is None:
                all_found = False
                break
            val = (multi_vals.get(tid) or [])[ci] if ci < len(multi_vals.get(tid) or []) else ir.new_column_values.get(tid, "")
            year_m = _r.search(r"(19|20)\d{2}", str(val))
            if not year_m:
                all_found = False
                break
            year = year_m.group(0)
            df = t.effective_df
            found_in_col = False
            for ec in df.columns:
                # 新カラム名と同一のカラムは除外（自己参照防止）
                if str(ec) == str(col_name):
                    continue
                non_null = [str(v) for v in df[ec].dropna() if str(v).lower() not in ("nan", "none", "")]
                if not non_null:
                    continue
                match_ratio = sum(1 for v in non_null if year in v) / len(non_null)
                if match_ratio >= 0.5:
                    found_in_col = True
                    break
            if not found_in_col:
                all_found = False
                break
            any_table = True
        if all_found and any_table and len(ir.table_ids) >= 2:
            # 統合後も各テーブルの行が既存カラムで自然に区別できるか確認する
            # → 各テーブルで検出された「年を含むカラム」の値が互いに重複しない場合のみ冗長とみなす
            year_val_sets: list = []
            distinguishable = True
            for tid in ir.table_ids:
                t = tables_dict.get(tid)
                val = (multi_vals.get(tid) or [])[ci] if ci < len(multi_vals.get(tid) or []) else ir.new_column_values.get(tid, "")
                year_m = _r.search(r"(19|20)\d{2}", str(val))
                if not year_m:
                    distinguishable = False
                    break
                year = year_m.group(0)
                df = t.effective_df
                vals_with_year: set = set()
                for ec in df.columns:
                    if str(ec) == str(col_name):
                        continue
                    non_null = [str(v) for v in df[ec].dropna() if str(v).lower() not in ("nan", "none", "")]
                    if not non_null:
                        continue
                    if sum(1 for v in non_null if year in v) / len(non_null) >= 0.5:
                        vals_with_year.update(non_null)
                year_val_sets.append(vals_with_year)
            # 各テーブルの値集合が互いに重複しなければ自然に区別可能
            if distinguishable and len(year_val_sets) >= 2:
                for i in range(len(year_val_sets)):
                    for j in range(i + 1, len(year_val_sets)):
                        if year_val_sets[i] & year_val_sets[j]:
                            distinguishable = False
                            break
                    if not distinguishable:
                        break
            if distinguishable:
                redundant.add(ci)
    return redundant


def _group_irs_by_similarity(irs, tables_dict):
    """リストのリストを返す: 各内部リストは類似IRのグループ。

    各グループの先頭要素が代表（展開表示）となり、
    残りはドロップダウンに折りたたまれる。
    """
    groups: list = []
    sig_to_group: dict = {}
    for ir in irs:
        sig = _ir_column_signature(ir, tables_dict)
        if sig not in sig_to_group:
            group: list = [ir]
            groups.append(group)
            sig_to_group[sig] = group
        else:
            sig_to_group[sig].append(ir)
    return groups


def _axis_type(axis_idx: int, multi_vals: dict, members: list) -> str:
    """軸の値がsheet名またはセクションタイトルと一致するかを確認し、'sheet'または'title'を返す。"""
    sheet_matches = 0
    title_matches = 0
    total = 0
    for m in members:
        vals = multi_vals.get(m.table_id) or []
        if axis_idx >= len(vals) or not vals[axis_idx]:
            continue
        val = str(vals[axis_idx])
        total += 1
        if val == m.sheet_name:
            sheet_matches += 1
        elif val == (m.title or ""):
            title_matches += 1
    if total == 0:
        return "sheet"
    return "sheet" if sheet_matches >= title_matches else "title"


def _derive_master_specs_for_ir(ir, tables_dict) -> list:
    """親を持つ軸ごとに1つずつ、マスタspecのリストを返す。

    各specは以下のキーを持つdict:
      child_col, parent_col, mapping {child_val: parent_val},
      parent_id, parent_label, axis_idx
    """
    col_names = getattr(ir, "new_column_names", []) or [ir.new_column_name]
    multi_vals = getattr(ir, "new_column_multi_values", {}) or {}
    members = [tables_dict.get(t) for t in ir.table_ids]
    members = [m for m in members if m is not None]
    if len(members) < 2:
        return []

    child_sheets = {m.sheet_name for m in members}

    # 軸ごとの親情報を解決する; axis 0については旧シングル軸フィールドにフォールバック
    axis_parents = list(getattr(ir, "axis_parent_table_ids", []) or [])
    axis_parent_cols = list(getattr(ir, "axis_parent_label_columns", []) or [])
    while len(axis_parents) < len(col_names):
        axis_parents.append(None)
    while len(axis_parent_cols) < len(col_names):
        axis_parent_cols.append(None)
    if axis_parents[0] is None and getattr(ir, "parent_table_id", None):
        axis_parents[0] = ir.parent_table_id
        if not axis_parent_cols[0]:
            axis_parent_cols[0] = getattr(ir, "parent_label_column", None)

    specs = []
    for axis_idx in range(len(col_names)):
        parent_tid = axis_parents[axis_idx]
        if not parent_tid:
            continue
        parent = tables_dict.get(parent_tid)
        if parent is None:
            continue

        child_col = col_names[axis_idx]
        parent_col = axis_parent_cols[axis_idx] or f"上位{child_col}"

        # 軸の値をメタデータと比較して軸タイプ（sheet vs title）を判定する
        atype = _axis_type(axis_idx, multi_vals, members)

        # 検証: 親は子の兄弟（同レベル）であってはならない
        if atype == "sheet":
            if parent.sheet_name in child_sheets:
                continue  # 親が子と同じsheetにある → 兄弟のため除外
            parent_label = parent.sheet_name
        else:  # title軸
            child_titles = {(m.title or "") for m in members}
            if (parent.title or parent.sheet_name) in child_titles:
                continue  # 親が子と同じタイトルを持つ → 兄弟のため除外
            parent_label = parent.title or parent.sheet_name

        # 実際の軸の値から child → parent_label のマッピングを構築する
        mapping: dict = {}
        for m in members:
            vals = multi_vals.get(m.table_id) or [
                ir.new_column_values.get(m.table_id, "")
            ]
            child_val = vals[axis_idx] if axis_idx < len(vals) else ""
            if not child_val:
                child_val = (
                    m.sheet_name if atype == "sheet" else (m.title or m.sheet_name)
                )
            if child_val:
                mapping[child_val] = (
                    parent_label  # 重複（同じ枝、異なるサービス）は集約する
                )

        if not mapping:
            continue

        specs.append(
            {
                "child_col": child_col,
                "parent_col": parent_col,
                "mapping": mapping,
                "parent_id": parent_tid,
                "parent_label": parent_label,
                "axis_idx": axis_idx,
            }
        )

    return specs


def _derive_master_spec(ir, tables_dict):
    """後方互換シム: axis 0 の最初のspecのみを返す。存在しない場合はNone。"""
    specs = _derive_master_specs_for_ir(ir, tables_dict)
    return specs[0] if specs else None


def _master_signature(spec):
    """specが生成するマスタの同一性を表す。どの統合由来かに関わらず、
    同じ child→parent ラベルマップを持つマスタは重複とみなす。"""
    return (
        spec["child_col"],
        spec["parent_col"],
        frozenset(spec["mapping"].items()),
    )


def _collect_unique_master_specs(irs, tables_dict) -> list:
    """全IRの全軸にわたって重複排除した (ir, spec) ペアのリストを返す。"""
    seen: set = set()
    result = []
    for ir in irs:
        for spec in _derive_master_specs_for_ir(ir, tables_dict):
            sig = _master_signature(spec)
            if sig in seen:
                continue
            seen.add(sig)
            result.append((ir, spec))
    return result


def _dedup_master_irs(master_irs, tables_dict):
    """後方互換シム: 有効なmaster specを少なくとも1つ持つIRを返す。"""
    seen_ir_ids: set = set()
    result = []
    for ir, _spec in _collect_unique_master_specs(master_irs, tables_dict):
        if ir.recommendation_id not in seen_ir_ids:
            seen_ir_ids.add(ir.recommendation_id)
            result.append(ir)
    return result


# ---------------------------------------------------------------------------
# 潜在テーブルヘルパー（Step 4）
# ---------------------------------------------------------------------------


def _dlt_virtual_table(
    dlt: DerivedLatentTable, tables_dict: dict
) -> Optional[DetectedTable]:
    """統合プレビュー用に DerivedLatentTable を仮想 DetectedTable としてラップする。"""
    parent = tables_dict.get(dlt.parent_table_id)
    if parent is None or dlt.df is None:
        return None
    return DetectedTable(
        table_id=dlt.proposal_id,
        sheet_name=parent.sheet_name,
        start_row=0,
        end_row=len(dlt.df),
        start_col=0,
        end_col=len(dlt.df.columns),
        df=dlt.df.copy(),
        title=dlt.derived_name,
    )


def _infer_col_name_from_values(values: List[str]) -> str:
    """値ラベルから妥当な列名を推定する。

    共通prefixの末尾にある区切り文字・数字を除去した語幹を返す。
    意味のある共通プレフィックスが見つからない場合は '種別' にフォールバックする。
    """
    import os.path as _osp

    if len(values) < 2:
        return "種別"
    prefix = _osp.commonprefix(values)
    prefix = re.sub(r"[-_・\s\d]+$", "", prefix).strip()
    return prefix if len(prefix) >= 2 else "種別"


def _infer_axis_name(values: List[str]) -> str:
    """値ラベル群から軸名称を動的に推定する。

    Step 1: 共通prefix（末尾の区切り・数字・英字を除去）で語幹を取り出す。
    Step 2: Step1 が失敗した場合、全値に共通して含まれる最長の部分文字列を探す。
            意味のある文字（日本語等）を含まない候補は除外する。
    どちらも失敗した場合は '種別' を返す。
    """
    import os.path as _osp

    if not values:
        return "種別"
    if len(values) == 1:
        return values[0]

    # Step 1: 共通prefix（末尾の区切り文字・数字・英字を除去）
    prefix = _osp.commonprefix(values)
    prefix_clean = re.sub(r"[-_・\s\dA-Za-zＡ-Ｚａ-ｚ０-９]+$", "", prefix).strip()
    if len(prefix_clean) >= 2:
        return prefix_clean

    # Step 2: 全値に含まれる最長の共通部分文字列を探す
    # 数字・ASCII英字・記号のみで構成される部分文字列は意味が薄いため除外する
    _HAS_MEANINGFUL = re.compile(r"[^\d\s\-_・A-Za-zＡ-Ｚａ-ｚ０-９（）()【】「」]")
    shortest = min(values, key=len)
    for length in range(len(shortest), 1, -1):
        for start in range(len(shortest) - length + 1):
            substr = shortest[start : start + length]
            if not _HAS_MEANINGFUL.search(substr):
                continue
            if all(substr in v for v in values):
                return substr.strip()

    return "種別"


def _strip_common_suffix(labels: List[str]) -> List[str]:
    """全ラベルに共通する末尾部分を除去して識別部分のみを返す。

    各ラベルから共通末尾（区切り文字含む）を取り除き、
    集計元を識別する語幹部分のみを返す。
    意味のある共通末尾が2文字未満の場合はそのまま返す。
    """
    import os.path as _osp

    if len(labels) < 2:
        return list(labels)

    rev = [lbl[::-1] for lbl in labels]
    suf_raw = _osp.commonprefix(rev)[::-1]

    # 区切り文字を除いた実質的な共通末尾が2文字以上あるか確認する
    suf_meaningful = re.sub(r"^[-_・\s（）()「」「」]+", "", suf_raw)
    if len(suf_meaningful) < 2:
        return list(labels)

    result = []
    for lbl in labels:
        if lbl.endswith(suf_raw):
            stripped = lbl[: -len(suf_raw)].rstrip("-_・ （）()「」「」").strip()
            result.append(stripped if stripped else lbl)
        else:
            result.append(lbl)
    return result


def _build_auto_irs_from_latent(
    latent_groups: List[LatentTableGroup],
    ext_tables_dict: dict,
    tables_dict: dict,
    _for_build: bool = False,
) -> List[tuple]:
    """承認された潜在グループに対して IntegrationRecommendation オブジェクトを構築する。

    (IntegrationRecommendation, group_key) タプルのフラットリストを返す。
    latent_group_decisions[group_key] が True かつDLTが少なくとも1つ存在するグループのみ
    IRを生成する。

    優先度ルール（グループに2軸統合IRが生成できる場合）:
      - 2軸IRは常に result に含める（ユーザーがいつでも選択を変更できるようにする）
      - 2軸統合が「統合しない」の場合のみ、1軸IRも追加する
    2軸統合が生成できない（メンバーが1件のみ）場合は常に1軸IRを返す。
    """
    result = []
    lg_dec = st.session_state.get("latent_group_decisions", {})
    if "latent_auto_int_decisions" not in st.session_state:
        st.session_state.latent_auto_int_decisions = {}
    auto_int_dec = st.session_state.latent_auto_int_decisions

    for group in latent_groups:
        gk = group.group_key
        if not lg_dec.get(gk, True):
            continue
        if not group.has_derived:
            continue

        members_with_dlt = [(lp, dlt) for lp, dlt in group.members if dlt is not None]

        # ── クロスシート集計メンバーを2軸統合から除外する ────────────────────────
        # あるメンバーの親テーブルの数値合計が他の全メンバーの合計に近い場合（許容誤差5%以内）、
        # そのメンバーは他シートを束ねたクロスシート集計であると判定し除外する。
        # 例: 拠点A + 拠点B = 全拠点集計 の場合、全拠点集計メンバーは冗長として除外する。
        if len(members_with_dlt) >= 3:

            def _num_total(tid: str) -> float:
                t = tables_dict.get(tid)
                if t is None or t.effective_df is None or t.effective_df.empty:
                    return 0.0
                try:
                    return float(
                        t.effective_df.select_dtypes(include="number").abs().values.sum()
                    )
                except Exception:
                    return 0.0

            _parent_totals = [
                _num_total(dlt.parent_table_id) for _, dlt in members_with_dlt
            ]
            _non_summary = []
            for _i, (lp, dlt) in enumerate(members_with_dlt):
                _others = sum(v for _j, v in enumerate(_parent_totals) if _j != _i)
                _this = _parent_totals[_i]
                if _others > 1e-9 and _this > 1e-9:
                    _ratio = abs(_this - _others) / max(_this, _others)
                    if _ratio < 0.05:
                        continue  # クロスシート集計 → 除外
                _non_summary.append((lp, dlt))
            if len(_non_summary) >= 2:
                members_with_dlt = _non_summary

        # ── 2軸統合IR を構築する（メンバーが2件以上の場合）─────────────────────
        cross_ir = None
        cross_rec_id = f"latent_auto_cross2_{gk}"

        if len(members_with_dlt) >= 2:
            member_sheets = [
                (
                    ext_tables_dict.get(lp.source_table_id)
                    or tables_dict.get(lp.source_table_id)
                )
                for lp, _ in members_with_dlt
            ]
            sheet_names_of_members = [t.sheet_name if t else "" for t in member_sheets]
            is_multi_sheet = len(set(sheet_names_of_members)) > 1

            # 軸2の値: 複数シートはsheet_name、同一シートはsource_titleから識別部分を抽出
            if is_multi_sheet:
                raw_axis2_vals = sheet_names_of_members
            else:
                raw_axis2_vals = [
                    (t.title or lp.source_table_id)
                    for (lp, _), t in zip(members_with_dlt, member_sheets)
                ]
                raw_axis2_vals = _strip_common_suffix(raw_axis2_vals)

            cross_ids: list = []
            cross_multi_vals: dict = {}
            for (lp, dlt), axis2_val in zip(members_with_dlt, raw_axis2_vals):
                for i, tid in enumerate(lp.detected_table_ids):
                    cat_val = (
                        lp.detected_names[i] if i < len(lp.detected_names) else tid
                    )
                    cross_ids.append(tid)
                    cross_multi_vals[tid] = [cat_val, axis2_val]
                cross_ids.append(dlt.proposal_id)
                cross_multi_vals[dlt.proposal_id] = [dlt.derived_name, axis2_val]

            if len(cross_ids) >= 4:
                cat_names_all = group.detected_names + group.missing_names
                cat_col_name = _infer_col_name_from_values(cat_names_all)
                axis2_unique = list(
                    dict.fromkeys(cross_multi_vals[tid][1] for tid in cross_ids)
                )
                axis2_col_name = _infer_axis_name(axis2_unique)
                if not axis2_col_name or axis2_col_name == "種別":
                    axis2_col_name = "シート" if is_multi_sheet else "区分"

                # デフォルトは「統合する」(True)
                if cross_rec_id not in auto_int_dec:
                    auto_int_dec[cross_rec_id] = True

                cross_ir = IntegrationRecommendation(
                    recommendation_id=cross_rec_id,
                    group_name=(
                        f"{'・'.join(group.missing_names)} × {axis2_col_name} "
                        f"2軸統合テーブル"
                    ),
                    description=(
                        f"差分推定した「{'・'.join(group.missing_names)}」を含む"
                        f"「{'・'.join(cat_names_all)}」を"
                        f"全{axis2_col_name}横断で統合した2軸テーブル"
                    ),
                    table_ids=cross_ids,
                    new_column_name=cat_col_name,
                    new_column_values={
                        tid: cross_multi_vals[tid][0] for tid in cross_ids
                    },
                    reasoning=(
                        f"潜在テーブル「{'・'.join(group.missing_names)}」の差分推定により、"
                        f"「{cat_col_name}」軸と「{axis2_col_name}」軸の2軸で"
                        f"全{axis2_col_name}のテーブルを統合できます。"
                    ),
                    new_column_names=[cat_col_name, axis2_col_name],
                    new_column_multi_values={
                        tid: list(v) for tid, v in cross_multi_vals.items()
                    },
                    # 構成要素軸(axis 0)の親を設定することでマスタ生成が機能する。
                    # 集計元軸(axis 1)は親なし（集計元自体が軸なのでマスタ不要）。
                    axis_parent_table_ids=[
                        members_with_dlt[0][0].source_table_id,
                        None,
                    ],
                    axis_parent_label_columns=[None, None],
                )

        # ── 1軸統合IR を構築する（集計元ごと）───────────────────────────────
        per_sheet_irs: list = []
        for lp, dlt in group.members:
            if dlt is None:
                continue
            table_ids = list(lp.detected_table_ids) + [dlt.proposal_id]
            col_vals: dict = {}
            for i, tid in enumerate(lp.detected_table_ids):
                col_vals[tid] = (
                    lp.detected_names[i] if i < len(lp.detected_names) else tid
                )
            col_vals[dlt.proposal_id] = dlt.derived_name

            col_name = _infer_col_name_from_values(list(col_vals.values()))
            rec_id = f"latent_auto_{gk}_{dlt.proposal_id}"

            # 1軸IRは2軸が「統合しない」に切り替えられた時に初めて表示される。
            # デフォルトを True にして、表示時に「統合する」が選択済みになるようにする。
            if rec_id not in auto_int_dec:
                auto_int_dec[rec_id] = True

            ir = IntegrationRecommendation(
                recommendation_id=rec_id,
                group_name=f"{'・'.join(group.missing_names)} 統合テーブル（{lp.source_title}）",
                description=(
                    f"差分推定した「{'・'.join(group.missing_names)}」を含む "
                    f"「{'・'.join(group.detected_names + group.missing_names)}」の統合テーブル"
                ),
                table_ids=table_ids,
                new_column_name=col_name,
                new_column_values={tid: v for tid, v in col_vals.items()},
                reasoning=(
                    f"潜在テーブル「{'・'.join(group.missing_names)}」の差分推定により、"
                    f"「{'・'.join(group.detected_names + group.missing_names)}」を"
                    f"1つの統合テーブルにまとめることができます。"
                ),
                new_column_names=[col_name],
                new_column_multi_values={tid: [v] for tid, v in col_vals.items()},
                # 親を設定することで、マスタテーブル生成がこのauto-IRに対して
                # 正しいマッピング（child値 → 親集計タイトル）を生成できるようにする。
                axis_parent_table_ids=[lp.source_table_id],
                axis_parent_label_columns=[None],
            )
            per_sheet_irs.append((ir, gk))

        # ── 優先度ルール: 2軸IRは常に表示し、「統合しない」の場合のみ1軸IRを追加 ──
        # _for_build=True（最終ビルド時）は決定に関わらず常に全IRを返し、
        # 呼び出し側がそれぞれの決定に基づいてフィルタリングする。
        if cross_ir is not None:
            result.append((cross_ir, gk))  # 常に2軸カードを表示
            if _for_build or not auto_int_dec.get(cross_rec_id, True):
                result.extend(per_sheet_irs)  # ビルド時は常に追加、表示時は2軸「統合しない」の場合のみ
        else:
            result.extend(per_sheet_irs)  # 2軸なし → 常に1軸

    return result


def _clip_ai_irs_by_latent_groups(
    irs: list,
    latent_groups: List[LatentTableGroup],
    tables_dict: dict,
) -> list:
    """潜在グループの検証済み兄弟関係と一致しないテーブルをAI統合提案から除去する。

    2段階の刈り込みを行う：

    ステップ1（同一シート刈り込み）:
      潜在グループがD1, D2を兄弟（検証済み集計構成要素）と認識している場合、
      {D1, D2, ...extras}を含むAI IRのうち、extrasがD1/D2と同一シートにある
      場合は {D1, D2} のみに刈り込む。

    ステップ2（クロスシート刈り込み）:
      ステップ1で刈り込まれなかったIRが、兄弟テーブル以外のテーブルを含む場合、
      全グループ横断の兄弟集合（all_sib_tables）外のテーブルを除去する。
      これにより、複数シートにまたがる誤ったIR（例: 兄弟X_1+X_2に加えて
      別集計軸テーブルX_Dを含むクロスシートIR）も正しく刈り込まれる。
    """
    # 全潜在グループメンバーから兄弟セットを収集する
    sibling_sets: List[frozenset] = []
    for grp in latent_groups:
        for lp, _ in grp.members:
            if len(lp.detected_table_ids) >= 2:
                sibling_sets.append(frozenset(lp.detected_table_ids))

    if not sibling_sets:
        return irs

    # 全兄弟セットの和集合（クロスシート刈り込みに使用）
    all_sib_tables: set = set()
    for ss in sibling_sets:
        all_sib_tables.update(ss)

    def _make_trimmed(base_ir, keep_set):
        """keep_set のテーブルのみ残した IntegrationRecommendation を返す。"""
        new_ids = [t for t in base_ir.table_ids if t in keep_set]
        if len(new_ids) < 2:
            return None
        return IntegrationRecommendation(
            recommendation_id=base_ir.recommendation_id,
            group_name=base_ir.group_name,
            description=base_ir.description,
            table_ids=new_ids,
            new_column_name=base_ir.new_column_name,
            new_column_values={
                k: v for k, v in base_ir.new_column_values.items() if k in keep_set
            },
            reasoning=base_ir.reasoning,
            parent_table_id=base_ir.parent_table_id,
            parent_label_column=base_ir.parent_label_column,
            user_decision=base_ir.user_decision,
            new_column_names=base_ir.new_column_names,
            new_column_multi_values={
                k: v
                for k, v in base_ir.new_column_multi_values.items()
                if k in keep_set
            },
            axis_parent_table_ids=base_ir.axis_parent_table_ids,
            axis_parent_label_columns=base_ir.axis_parent_label_columns,
        )

    result = []
    for ir in irs:
        ir_set = frozenset(ir.table_ids)
        trimmed_ir = ir

        # ── ステップ1: 同一シート内の余分なテーブルを刈り込む ────────────────
        for sib_set in sibling_sets:
            if not (sib_set < ir_set and len(sib_set) >= 2):
                continue
            extras = ir_set - sib_set
            sib_sheets = {
                tables_dict[t].sheet_name for t in sib_set if t in tables_dict
            }
            extra_sheets = {
                tables_dict[t].sheet_name for t in extras if t in tables_dict
            }
            if not sib_sheets or not (extra_sheets <= sib_sheets):
                continue  # クロスsheetのextras — ステップ2で処理
            clipped = _make_trimmed(ir, sib_set)
            if clipped is not None:
                trimmed_ir = clipped
            break  # 最初に一致した兄弟セットを使用する

        # ── ステップ2: クロスシートIRから非兄弟テーブルを除去 ────────────────
        # ステップ1で刈り込まれていない場合のみ実行する。
        # IRが兄弟テーブルと非兄弟テーブルの両方を含む場合、
        # 全グループ横断の兄弟集合に含まれないテーブルを除去する。
        # （例: 複数シートにまたがる {X_1_A, X_2_A, X_D_A, X_1_B, X_2_B, X_D_B}
        #        → 非兄弟の X_D_A, X_D_B を除去して {X_1_A, X_2_A, X_1_B, X_2_B} にする）
        if trimmed_ir is ir:
            cur_set = frozenset(trimmed_ir.table_ids)
            in_sib = cur_set & all_sib_tables
            out_sib = cur_set - all_sib_tables
            if in_sib and out_sib:
                clipped = _make_trimmed(trimmed_ir, all_sib_tables)
                if clipped is not None:
                    trimmed_ir = clipped

        result.append(trimmed_ir)
    return result


def _render_latent_group_card(group: LatentTableGroup, tables_dict: dict) -> None:
    """潜在テーブルグループ（全sheet横断の提案 + 派生）の統合カードをレンダリングする。"""
    gk = group.group_key
    if "latent_group_decisions" not in st.session_state:
        st.session_state.latent_group_decisions = {}
    if gk not in st.session_state.latent_group_decisions:
        st.session_state.latent_group_decisions[gk] = True

    # st.radio() が呼ばれる前にradioウィジェットのキーを事前初期化する。
    # これにより、Streamlitの「index vs. key」競合による1rerun遅延を回避する:
    # `key=` と共に `index=` を使用すると、レンダリング時に古い latent_group_decisions
    # 値からindexが再計算されるため、ユーザーが変更した後もStreamlitがウィジェットを
    # index値にリセットしてしまう可能性がある。
    _radio_key = f"radio_latent_group_{gk}"
    if _radio_key not in st.session_state:
        st.session_state[_radio_key] = (
            "✅ 追加する"
            if st.session_state.latent_group_decisions.get(gk, True)
            else "❌ 追加しない"
        )

    _NOTE_TYPE_LABEL = {
        "aggregation": ("集計注記", "🧮"),
        "exclusion": ("除外注記", "➖"),
        "reference": ("参照注記", "🔗"),
        "general": ("注記", "📝"),
    }
    type_label, type_icon = _NOTE_TYPE_LABEL.get(group.note_type, ("注記", "📝"))
    missing_label = "・".join(group.missing_names)
    n_sheets = len(group.members)
    n_derived = sum(1 for _, dlt in group.members if dlt is not None)

    with st.container(border=True):
        sheet_badge = (
            f"<span style='font-size:0.72rem;color:#607090;margin-left:8px;'>"
            f"（{n_sheets} シート）</span>"
            if n_sheets > 1
            else ""
        )
        st.markdown(
            f"#### {type_icon} {missing_label}（差分推定）{sheet_badge}",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='background:#0d1520;border-left:3px solid #2a4060;"
            f"border-radius:0 6px 6px 0;padding:8px 14px;"
            f"font-size:0.82rem;color:#7090b0;margin:6px 0'>"
            f"{group.note_text}</div>",
            unsafe_allow_html=True,
        )

        # 代表メンバーの集計テーブル名（ソーステーブル = 注記の記載元 = 集計親）
        _rep_source_title = group.members[0][0].source_title if group.members else ""
        c_det, c_miss = st.columns(2)
        with c_det:
            st.markdown("**検出テーブル**")
            if _rep_source_title:
                st.markdown(f"📊 {_rep_source_title}（集計対象）")
            for name in group.detected_names:
                st.markdown(f"✅ {name}")
        with c_miss:
            st.markdown("**潜在テーブル候補**")
            for name in group.missing_names:
                st.markdown(
                    f"<span style='color:#e08080'>❓ {name}</span>",
                    unsafe_allow_html=True,
                )

        if n_derived > 0:
            derived_members = [
                (lp, dlt) for lp, dlt in group.members if dlt is not None
            ]
            # 代表: 最初のDLTを直接表示
            _render_derived_before_after(derived_members[0][1], tables_dict)
            # その他: 折りたたみexpander
            if len(derived_members) > 1:
                others = derived_members[1:]
                with st.expander(f"同様の推定 他 {len(others)} 件", expanded=False):
                    for lp, dlt in others:
                        st.markdown(f"**{lp.source_title}**")
                        _render_derived_before_after(dlt, tables_dict)
                        st.divider()

        st.divider()
        _splitter_marker(f"s4-lg-{gk}")
        c_info, c_dec = st.columns([2, 1])
        with c_info:
            st.markdown("**💡 推奨理由**")
            st.caption(group.members[0][0].reasoning)
        with c_dec:
            st.markdown("<br>", unsafe_allow_html=True)
            decision = st.radio(
                "この潜在テーブルを追加しますか？",
                ["✅ 追加する", "❌ 追加しない"],
                horizontal=True,
                key=_radio_key,
            )
            st.session_state.latent_group_decisions[gk] = decision == "✅ 追加する"
            if decision == "✅ 追加する" and n_derived > 0:
                st.caption("💡 下の統合テーブル案に自動反映されます")


def _render_unified_ir_card(
    ir: IntegrationRecommendation,
    tables_for_render: dict,
    is_auto: bool = False,
) -> None:
    """統合推奨カードをレンダリングする — AI提案と自動生成の両方に対応する。"""
    rec_id = ir.recommendation_id
    dec_store = "latent_auto_int_decisions" if is_auto else "integration_decisions"
    if dec_store not in st.session_state:
        st.session_state[dec_store] = {}
    if rec_id not in st.session_state[dec_store]:
        st.session_state[dec_store][rec_id] = True

    with st.container(border=True):
        st.markdown(f"#### {ir.group_name}")
        if is_auto:
            st.markdown(
                "<span style='display:inline-block;background:#1a3a1a;border:1px solid #2a6a2a;"
                "border-radius:12px;padding:2px 10px;font-size:0.72rem;color:#7aba7a;"
                "margin-bottom:8px;'>💡 潜在テーブル追加による自動提案</span>",
                unsafe_allow_html=True,
            )
        st.markdown(f"_{ir.description}_")

        _render_integration_before_after(ir, tables_for_render)

        st.divider()
        _splitter_marker(f"s4-ir-{rec_id}")
        c_info, c_dec = st.columns([2, 1])
        with c_info:
            _col_names = ir.new_column_names or [ir.new_column_name]
            _multi_vals = ir.new_column_multi_values or {}
            st.markdown(f"**追加列名**: {', '.join(f'`{n}`' for n in _col_names)}")
            for tid in ir.table_ids:
                vals = _multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
                val_str = " / ".join(str(v) for v in vals)
                if is_auto:
                    t_entry = tables_for_render.get(tid)
                    lbl = f"（{t_entry.title}）" if t_entry and t_entry.title else ""
                    st.markdown(f"  - `{tid}`{lbl} → **{val_str}**")
                else:
                    st.markdown(f"  - `{tid}` → **{val_str}**")
            st.caption(f"💡 推奨理由: {ir.reasoning}")
        with c_dec:
            st.markdown("<br>", unsafe_allow_html=True)
            _ir_radio_key = f"radio_{rec_id}"
            # index= を使うと 2クリック問題が起きるため、事前初期化 + index なし で対応する
            if _ir_radio_key not in st.session_state:
                st.session_state[_ir_radio_key] = (
                    "✅ 統合する"
                    if st.session_state[dec_store].get(rec_id, True)
                    else "❌ 統合しない"
                )
            decision = st.radio(
                "この統合を実施しますか？",
                ["✅ 統合する", "❌ 統合しない"],
                horizontal=True,
                key=_ir_radio_key,
            )
            st.session_state[dec_store][rec_id] = decision == "✅ 統合する"


def step4():
    st.header("✅ ステップ 5 : 新規テーブル案生成")

    components.html(
        """<script>
        (function(){
            Object.keys(localStorage)
                .filter(function(k){ return k.startsWith('split-s4-'); })
                .forEach(function(k){ localStorage.removeItem(k); });
        })();
        </script>""",
        height=0,
    )

    analysis: AIAnalysisResult = st.session_state.ai_analysis

    if (
        st.session_state.auto_processing
        and st.session_state.get("run_mode") == "fullauto"
    ):
        with st.spinner("新規テーブル案・マスタ案を自動設定中..."):
            _build_final_tables()
        st.session_state.step = 6
        st.rerun()
    elif st.session_state.auto_processing:
        # セミオート: ユーザーが提案を確認できるようここで停止する
        st.session_state.auto_processing = False
        st.session_state.auto_completed = True

    tables_dict = {t.table_id: t for t in st.session_state.detected_tables}

    # ── 潜在グループを計算する（複数のセクションで使用） ───────────────────
    _latent_proposals = find_latent_tables(st.session_state.detected_tables)
    _derived_latent = derive_latent_tables(st.session_state.detected_tables)
    _latent_groups = group_latent_proposals(_latent_proposals, _derived_latent)

    # 拡張 tables_dict: 各DLT用の仮想 DetectedTable エントリ
    _ext_tables = dict(tables_dict)
    for _dlt in _derived_latent:
        _vt = _dlt_virtual_table(_dlt, tables_dict)
        if _vt is not None:
            _ext_tables[_dlt.proposal_id] = _vt

    # 暫定刈り込み: 早期の「表示するものなし」チェックと has_masters の計算にのみ使用する。
    # 完全な動的計算は後で、潜在グループカードがレンダリングされた後（決定が更新された後）に実行する。
    _ai_irs_prelim = _clip_ai_irs_by_latent_groups(
        analysis.integration_recommendations, _latent_groups, tables_dict
    )
    has_integrations = bool(_ai_irs_prelim) or bool(_latent_groups)
    has_masters = bool(_collect_unique_master_specs(_ai_irs_prelim, tables_dict))
    has_latent = bool(_latent_groups)

    if not has_integrations and not has_masters and not has_latent:
        st.info("新規テーブルの生成推奨はありません。このステップはスキップします。")
        c1, c2 = st.columns([1, 4])
        with c1:
            st.button("← 戻る", on_click=_go_to, args=(4,))
        with c2:
            st.button(
                "次へ：テーブル選択 →",
                type="primary",
                use_container_width=True,
                on_click=_build_and_go_step6,
            )
        return

    # ── セクション 1: 潜在テーブル推定（最初に表示） ──────────────────────────
    if has_latent:
        st.subheader("🔢 潜在テーブル推定")
        st.caption(
            "テーブルの注記に記載された集計関係を分析し、未検出テーブルを差分計算で推定しました。"
            "「追加する」を選択すると下の統合テーブル案に自動反映されます。"
        )
        for _group in _latent_groups:
            _render_latent_group_card(_group, tables_dict)

    # ── セクション 2: 統合テーブル ────────────────────────────────────────────
    # auto-IRは現在の潜在グループ決定を動的に反映する
    _auto_irs_flat = _build_auto_irs_from_latent(
        _latent_groups, _ext_tables, tables_dict
    )

    # AI IRを刈り込む: 検証済み兄弟セットに属さない同一sheet内テーブルを除外する。
    # 例: 潜在検出がT2+T3を兄弟と判定した場合、{T2,T3,T4}からX_D（T4）を除外する。
    _ai_irs_trimmed = _clip_ai_irs_by_latent_groups(
        analysis.integration_recommendations, _latent_groups, tables_dict
    )

    # 実検出テーブルが全て承認済みauto-IRに含まれるAI IRを抑制する。
    # 潜在X-3が承認された場合: auto-IRの実={T2,T3} ⊆ AI IR {T2,T3} → AI IRを抑制する。
    # （auto-IRが潜在テーブルを追加することでAI IRを置き換える。）
    _superseded_ai_ids: set = set()
    for _sup_ir, _ in _auto_irs_flat:
        # auto-IR内の実（仮想でない）テーブルID
        _sup_real = {t for t in _sup_ir.table_ids if t in tables_dict}
        if len(_sup_real) < 2:
            continue
        for _ai_ir in _ai_irs_trimmed:
            # AI IRの全テーブルがauto-IRの実テーブルに含まれる場合は抑制する
            # （auto-IRがDLTを加えてAI IRを上書きするため）
            if set(_ai_ir.table_ids).issubset(_sup_real):
                _superseded_ai_ids.add(_ai_ir.recommendation_id)

    # マージ: auto-IRを先頭に（潜在テーブルが承認された際に上部に表示されるよう）、
    # 次に抑制されていないAI IRを続ける。
    _all_irs_flagged: list = [(ir, True) for ir, _ in _auto_irs_flat]
    _all_irs_flagged += [
        (ir, False)
        for ir in _ai_irs_trimmed
        if ir.recommendation_id not in _superseded_ai_ids
    ]
    _is_auto_map: dict = {ir.recommendation_id: flag for ir, flag in _all_irs_flagged}
    _all_irs = [ir for ir, _ in _all_irs_flagged]

    if _all_irs:
        if has_latent:
            st.divider()
        st.subheader("🔀 統合テーブル")
        st.caption(
            "各統合について実施するかどうかをお選びください。"
            "潜在テーブルを「追加する」にすると、関連する統合提案が自動で追加されます。"
        )

        # 結合リスト全体で列シグネチャによりグループ化する
        _ir_groups = _group_irs_by_similarity(_all_irs, _ext_tables)

        for _grp in _ir_groups:
            _rep = _grp[0]
            _similar = _grp[1:]
            _render_unified_ir_card(
                _rep,
                _ext_tables,
                is_auto=_is_auto_map.get(_rep.recommendation_id, False),
            )
            if _similar:
                _rep_cols = _rep.new_column_names or [_rep.new_column_name]
                _axes_lbl = " × ".join(_rep_cols)
                with st.expander(
                    f"同様の統合 他 {len(_similar)} 件（{_axes_lbl} 軸）",
                    expanded=False,
                ):
                    for _ir in _similar:
                        _render_unified_ir_card(
                            _ir,
                            _ext_tables,
                            is_auto=_is_auto_map.get(_ir.recommendation_id, False),
                        )

    # ── セクション 3: マスタ自動生成 ─────────────────────────────────────────
    # 動的に再計算する: 刈り込み済みAI IR（X_D除外）+ 承認済みauto-IR
    # （axis_parent_table_idsが設定されているため、マスタ生成がX_3マッピングを生成できる）を使用する。
    # 仮想DLTテーブル（X_3）を解決するため _ext_tables を使用する。
    _master_ir_list = [ir for ir, _ in _auto_irs_flat] + [
        ir for ir in _ai_irs_trimmed if ir.recommendation_id not in _superseded_ai_ids
    ]
    unique_master_specs = _collect_unique_master_specs(_master_ir_list, _ext_tables)
    _has_masters_now = bool(unique_master_specs)

    if _has_masters_now:
        st.divider()
        st.subheader("🗂️ マスタ自動生成")
        st.caption(
            "元データ内の上位集計テーブルと各子テーブルの軸値の対応関係から、"
            "「下位区分 → 上位区分」の対応マスタを自動生成できます。"
            "このマスタを統合テーブルに結合することで、上位区分での再集計が可能になります。"
        )

        for ir, spec in unique_master_specs:
            child_col = spec["child_col"]
            parent_col = spec["parent_col"]
            axis_idx = spec.get("axis_idx", 0)
            dm_key = f"dim_master_{ir.recommendation_id}_ax{axis_idx}"
            if dm_key not in st.session_state.master_decisions:
                st.session_state.master_decisions[dm_key] = True

            master_rows = [
                {child_col: ck, parent_col: pv} for ck, pv in spec["mapping"].items()
            ]
            master_preview_df = pd.DataFrame(master_rows)

            with st.container(border=True):
                title = f"{child_col} × {parent_col} マスタ"
                st.markdown(f"#### {title}")
                st.markdown(
                    f"元データの上位集計テーブル `{spec['parent_id']}` と各子テーブルの"
                    f"「`{child_col}`」値の対応関係から生成するマスタテーブル。"
                    f"統合テーブルに結合することで「`{parent_col}`」単位での再集計が可能になります。"
                )

                _splitter_marker(f"s4-dm-{ir.recommendation_id}-ax{axis_idx}")
                c_prev, c_info = st.columns([1, 1])
                with c_prev:
                    st.caption("生成されるマスタのプレビュー（全件）")
                    st.dataframe(
                        master_preview_df, use_container_width=True, hide_index=True
                    )
                with c_info:
                    st.markdown(f"**キー列（結合用）**: `{child_col}`")
                    st.markdown(f"**上位区分列**: `{parent_col}`")
                    st.markdown(f"**参照元テーブル**: `{spec['parent_id']}`")
                    st.markdown(f"**行数**: {len(master_rows)} 行")
                    st.caption(
                        f"統合テーブルの `{child_col}` 列でこのマスタと結合すると、"
                        f"各行の `{parent_col}` を参照して上位集計できます。"
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                    decision = st.radio(
                        "このマスタを生成しますか？",
                        ["✅ マスタを作成する", "❌ マスタを作成しない"],
                        horizontal=True,
                        key=f"radio_dm_{ir.recommendation_id}_ax{axis_idx}",
                        index=(
                            0
                            if st.session_state.master_decisions.get(dm_key, True)
                            else 1
                        ),
                    )
                    st.session_state.master_decisions[dm_key] = (
                        decision == "✅ マスタを作成する"
                    )

    # ── ナビゲーション ────────────────────────────────────────────────────────
    _inject_splitter_js()
    st.divider()
    c1, c2 = st.columns([1, 4])
    with c1:
        st.button("← 戻る", on_click=_go_to, args=(4,))
    with c2:
        st.button(
            "次へ：テーブル選択 →",
            type="primary",
            use_container_width=True,
            on_click=_build_and_go_step6,
        )


# ---------------------------------------------------------------------------
# 最終テーブルリストを構築する（Step 4 離脱時に呼び出す）
# ---------------------------------------------------------------------------


def _build_final_tables():
    analysis: AIAnalysisResult = st.session_state.ai_analysis
    tables_dict = {t.table_id: t for t in st.session_state.detected_tables}
    ta_by_id = {ta.table_id: ta for ta in analysis.table_analyses}

    final: Dict[str, dict] = {}
    integrated_ids: Set[str] = set()
    seen_master_sigs: Set = set()  # 同一の child→parent ラベルマップを持つマスタを重複排除する

    # ── 潜在グループの検出（1回のみ）────────────────────────────────────────
    _bft_proposals = find_latent_tables(st.session_state.detected_tables)
    _bft_derived = derive_latent_tables(st.session_state.detected_tables)
    _bft_groups = group_latent_proposals(_bft_proposals, _bft_derived)
    _bft_ext = dict(tables_dict)
    for _dlt in _bft_derived:
        _vt = _dlt_virtual_table(_dlt, tables_dict)
        if _vt:
            _bft_ext[_dlt.proposal_id] = _vt

    # ── 承認済みauto-IRによって置き換えられるAI IRを計算する（step4と同じロジック）──
    _bft_ai_irs_trimmed = _clip_ai_irs_by_latent_groups(
        analysis.integration_recommendations, _bft_groups, tables_dict
    )
    _bft_auto_irs = _build_auto_irs_from_latent(_bft_groups, _bft_ext, tables_dict)
    _superseded_in_bft: set = set()
    for _bft_auto_ir, _ in _bft_auto_irs:
        if not st.session_state.get("latent_auto_int_decisions", {}).get(
            _bft_auto_ir.recommendation_id, True
        ):
            continue
        _auto_real = {t for t in _bft_auto_ir.table_ids if t in tables_dict}
        if len(_auto_real) < 2:
            continue
        for _ai_check in _bft_ai_irs_trimmed:
            # AI IRの全テーブルがauto-IRの実テーブルに含まれる場合は抑制する
            if set(_ai_check.table_ids).issubset(_auto_real):
                _superseded_in_bft.add(_ai_check.recommendation_id)

    _lg_dec = st.session_state.get("latent_group_decisions", {})
    _lai_dec = st.session_state.get("latent_auto_int_decisions", {})

    # ── 潜在グループ: DLT + auto-IR統合（step4と同じ順序で先に処理）────────
    for _grp in _bft_groups:
        _gk = _grp.group_key
        # グループ決定: 新しいキーを優先し、DLTごとの derived_decisions にフォールバック
        if _gk in _lg_dec:
            _grp_accepted = _lg_dec[_gk]
        else:
            _grp_accepted = any(
                st.session_state.get("derived_decisions", {}).get(dlt.proposal_id, True)
                for _, dlt in _grp.members
                if dlt is not None
            )
        if not _grp_accepted:
            continue

        for _lp, _dlt in _grp.members:
            if _dlt is None:
                continue
            # DLTは統合の統合元として使用する中間テーブルのため、
            # 単体では最小粒度データではなく非推奨テーブルとして扱う。
            # クロスシート集計除外で auto-IR の table_ids に含まれないDLTも
            # 同様に非推奨とする。
            final[_dlt.proposal_id] = {
                "df": _dlt.df,
                "display_name": _dlt.derived_name,
                "description": _dlt.derivation_formula,
                "reasoning": _dlt.reasoning,
                "is_integrated": False,
                "source_ids": _dlt.source_display_order,
                "recommended": False,
                "granularity": "detail",
                "is_minimum": False,
                "is_master": False,
            }

        # 承認済みauto-IR統合テーブル
        _auto_irs_bft = _build_auto_irs_from_latent([_grp], _bft_ext, tables_dict)
        for _auto_ir, _ in _auto_irs_bft:
            _rec_id = _auto_ir.recommendation_id
            if not _lai_dec.get(_rec_id, True):
                continue

            _col_names_bft = _auto_ir.new_column_names or [_auto_ir.new_column_name]
            _multi_vals_bft = _auto_ir.new_column_multi_values or {}
            _redundant_bft = _detect_redundant_axes(
                _col_names_bft, _multi_vals_bft, _auto_ir,
                {t.table_id: t for t in st.session_state.get("detected_tables", [])}
            )

            _frames_bft = []
            for _tid in _auto_ir.table_ids:
                _t = _bft_ext.get(_tid)
                if _t and _t.effective_df is not None and not _t.effective_df.empty:
                    _df_copy = _t.effective_df.copy()
                    _vals = _multi_vals_bft.get(_tid) or [
                        _auto_ir.new_column_values.get(_tid, "")
                    ]
                    for _ci in range(len(_col_names_bft) - 1, -1, -1):
                        if _ci in _redundant_bft:
                            continue
                        _val = _vals[_ci] if _ci < len(_vals) else ""
                        _df_copy.insert(0, _col_names_bft[_ci], _val)
                    _frames_bft.append(_df_copy)

            if not _frames_bft:
                continue
            try:
                _merged_bft = pd.concat(_frames_bft, ignore_index=True)
            except Exception:
                continue

            # 統合元テーブルを「統合済み」として扱う
            # （「統合する」選択時は最小粒度データとして個別表示しない）
            for _tid in _auto_ir.table_ids:
                if _tid in tables_dict:
                    integrated_ids.add(_tid)
                elif _tid in final:
                    # DLT（仮想テーブル）は tables_dict にないため integrated_ids で管理できない。
                    # 代わりに final エントリを非推奨・非最小粒度に降格させ、
                    # 非推奨テーブルの折りたたみセクションに移動させる。
                    final[_tid]["is_minimum"] = False
                    final[_tid]["recommended"] = False

            _int_key = f"latent_auto_int_{_rec_id}"
            final[_int_key] = {
                "df": _merged_bft,
                "display_name": _auto_ir.group_name,
                "description": _auto_ir.description,
                "reasoning": _auto_ir.reasoning,
                "is_integrated": True,
                "source_ids": _auto_ir.table_ids,
                "recommended": True,
                "granularity": "detail",
                "is_minimum": True,
                "is_master": False,
                "new_col_names": _col_names_bft,
            }

            # auto-IRに対してもマスタを生成する（AI IRと同じロジック）
            for _spec in _derive_master_specs_for_ir(_auto_ir, tables_dict):
                _master_sig = _master_signature(_spec)
                if _master_sig in seen_master_sigs:
                    continue
                seen_master_sigs.add(_master_sig)
                _axis_idx = _spec.get("axis_idx", 0)
                _dm_key = f"dim_master_{_rec_id}_ax{_axis_idx}"
                if not st.session_state.master_decisions.get(_dm_key, True):
                    continue
                _child_col = _spec["child_col"]
                _parent_col = _spec["parent_col"]
                _master_rows = [
                    {_child_col: ck, _parent_col: pv}
                    for ck, pv in _spec["mapping"].items()
                ]
                if _master_rows:
                    _master_df = pd.DataFrame(_master_rows)
                    final[_dm_key] = {
                        "df": _master_df,
                        "display_name": f"{_child_col} × {_parent_col} マスタ",
                        "description": (
                            f"元データの上位集計テーブル（{_spec['parent_id']}）と各子テーブルの"
                            f"「{_child_col}」値の対応関係から生成したマスタテーブル。"
                            f"統合テーブルを「{_child_col}」で結合すると「{_parent_col}」単位での再集計が可能になる。"
                        ),
                        "reasoning": (
                            f"元データ内の上位集計テーブル {_spec['parent_id']} の階層関係から自動生成。"
                            f"統合テーブル自体ではなく、ソースデータの親子関係をもとにした対応表。"
                        ),
                        "is_integrated": False,
                        "source_ids": [_spec["parent_id"]] + _auto_ir.table_ids,
                        "recommended": True,
                        "granularity": "master",
                        "is_minimum": False,
                        "is_master": True,
                    }

    # ── AI IR統合（抑制・不承認のものはスキップ）────────────────────────────
    for ir in _bft_ai_irs_trimmed:
        if ir.recommendation_id in _superseded_in_bft:
            continue  # 派生テーブルを含むauto-IRによって置き換えられた
        if not st.session_state.integration_decisions.get(ir.recommendation_id, True):
            continue

        col_names = getattr(ir, "new_column_names", []) or [ir.new_column_name]
        multi_vals = getattr(ir, "new_column_multi_values", {}) or {}
        redundant_axes = _detect_redundant_axes(col_names, multi_vals, ir, tables_dict)

        frames = []
        for tid in ir.table_ids:
            t = tables_dict.get(tid)
            if t and t.effective_df is not None and not t.effective_df.empty:
                df_copy = t.effective_df.copy()
                vals = multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
                for i in range(len(col_names) - 1, -1, -1):
                    if i in redundant_axes:
                        continue
                    val = vals[i] if i < len(vals) else ""
                    df_copy.insert(0, col_names[i], val)
                frames.append(df_copy)
                integrated_ids.add(tid)

        if not frames:
            continue
        try:
            merged_df = pd.concat(frames, ignore_index=True)
        except Exception:
            for tid in ir.table_ids:
                integrated_ids.discard(tid)
            continue

        src_ta = next((ta_by_id[tid] for tid in ir.table_ids if tid in ta_by_id), None)
        key = f"integrated_{ir.recommendation_id}"
        final[key] = {
            "df": merged_df,
            "display_name": ir.group_name,
            "description": ir.description,
            "reasoning": ir.reasoning,
            "is_integrated": True,
            "source_ids": ir.table_ids,
            "recommended": True,
            "granularity": src_ta.granularity_level if src_ta else "detail",
            "is_minimum": src_ta.is_minimum_granularity_candidate if src_ta else False,
            "is_master": src_ta.is_master_table if src_ta else False,
            "new_col_names": [c for i, c in enumerate(col_names) if i not in redundant_axes],
        }

        # 親を持つ全軸に対してディメンションマスタを自動生成する。
        for spec in _derive_master_specs_for_ir(ir, tables_dict):
            master_sig = _master_signature(spec)
            if master_sig in seen_master_sigs:
                continue
            seen_master_sigs.add(master_sig)
            axis_idx = spec.get("axis_idx", 0)
            dm_key = f"dim_master_{ir.recommendation_id}_ax{axis_idx}"
            if not st.session_state.master_decisions.get(dm_key, True):
                continue
            child_col = spec["child_col"]
            parent_col = spec["parent_col"]
            master_rows = [
                {child_col: ck, parent_col: pv} for ck, pv in spec["mapping"].items()
            ]
            if master_rows:
                master_df = pd.DataFrame(master_rows)
                final[dm_key] = {
                    "df": master_df,
                    "display_name": f"{child_col} × {parent_col} マスタ",
                    "description": (
                        f"元データの上位集計テーブル（{spec['parent_id']}）と各子テーブルの"
                        f"「{child_col}」値の対応関係から生成したマスタテーブル。"
                        f"統合テーブルを「{child_col}」で結合すると「{parent_col}」単位での再集計が可能になる。"
                    ),
                    "reasoning": (
                        f"元データ内の上位集計テーブル {spec['parent_id']} の階層関係から自動生成。"
                        f"統合テーブル自体ではなく、ソースデータの親子関係をもとにした対応表。"
                    ),
                    "is_integrated": False,
                    "source_ids": [spec["parent_id"]] + ir.table_ids,
                    "recommended": True,
                    "granularity": "master",
                    "is_minimum": False,
                    "is_master": True,
                }

    # 個別の非統合テーブル
    for ta in analysis.table_analyses:
        if ta.table_id in integrated_ids:
            continue
        t = tables_dict.get(ta.table_id)
        if not t or t.effective_df is None or t.effective_df.empty:
            continue
        final[ta.table_id] = {
            "df": t.effective_df,
            "display_name": ta.display_name,
            "description": ta.description,
            "reasoning": ta.reasoning,
            "is_integrated": False,
            "source_ids": [ta.table_id],
            "recommended": ta.recommended_for_extraction,
            "granularity": ta.granularity_level,
            "is_minimum": ta.is_minimum_granularity_candidate,
            "is_master": ta.is_master_table,
        }

    # セーフティネット — 分析対象外のテーブル
    analyzed_ids = set(ta_by_id.keys())
    for t in st.session_state.detected_tables:
        if t.table_id in analyzed_ids or t.table_id in integrated_ids:
            continue
        if t.effective_df is None or t.effective_df.empty:
            continue
        final[t.table_id] = {
            "df": t.effective_df,
            "display_name": t.table_id,
            "description": "テーブル関係分析対象外のテーブル",
            "reasoning": "自動検出されましたが テーブル関係分析から除外されました",
            "is_integrated": False,
            "source_ids": [t.table_id],
            "recommended": False,
            "granularity": "unknown",
            "is_minimum": False,
            "is_master": False,
        }

    st.session_state.final_tables = final
    # 推奨テーブルを事前選択する
    st.session_state.selected_ids = {
        tid for tid, info in final.items() if info["recommended"]
    }


# ---------------------------------------------------------------------------
# Step 5 — テーブル選択
# ---------------------------------------------------------------------------

# 各新規列にはそれぞれ異なるパレットを割り当てる: (cell_bg, cell_fg, hdr_bg, hdr_fg)
# hdr_bgは視覚的なコントラストのためcell_bgより明らかに暗い色を使用する。
_COL_PALETTES = [
    ("#d0faf0", "#0d4b36", "#2aab87", "#ffffff"),  # teal
    ("#fdebd0", "#7a3a0a", "#d4793a", "#ffffff"),  # orange
    ("#d8e8fd", "#0d2f7f", "#4a7de0", "#ffffff"),  # blue
    ("#f0d5f8", "#5b0f7f", "#9b4fc4", "#ffffff"),  # purple
]

# 統合プレビュー用の軸ごとのカラーファミリー。
# 各ファミリーは8つの明確に区別できる色を持つ（シェードではなく個別の色）: (hdr_bg, hdr_fg, cell_bg, cell_fg)
# Axis 0 = WARM ファミリー（赤/オレンジ/ピンク/アンバー — 値ごとに視覚的に区別可能）
# Axis 1 = COOL ファミリー（青/ティール/紫/シアン — 値ごとに視覚的に区別可能）
# Axis 2 = NATURE ファミリー（緑/ライム/オリーブ/フォレスト）
# Axis 3 = ACCENT ファミリー（ディープオレンジ/インディゴ/ローズ/ミント）
_AXIS_FAMILIES: list = [
    [  # axis-0: WARM — 各値は明確に異なるウォームな色相
        ("#e74c3c", "#fff", "#fadbd8", "#7b0c0c"),  # red
        ("#e67e22", "#fff", "#fae5d3", "#7a3a0a"),  # orange
        ("#e91e63", "#fff", "#fce4ec", "#7c0024"),  # magenta/pink
        ("#f39c12", "#1a1a1a", "#fef9e7", "#5d3a00"),  # amber
        ("#c0392b", "#fff", "#f5b7b1", "#6e1006"),  # dark red
        ("#d35400", "#fff", "#fad5b0", "#6b2800"),  # burnt orange
        ("#ec407a", "#fff", "#fce8ef", "#7a0036"),  # rose
        ("#f57c00", "#fff", "#fff0d9", "#6b3800"),  # deep orange
    ],
    [  # axis-1: COOL — 各値は明確に異なるクールな色相
        ("#2980b9", "#fff", "#d6eaf8", "#0d2f6e"),  # blue
        ("#1abc9c", "#fff", "#d1f2eb", "#0a4038"),  # teal
        ("#8e44ad", "#fff", "#e8daef", "#4a1a72"),  # purple
        ("#00acc1", "#fff", "#e0f7fa", "#00474f"),  # cyan
        ("#3f51b5", "#fff", "#e8eaf6", "#1a237e"),  # indigo
        ("#16a085", "#fff", "#cde8e4", "#0a3630"),  # dark teal
        ("#6c3483", "#fff", "#e4d0ef", "#3b1260"),  # deep purple
        ("#0288d1", "#fff", "#e1f1fb", "#013d6e"),  # light blue
    ],
    [  # axis-2: NATURE — グリーン/ライム/フォレスト
        ("#27ae60", "#fff", "#d5f5e3", "#0b3c2e"),  # green
        ("#8bc34a", "#1a1a1a", "#f1f8e9", "#33691e"),  # lime green
        ("#00695c", "#fff", "#cce5e2", "#00332e"),  # forest
        ("#558b2f", "#fff", "#dcedc8", "#2a4200"),  # olive
        ("#2e7d32", "#fff", "#c8e6c9", "#0a2d0b"),  # dark green
        ("#76ff03", "#1a1a1a", "#f4ffe0", "#3a5f00"),  # neon lime
        ("#1b5e20", "#fff", "#c3e8c4", "#0a1c0a"),  # deep forest
        ("#aed581", "#1a1a1a", "#ecf6dc", "#3a5f00"),  # light olive
    ],
    [  # axis-3: ACCENT — 各種鮮やかな色
        ("#ff5722", "#fff", "#fbe9e7", "#bf360c"),  # deep orange
        ("#9c27b0", "#fff", "#f3e5f5", "#4a148c"),  # deep purple
        ("#009688", "#fff", "#e0f2f1", "#004d40"),  # teal accent
        ("#ffc107", "#1a1a1a", "#fff8e1", "#6b4900"),  # yellow accent
        ("#5c6bc0", "#fff", "#e8eaf6", "#1a237e"),  # indigo accent
        ("#ef5350", "#fff", "#ffebee", "#7b0000"),  # red accent
        ("#26a69a", "#fff", "#e0f2f1", "#00352f"),  # teal light
        ("#ab47bc", "#fff", "#f3e5f5", "#4a0072"),  # purple accent
    ],
]


def _styled_df(df: "pd.DataFrame", new_cols: list) -> "pd.io.formats.style.Styler":
    """new_colsをハイライトした Pandas Styler を返す。各列に異なるパレットを割り当て、
    列ヘッダーはセル背景より暗くレンダリングされる。"""
    df_str = df.astype(str)
    valid = [c for c in new_cols if c in df_str.columns]
    styler = df_str.style

    if not valid:
        return styler

    col_pal = {c: _COL_PALETTES[i % len(_COL_PALETTES)] for i, c in enumerate(valid)}

    # ── セル背景 ──
    for col, (cbg, cfg, _, _) in col_pal.items():
        styler = styler.set_properties(
            subset=[col], **{"background-color": cbg, "color": cfg}
        )

    # ── 列ヘッダー背景 ──
    # Method A: apply_index (pandas ≥ 1.4, Streamlit ≥ 1.26)
    def _hdr(idx: "pd.Index") -> list:
        out = []
        for c in idx:
            if c in col_pal:
                _, _, hbg, hfg = col_pal[c]
                out.append(f"background-color: {hbg}; color: {hfg};")
            else:
                out.append("")
        return out

    try:
        styler = styler.apply_index(_hdr, axis="columns")
    except Exception:
        pass

    # Method B: set_table_styles — Pandas HTMLクラスセレクター（th.col_heading.colN）を対象とする。
    # apply_indexを無視するが set_table_styles 経由で注入されたテーブルレベルの
    # CSSには対応しているStreamlitバージョンで機能する。
    tbl_styles = []
    for col, (_, _, hbg, hfg) in col_pal.items():
        try:
            col_idx = list(df_str.columns).index(col)
            tbl_styles.append(
                {
                    "selector": f"th.col_heading.col{col_idx}",
                    "props": f"background-color: {hbg} !important; color: {hfg} !important;",
                }
            )
        except ValueError:
            pass
    if tbl_styles:
        try:
            styler = styler.set_table_styles(tbl_styles, overwrite=False)
        except Exception:
            pass

    return styler


def _render_integration_before_after(
    ir,
    tables_dict: dict,
    compact: bool = False,
    full_df_size: tuple = None,
    source_ids: list = None,
) -> None:
    """構造化された統合前 → 統合後のプレビューをレンダリングする。

    マルチ軸カラーリング: 各軸ディメンションには異なるカラーファミリー
    （緑/青/紫/オレンジ）を割り当てる。各ファミリー内では、ユニークな軸値ごとに
    異なる色を割り当てる。ソーステーブルカードは軸ごとに色付きのピルを表示し、
    統合テーブルは各軸列のセルを値によって色分けする。
    最初の3つのソーステーブルは直接表示され、追加テーブルはexpanderに収納する。

    full_df_size: 実際の統合済みテーブルの (n_rows, n_cols)。
                  統合後プレビューの右下に表示される。
    source_ids:   expander の後に表示するソーステーブルIDのリスト。
    """
    col_names = getattr(ir, "new_column_names", []) or [ir.new_column_name]
    multi_vals = getattr(ir, "new_column_multi_values", {}) or {}

    # インラインプレビューボックスの可視行の高さ。
    # (a) スクロールで全行を表示し、(b) ネイティブのFullscreenボタンで全レコードを
    # 表示できるよう、常に完全なDataFrameを渡す。
    _BEFORE_VISIBLE = 3  # 統合前カード: スクロール前に表示する行数
    _AFTER_VISIBLE = 10  # 統合後プレビュー: スクロール前に表示する行数
    _ROW_PX = 35  # st.dataframe の1データ行のおよそのpx高さ
    _HDR_PX = 38  # ヘッダー行の高さ

    # ── 軸ごとのユニーク値の順序を構築する ──────────────────────────────────
    axis_val_order: list = [[] for _ in col_names]
    for tid in ir.table_ids:
        vals = multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
        for ai, v in enumerate(vals):
            if ai < len(axis_val_order):
                sv = str(v)
                if sv not in axis_val_order[ai]:
                    axis_val_order[ai].append(sv)

    def _axis_color(ai: int, val: str) -> tuple:
        """指定された軸+値に対する (hdr_bg, hdr_fg, cell_bg, cell_fg) を返す。"""
        family = _AXIS_FAMILIES[ai % len(_AXIS_FAMILIES)]
        vals_list = axis_val_order[ai] if ai < len(axis_val_order) else []
        vi = vals_list.index(val) if val in vals_list else 0
        return family[vi % len(family)]

    # ── ヘルパー: ソーステーブルカード（ヘッダー + DataFrame）を1つレンダリングする ──
    import html as _html

    def _src_card(tid: str) -> None:
        t = tables_dict.get(tid)
        vals = multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
        pills = "".join(
            f'<span style="background:{_axis_color(ai, str(v))[0]};'
            f"color:{_axis_color(ai, str(v))[1]};"
            f"padding:2px 9px;border-radius:12px;font-size:0.72rem;font-weight:600;"
            f'margin-right:4px;display:inline-block;margin-bottom:3px;">'
            f"{_html.escape(cn)}:&nbsp;{_html.escape(str(v))}</span>"
            for ai, (cn, v) in enumerate(zip(col_names, vals))
        )
        st.markdown(
            f'<div style="background:#161c2c;border:1px solid #2d3748;'
            f'border-radius:6px 6px 0 0;padding:7px 10px;margin-bottom:0;">'
            f'<div style="font-size:0.7rem;color:#7a8599;margin-bottom:4px;">'
            f"📋&nbsp;{_html.escape(tid)}</div>{pills}</div>",
            unsafe_allow_html=True,
        )
        if t is not None and t.effective_df is not None and not t.effective_df.empty:
            n_rows = len(t.effective_df)
            st.dataframe(
                t.effective_df.astype(str),
                use_container_width=True,
                hide_index=True,
                height=min(
                    n_rows * _ROW_PX + _HDR_PX, _BEFORE_VISIBLE * _ROW_PX + _HDR_PX
                ),
            )
        else:
            st.caption("（データなし）")

    def _src_card_grid(tids: list) -> None:
        cols = st.columns(len(tids), gap="small")
        for i, tid in enumerate(tids):
            with cols[i]:
                _src_card(tid)

    # ════════════════════════════════════════════════════════════════════════
    # 統合前
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:6px 0 14px;">'
        '<div style="width:5px;height:22px;background:#4a7de0;border-radius:3px;flex-shrink:0;"></div>'
        '<span style="font-size:1.05rem;font-weight:800;color:#c8d4e8;letter-spacing:.04em;">統合前</span>'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,rgba(74,125,224,.4),transparent);"></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    SAMPLE_LIMIT = 3
    preview_tids = ir.table_ids[:SAMPLE_LIMIT]
    extra_tids = ir.table_ids[SAMPLE_LIMIT:]

    if preview_tids:
        _src_card_grid(preview_tids)

    if extra_tids:
        with st.expander(f"他 {len(extra_tids)} テーブルを見る", expanded=False):
            n_ex = min(len(extra_tids), 3)
            for start in range(0, len(extra_tids), n_ex):
                _src_card_grid(extra_tids[start : start + n_ex])

    # ── 統合元テーブル一覧 (expander下) ─────────────────────────────────────
    # expander内のst.columns()は避ける — 同じ縦方向コンテナ内のネストした columns は
    # 上部でレンダリングされたソーステーブルグリッドの幅が不均等になる可能性がある。
    if source_ids:
        _INLINE_LIMIT = 4
        if len(source_ids) <= _INLINE_LIMIT:
            st.caption(f"🔗 統合元テーブル一覧: {' ／ '.join(source_ids)}")
        else:
            with st.expander(
                f"🔗 統合元テーブル一覧 （{len(source_ids)} テーブル）", expanded=False
            ):
                for sid in source_ids:
                    st.caption(f"• {sid}")

    # ── 統合処理セパレーター ─────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0;margin:22px 0 18px;">'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,transparent,rgba(39,174,96,.5));"></div>'
        '<div style="border:1.5px solid rgba(39,174,96,.7);border-radius:24px;'
        "padding:6px 22px;margin:0 16px;font-size:1.05rem;font-weight:800;"
        "color:#7FFFD4;letter-spacing:.08em;"
        "background:linear-gradient(135deg,rgba(39,174,96,.12),rgba(26,188,156,.08));"
        'display:flex;align-items:center;gap:8px;white-space:nowrap;">'
        "↓&nbsp;&nbsp;統合処理"
        "</div>"
        '<div style="flex:1;height:1px;background:linear-gradient(to left,transparent,rgba(39,174,96,.5));"></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════════════
    # 統合後
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:6px 0 14px;">'
        '<div style="width:5px;height:22px;background:#27ae60;border-radius:3px;flex-shrink:0;"></div>'
        '<span style="font-size:1.05rem;font-weight:800;color:#c8d4e8;letter-spacing:.04em;">統合後</span>'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,rgba(39,174,96,.4),transparent);"></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    redundant_preview = _detect_redundant_axes(col_names, multi_vals, ir, tables_dict)

    frames = []
    for tid in ir.table_ids:
        t = tables_dict.get(tid)
        if t is not None and t.effective_df is not None and not t.effective_df.empty:
            row = t.effective_df.copy()  # full data — scroll + Fullscreen reveal all rows
            vals = multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
            for ci in range(len(col_names) - 1, -1, -1):
                if ci in redundant_preview:
                    continue
                val = vals[ci] if ci < len(vals) else ""
                row.insert(0, col_names[ci], val)
            frames.append(row)

    if not frames:
        st.caption("（プレビュー生成不可）")
        return

    try:
        combined = pd.concat(frames, ignore_index=True)
        df_str = combined.astype(str)
        valid_cols = [c for c in col_names if c in df_str.columns]

        # 各軸セルをそのカラーファミリー内の値によって色付けする
        def _color_row(row):
            s = pd.Series("", index=row.index)
            for ai, c in enumerate(valid_cols):
                if c in row.index:
                    _, _, cbg, cfg = _axis_color(ai, str(row[c]))
                    s[c] = f"background-color:{cbg};color:{cfg};"
            return s

        styler = df_str.style.apply(_color_row, axis=1) if valid_cols else df_str.style

        # 軸列ヘッダー: 各軸ファミリーのベース（shade 0）ヘッダー色を使用する
        def _hdr_fn(idx):
            out = []
            for c in idx:
                if c in valid_cols:
                    ai = valid_cols.index(c)
                    hbg, hfg = _AXIS_FAMILIES[ai % len(_AXIS_FAMILIES)][0][:2]
                    out.append(f"background-color:{hbg};color:{hfg};font-weight:bold;")
                else:
                    out.append("")
            return out

        try:
            styler = styler.apply_index(_hdr_fn, axis="columns")
        except Exception:
            pass

        tbl_styles = []
        for c in valid_cols:
            try:
                ci = list(df_str.columns).index(c)
                ai = valid_cols.index(c)
                hbg, hfg = _AXIS_FAMILIES[ai % len(_AXIS_FAMILIES)][0][:2]
                tbl_styles.append(
                    {
                        "selector": f"th.col_heading.col{ci}",
                        "props": (
                            f"background-color:{hbg} !important;"
                            f"color:{hfg} !important;font-weight:bold !important;"
                        ),
                    }
                )
            except ValueError:
                pass
        if tbl_styles:
            try:
                styler = styler.set_table_styles(tbl_styles, overwrite=False)
            except Exception:
                pass

        n_data = len(df_str)
        st.dataframe(
            styler,
            use_container_width=True,
            hide_index=True,
            height=min(n_data * _ROW_PX + _HDR_PX, _AFTER_VISIBLE * _ROW_PX + _HDR_PX),
        )
        # 統合済みテーブルのサイズ — プレビューの右下に表示する
        if full_df_size:
            n_rows, n_cols_full = full_df_size
            st.markdown(
                f'<div style="text-align:right;font-size:0.72rem;'
                f'color:#7a8599;margin-top:2px;">'
                f"📊 {n_rows:,} 行 × {n_cols_full} 列</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.caption("（プレビュー生成不可）")


def _render_derived_before_after(
    dlt: "DerivedLatentTable",
    tables_dict: dict,
) -> None:
    """派生潜在テーブルの推定前/推定後ビューをレンダリングする。

    「推定前」は集計親テーブルと各検出済み構成要素を表示する。
    「推定後」は数値的に派生（差し引き）されたテーブルを表示する。
    """
    _ROW_PX = 35
    _HDR_PX = 38
    _BEFORE_VISIBLE = 3
    _AFTER_VISIBLE = 4

    import html as _html

    # ── 推定前 ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:6px 0 14px;">'
        '<div style="width:5px;height:22px;background:#4a7de0;border-radius:3px;flex-shrink:0;"></div>'
        '<span style="font-size:1.05rem;font-weight:800;color:#c8d4e8;letter-spacing:.04em;">推定前（検出テーブル）</span>'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,rgba(74,125,224,.4),transparent);"></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    def _table_card_mini(tid: str, role_label: str, color: str) -> None:
        t = tables_dict.get(tid)
        title = (t.title if t else None) or tid
        st.markdown(
            f'<div style="background:#161c2c;border:1px solid #2d3748;'
            f'border-radius:6px 6px 0 0;padding:7px 10px;margin-bottom:0;">'
            f'<div style="font-size:0.7rem;color:#7a8599;margin-bottom:4px;">📋&nbsp;{_html.escape(tid)}</div>'
            f'<span style="background:{color};color:#fff;padding:2px 9px;border-radius:12px;'
            f'font-size:0.72rem;font-weight:600;">{_html.escape(role_label)}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if t is not None and t.effective_df is not None and not t.effective_df.empty:
            n = len(t.effective_df)
            st.dataframe(
                t.effective_df.astype(str),
                use_container_width=True,
                hide_index=True,
                height=min(n * _ROW_PX + _HDR_PX, _BEFORE_VISIBLE * _ROW_PX + _HDR_PX),
            )
        else:
            st.caption("（データなし）")

    # 親（集計）+ 子をcolumnsに表示する
    all_display = dlt.source_display_order  # [parent_id, child1_id, ...]
    n_disp = min(len(all_display), 4)
    cols = st.columns(n_disp, gap="small")
    for i, tid in enumerate(all_display[:n_disp]):
        with cols[i]:
            if tid == dlt.parent_table_id:
                _table_card_mini(tid, f"集計テーブル：{dlt.parent_title}", "#2a607a")
            else:
                _child_t = tables_dict.get(tid)
                _child_title = (_child_t.title if _child_t else None) or tid
                _table_card_mini(tid, f"集計元テーブル：{_child_title}", "#4a5a3a")
    if len(all_display) > n_disp:
        with st.expander(
            f"他 {len(all_display) - n_disp} テーブルを見る", expanded=False
        ):
            for tid in all_display[n_disp:]:
                if tid == dlt.parent_table_id:
                    _table_card_mini(
                        tid, f"集計テーブル：{dlt.parent_title}", "#2a607a"
                    )
                else:
                    _child_t = tables_dict.get(tid)
                    _child_title = (_child_t.title if _child_t else None) or tid
                    _table_card_mini(tid, f"集計元テーブル：{_child_title}", "#4a5a3a")

    # ── 差分計算セパレーター ─────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0;margin:18px 0 14px;">'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,transparent,rgba(229,168,60,.5));"></div>'
        '<div style="border:1.5px solid rgba(229,168,60,.7);border-radius:24px;'
        "padding:6px 22px;margin:0 16px;font-size:0.95rem;font-weight:800;"
        "color:#f5d06a;letter-spacing:.06em;"
        "background:linear-gradient(135deg,rgba(229,168,60,.12),rgba(200,140,40,.08));"
        'white-space:nowrap;">↓&nbsp;&nbsp;差分計算</div>'
        '<div style="flex:1;height:1px;background:linear-gradient(to left,transparent,rgba(229,168,60,.5));"></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#1a1508;border-left:3px solid rgba(229,168,60,.6);'
        f"border-radius:0 6px 6px 0;padding:6px 14px;font-size:0.82rem;"
        f'color:#c8a030;font-family:monospace;margin-bottom:14px;">'
        f"{_html.escape(dlt.derivation_formula)}</div>",
        unsafe_allow_html=True,
    )

    # ── 推定後 ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:6px 0 14px;">'
        '<div style="width:5px;height:22px;background:#e5a83c;border-radius:3px;flex-shrink:0;"></div>'
        '<span style="font-size:1.05rem;font-weight:800;color:#c8d4e8;letter-spacing:.04em;">推定後（潜在テーブル）</span>'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,rgba(229,168,60,.4),transparent);"></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if dlt.df is not None and not dlt.df.empty:
        n = len(dlt.df)
        st.dataframe(
            dlt.df.astype(str),
            use_container_width=True,
            hide_index=True,
            height=min(n * _ROW_PX + _HDR_PX, _AFTER_VISIBLE * _ROW_PX + _HDR_PX),
        )
        st.caption(
            f"📐 {len(dlt.df)} 行 × {len(dlt.df.columns)} 列  "
            f"（{dlt.parent_title} から {', '.join(dlt.detected_child_ids)} を差し引いた推定値）"
        )
    else:
        st.caption("（推定データなし）")


def _granularity_badge(info: dict) -> str:
    if info["is_integrated"]:
        return "<span class='step-badge badge-integrated'>🔀 統合</span>"
    g = info.get("granularity", "unknown")
    if info.get("is_master"):
        return "<span class='step-badge badge-master'>📚 マスタ</span>"
    if info.get("is_minimum"):
        return "<span class='step-badge badge-detail'>⭐ 最小粒度</span>"
    if g == "summary":
        return "<span class='step-badge badge-summary'>📈 集計</span>"
    if g == "detail":
        return "<span class='step-badge badge-detail'>🔍 詳細</span>"
    return "<span class='step-badge badge-ref'>📄 その他</span>"


def _table_card(tid: str, info: dict, ir=None, tables_dict=None):
    df: pd.DataFrame = info["df"]
    is_sel = tid in st.session_state.selected_ids
    badge = _granularity_badge(info)

    def _sel_button():
        if is_sel:
            if st.button(
                "✅ 選択中", key=f"sel_{tid}", use_container_width=True, type="primary"
            ):
                st.session_state.selected_ids.discard(tid)
                st.rerun()
        else:
            if st.button("＋ 選択", key=f"sel_{tid}", use_container_width=True):
                st.session_state.selected_ids.add(tid)
                st.rerun()

    with st.container(border=True):
        st.markdown(
            f"**{info['display_name']}** &nbsp; `{tid}` {badge}",
            unsafe_allow_html=True,
        )

        if info["is_integrated"] and ir is not None and tables_dict is not None:
            # 統合前/統合後プレビューの上に説明ブロックを表示し、
            # テーブル比較を見る前に統合内容をユーザーが読めるようにする。
            st.markdown(f"_{info['description']}_")
            if info.get("reasoning"):
                st.caption(f"💡 {info['reasoning']}")

            # 統合前/統合後ビュー — ソース一覧とサイズは内部でレンダリングする
            _render_integration_before_after(
                ir,
                tables_dict,
                compact=True,
                full_df_size=(len(df), len(df.columns)),
                source_ids=info.get("source_ids") or list(ir.table_ids),
            )

            st.divider()
            col_gap, col_btn = st.columns([4, 1])
            with col_btn:
                _sel_button()
        else:
            # 非統合テーブルの標準ビュー
            _splitter_marker(f"s5-{tid}")
            col_prev, col_info = st.columns([1, 1])
            with col_prev:
                _new_cols = info.get("new_col_names") or []
                _row_px, _hdr_px, _max_visible = 35, 38, 10
                st.dataframe(
                    _styled_df(df, _new_cols) if _new_cols else df.astype(str),
                    use_container_width=True,
                    hide_index=True,
                    height=min(len(df) * _row_px + _hdr_px, _max_visible * _row_px + _hdr_px),
                )
            with col_info:
                st.markdown(f"_{info['description']}_")
                st.caption(f"📊 {len(df)} 行 × {len(df.columns)} 列")
                st.caption(f"💡 {info['reasoning']}")
                if info.get("source_ids") and len(info["source_ids"]) > 1:
                    st.caption(f"🔗 統合元: {', '.join(info['source_ids'])}")
                st.markdown("<br>", unsafe_allow_html=True)
                _sel_button()


def step5():
    st.header("📋 ステップ 6 : テーブル選択")

    # Step 5 が読み込まれるたびに localStorage のsplitter位置をリセットし、
    # 前回の古いドラッグ位置で右列が折りたたまれないようにする。
    components.html(
        """<script>
        (function(){
            Object.keys(localStorage)
                .filter(function(k){ return k.startsWith('split-s5-'); })
                .forEach(function(k){ localStorage.removeItem(k); });
        })();
        </script>""",
        height=0,
    )

    final: Dict[str, dict] = st.session_state.final_tables

    if not final:
        st.warning("表示できるテーブルがありません")
        st.button("← 戻る", on_click=_go_to, args=(5,))
        return

    # このフィールドが追加される前に保存された古い .tep ファイルから読み込まれたエントリに
    # new_col_names を補完する。プロジェクト復元後も列ハイライトが機能するよう
    # ai_analysis から派生させる。
    _analysis = st.session_state.get("ai_analysis")
    if _analysis:
        _ir_map = {
            f"integrated_{ir.recommendation_id}": ir
            for ir in _analysis.integration_recommendations
        }
        for k, info in final.items():
            if (
                info.get("is_integrated")
                and not info.get("new_col_names")
                and k in _ir_map
            ):
                ir = _ir_map[k]
                info["new_col_names"] = getattr(ir, "new_column_names", []) or [
                    ir.new_column_name
                ]

    # 自動処理はStep 5で終了する
    if st.session_state.auto_processing:
        if st.session_state.run_mode == "fullauto":
            # フルオート: 推奨テーブルを自動選択してエクスポートへ進む
            recommended = {
                tid for tid, info in final.items() if info.get("recommended", False)
            }
            st.session_state.selected_ids = (
                recommended if recommended else set(final.keys())
            )
            st.session_state.auto_processing = False
            st.session_state.auto_completed = True
            st.session_state.step = 7
            st.rerun()
        else:
            # セミオート: 前進パスはここで終了 — ユーザーが手動でテーブルを選択する
            st.session_state.auto_processing = False
            st.session_state.auto_completed = True

    st.info(
        "分析対象とするテーブルを選択してください。"
        "推奨テーブルは初期選択済みです（個別に変更できます）。"
    )

    st.markdown(
        f"**選択中: {len(st.session_state.selected_ids)} / {len(final)} テーブル**"
    )

    # 一括操作ボタン
    c1, c2, _ = st.columns([1, 1, 5])
    with c1:
        if st.button("✅ 全選択", use_container_width=True):
            st.session_state.selected_ids = set(final.keys())
            st.rerun()
    with c2:
        if st.button("❌ 全解除", use_container_width=True):
            st.session_state.selected_ids = set()
            st.rerun()

    st.divider()

    # 統合テーブルの前/後表示用のルックアップを構築する
    _s5_analysis = st.session_state.get("ai_analysis")
    _s5_ir_by_rec: dict = {}
    if _s5_analysis:
        _s5_ir_by_rec = {
            ir.recommendation_id: ir for ir in _s5_analysis.integration_recommendations
        }
    _s5_tbls = {t.table_id: t for t in st.session_state.get("detected_tables", [])}

    def _get_ir_for(k: str):
        if k.startswith("integrated_"):
            return _s5_ir_by_rec.get(k[len("integrated_") :])
        return None

    # --- 統合テーブル（列シグネチャでグループ化） ---
    integrated = {k: v for k, v in final.items() if v["is_integrated"]}
    if integrated:
        st.markdown("### 🔀 統合テーブル")

        def _integrated_col_sig(info: dict) -> frozenset:
            df = info.get("df")
            if df is not None:
                return frozenset(df.columns)
            return frozenset()

        # 列シグネチャでグループ化: 代表を先頭に、類似はまとめて折りたたむ
        int_groups: list = []
        sig_to_int_group: dict = {}
        for tid, info in integrated.items():
            sig = _integrated_col_sig(info)
            if sig not in sig_to_int_group:
                group = [(tid, info)]
                int_groups.append(group)
                sig_to_int_group[sig] = group
            else:
                sig_to_int_group[sig].append((tid, info))

        for group in int_groups:
            rep_tid, rep_info = group[0]
            similar_int = group[1:]

            _table_card(rep_tid, rep_info)

            if similar_int:
                with st.expander(
                    f"同様の統合テーブル 他 {len(similar_int)} 件",
                    expanded=False,
                ):
                    for tid, info in similar_int:
                        _table_card(tid, info)

    # --- 最小粒度テーブル ---
    min_tables = {
        k: v for k, v in final.items() if not v["is_integrated"] and v.get("is_minimum")
    }
    if min_tables:
        st.markdown("### ⭐ 最小粒度データ")
        for tid, info in min_tables.items():
            _table_card(tid, info)

    # --- マスタテーブル ---
    master_tables = {
        k: v
        for k, v in final.items()
        if not v["is_integrated"] and not v.get("is_minimum") and v.get("is_master")
    }
    if master_tables:
        st.markdown("### 📚 マスタテーブル")
        for tid, info in master_tables.items():
            _table_card(tid, info)

    # --- その他の推奨テーブル ---
    shown = set(integrated) | set(min_tables) | set(master_tables)
    other_rec = {
        k: v
        for k, v in final.items()
        if k not in shown and v.get("recommended") and not v["is_integrated"]
    }
    if other_rec:
        st.markdown("### 📊 その他の推奨テーブル")
        for tid, info in other_rec.items():
            _table_card(tid, info)

    # --- 非推奨テーブル（折りたたみ） ---
    shown |= set(other_rec)
    non_rec = {k: v for k, v in final.items() if k not in shown}
    if non_rec:
        with st.expander(
            f"📄 分析対象 非推奨テーブル（{len(non_rec)} 件）— 任意で選択可能"
        ):
            for tid, info in non_rec.items():
                _table_card(tid, info)

    _inject_splitter_js()

    st.divider()
    c1, c2 = st.columns([1, 4])
    with c1:
        st.button("← 戻る", on_click=_go_to, args=(5,))
    with c2:
        n = len(st.session_state.selected_ids)
        if n == 0:
            st.warning("テーブルを 1 件以上選択してください")
        else:
            st.button(
                f"📥 選択した {n} テーブルをエクスポート →",
                type="primary",
                use_container_width=True,
                on_click=_go_to,
                args=(7,),
            )


# ---------------------------------------------------------------------------
# Step 6 — エクスポート
# ---------------------------------------------------------------------------


def step6():
    st.header("📥 ステップ 7 : エクスポート")

    final: Dict[str, dict] = st.session_state.final_tables
    selected = {
        tid: info for tid, info in final.items() if tid in st.session_state.selected_ids
    }

    if not selected:
        st.warning("エクスポート対象のテーブルが選択されていません")
        st.button("← テーブル選択に戻る", on_click=_go_to, args=(6,))
        return

    st.success(f"✅ **{len(selected)} テーブル** のエクスポート準備が完了しました")

    zip_buf = io.BytesIO()
    export_files: Dict[str, bytes] = {}

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for tid, info in selected.items():
            df: pd.DataFrame = info["df"]
            safe_name = (
                info["display_name"]
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )
            fname = f"{safe_name}.csv"

            csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            export_files[fname] = csv_bytes
            zf.writestr(fname, csv_bytes)

    zip_buf.seek(0)

    # 一括ダウンロード
    stem = Path(st.session_state.filename).stem
    st.download_button(
        "📦 全テーブルを ZIP でまとめてダウンロード",
        data=zip_buf.getvalue(),
        file_name=f"{stem}_抽出テーブル.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.divider()
    st.markdown("### 個別ダウンロード")

    for tid, info in selected.items():
        df: pd.DataFrame = info["df"]
        safe_name = (
            info["display_name"].replace("/", "_").replace("\\", "_").replace(" ", "_")
        )
        fname = export_files.get(f"{safe_name}.csv", b"")

        with st.container(border=True):
            c_info, c_dl = st.columns([4, 1])
            with c_info:
                badge = _granularity_badge(info)
                st.markdown(
                    f"**{info['display_name']}** {badge}",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"📊 {len(df)} 行 × {len(df.columns)} 列  →  `{safe_name}.csv`"
                )
                with st.expander("プレビュー"):
                    st.dataframe(
                        df.astype(str),
                        use_container_width=True,
                        hide_index=True,
                        height=210,
                    )
            with c_dl:
                st.markdown("<br>", unsafe_allow_html=True)
                csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode(
                    "utf-8-sig"
                )
                st.download_button(
                    "⬇️ CSV",
                    data=csv_bytes,
                    file_name=f"{safe_name}.csv",
                    mime="text/csv",
                    key=f"dl_{tid}",
                    use_container_width=True,
                )

    st.divider()
    if st.button("🔄 新しいファイルを分析する", use_container_width=True):
        _reset()


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------


_SCROLL_TO_TOP_JS = (
    "<script>"
    "(function(){"
    "var fn=function(){"
    "var d=window.parent.document;"
    "d.documentElement.scrollTop=0;"
    "d.body.scrollTop=0;"
    "['[data-testid=\"stMain\"]','[data-testid=\"stAppViewContainer\"]','.main']"
    ".forEach(function(s){var el=d.querySelector(s);if(el)el.scrollTop=0;});"
    "};"
    "fn();"
    "[80,250,600].forEach(function(t){setTimeout(fn,t);});"
    "})();"
    "</script>"
)


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
