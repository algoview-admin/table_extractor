import io
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.ai_analyzer import analyze_tables
from src.excel_parser import parse_csv, parse_excel
from src.models import AIAnalysisResult, DetectedTable

load_dotenv()

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Excel テーブル抽出 AI エージェント",
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

    /* ── Hide Streamlit default header / menu / footer ── */
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer    { visibility: hidden !important; }

    /* ── Remove default top padding ── */
    .block-container { padding-top: 0 !important; }

    /* ── Custom progress bar ── */
    .app-progress-wrap  { margin-top: 1rem; }
    .app-progress-track {
        background: rgba(127, 255, 212, 0.15);
        border-radius: 999px;
        height: 5px;
        overflow: hidden;
    }
    .app-progress-fill {
        height: 100%;
        background: #7FFFD4;
        border-radius: 999px;
        width: 0%;
        transition: width 1.0s cubic-bezier(0.4, 0, 0.2, 1);
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
    }
    /* Button styles also apply inside the portal clone */
    #_appFixedHdr div[data-testid="stHorizontalBlock"] button {
        padding: 4px 6px !important;
        border-radius: 6px !important;
        min-height: 2rem !important;
    }
    #_appFixedHdr button[data-testid="stBaseButton-primary"] {
        background-color: #7FFFD4 !important;
        color: #0e1117 !important;
        border-color: #7FFFD4 !important;
    }
    #_appFixedHdr button[data-testid="stBaseButton-secondary"] {
        border-color: #7FFFD4 !important;
        color: #7FFFD4 !important;
    }

    /* ── Step nav buttons: compact tab-style ── */
    div[data-testid="stHorizontalBlock"] button {
        padding: 4px 6px !important;
        border-radius: 6px !important;
        min-height: 2rem !important;
    }

    /* ── Primary buttons (次へ系): aquamarine fill ── */
    button[data-testid="stBaseButton-primary"] {
        background-color: #7FFFD4 !important;
        color: #0e1117 !important;
        border-color: #7FFFD4 !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #5ee8be !important;
        border-color: #5ee8be !important;
    }

    /* ── Secondary buttons (戻る系): aquamarine outline ── */
    button[data-testid="stBaseButton-secondary"] {
        border-color: #7FFFD4 !important;
        color: #7FFFD4 !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: rgba(127, 255, 212, 0.1) !important;
        border-color: #7FFFD4 !important;
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
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

STEP_LABELS = [
    "ファイル選択",
    "テーブル検出",
    "テーブル関係分析",
    "新規テーブル確認",
    "テーブル選択",
    "エクスポート",
]


def _init():
    defaults = {
        "step": 1,
        "file_content": None,
        "filename": None,
        "file_ext": None,
        "detected_tables": [],
        "sheet_names": [],
        "ai_analysis": None,
        "integration_decisions": {},  # rec_id -> bool
        "master_decisions": {},  # dim_master_{rec_id} -> bool
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


# ---------------------------------------------------------------------------
# Resizable left/right splitter
# ---------------------------------------------------------------------------


def _splitter_marker(split_id: str) -> None:
    """Invisible marker that JS uses to locate the immediately following st.columns()."""
    st.markdown(
        f'<div class="split-init-marker" data-split-id="{split_id}"></div>',
        unsafe_allow_html=True,
    )


def _inject_splitter_js() -> None:
    """Inject drag-to-resize behaviour for every split-init-marker on the page.

    Uses height=42 as a unique tag so the CSS rule collapses the iframe wrapper
    to 0 px while keeping the script alive and executing.
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
                hBlock.style.alignItems = 'stretch';
                L.style.flex    = '0 0 ' + saved + '%';
                L.style.minWidth = '0';
                L.style.overflow = 'hidden';

                R.style.flex     = '1 1 auto';
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

                /* ── 3. Forward clicks to real Streamlit buttons in srcHdr ── */
                portal.onclick = function (e) {
                    var btn = e.target && e.target.closest('button');
                    if (!btn) return;
                    var txt = btn.textContent.trim();
                    var realBtns = Array.from(srcHdr.querySelectorAll('button'));
                    for (var i = 0; i < realBtns.length; i++) {
                        if (realBtns[i].textContent.trim() === txt) {
                            realBtns[i].click();
                            break;
                        }
                    }
                    e.preventDefault();
                };

                /* ── 4. Collapse srcHdr (keeps it in DOM for Streamlit state) ── */
                srcHdr.style.setProperty('overflow',   'hidden',  'important');
                srcHdr.style.setProperty('max-height', '0',       'important');
                srcHdr.style.setProperty('padding',    '0',       'important');
                srcHdr.style.setProperty('margin',     '0',       'important');
                srcHdr.style.setProperty('visibility', 'hidden',  'important');

                /* ── 5. Push rootBlock content below the portal header ── */
                var ph = portal.getBoundingClientRect().height;
                if (ph > 0) rootBlock.style.setProperty('padding-top', ph + 'px', 'important');
            }

            /* ── Animate custom progress bars to their data-pct target ──
               init() fires 5 times (0/80/250/600/1400 ms), so we guard with
               win._progressAnimPct: only animate when the pct value changes. */
            function animateProgress() {
                var wrap = pdoc.querySelector('.app-progress-wrap');
                if (!wrap) return;
                var pct = parseFloat(wrap.getAttribute('data-pct')) || 0;
                if (win._progressAnimPct === pct) return; /* already done this step */
                win._progressAnimPct = pct;

                pdoc.querySelectorAll('.app-progress-wrap').forEach(function (w) {
                    var fill = w.querySelector('.app-progress-fill');
                    if (!fill) return;
                    fill.style.transition = 'none';
                    fill.style.width = '0%';
                    void fill.offsetWidth;   /* force reflow so transition fires */
                    fill.style.transition = '';
                    fill.style.width = pct + '%';
                });
            }

            /* ── Find markers and wire up ── */
            function init() {
                buildFixedHeader();
                animateProgress();
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
        })();
        </script>
        """,
        height=42,
    )


# ---------------------------------------------------------------------------
# Header & step indicator
# ---------------------------------------------------------------------------


def _can_navigate_to(target: int) -> bool:
    """Return True if the target step has its prerequisite data available."""
    s = st.session_state
    if target == 1:
        return True
    if target == 2:
        return bool(s.get("file_content"))
    if target == 3:
        return bool(s.get("detected_tables"))
    if target == 4:
        return s.get("ai_analysis") is not None
    if target == 5:
        return bool(s.get("final_tables"))
    if target == 6:
        return bool(s.get("selected_ids"))
    return False


def _render_header():
    # Sentinel: JS uses this to locate the header's stVerticalBlock reliably
    st.markdown(
        '<span class="app-hdr-sentinel" style="display:none"></span>',
        unsafe_allow_html=True,
    )
    st.title("📊 Table Extractor AI")
    st.caption("Excel / CSV ファイルから分析対象とするテーブルを抽出します。")

    current = st.session_state.step
    cols = st.columns(len(STEP_LABELS))

    for i, (col, label) in enumerate(zip(cols, STEP_LABELS), 1):
        with col:
            is_current = i == current
            is_done = i < current
            accessible = _can_navigate_to(i)

            if is_current:
                # Current step — green border highlight (HTML div, not a button)
                st.markdown(
                    f'<div style="'
                    f"background:#7FFFD4;"
                    f"border:2px solid #7FFFD4;"
                    f"border-radius:6px;"
                    f"text-align:center;"
                    f"padding:4px 6px;"
                    f"font-weight:600;"
                    f"color:#0e1117;"
                    f"min-height:2rem;"
                    f"display:flex;"
                    f"align-items:center;"
                    f"justify-content:center;"
                    f"box-sizing:border-box;"
                    f'">▶ {label}</div>',
                    unsafe_allow_html=True,
                )
            elif is_done:
                # Completed step — clickable
                if col.button(f"✅ {label}", key=f"nav_{i}", use_container_width=True):
                    st.session_state.step = i
                    st.rerun()
            elif accessible:
                # Future step with data available — clickable
                if col.button(f"○ {label}", key=f"nav_{i}", use_container_width=True):
                    st.session_state.step = i
                    st.rerun()
            else:
                # Future step, data not yet available — disabled
                st.button(
                    f"○ {label}",
                    key=f"nav_{i}",
                    use_container_width=True,
                    disabled=True,
                )

    pct = (current - 1) / (len(STEP_LABELS) - 1) * 100
    st.markdown(
        f'<div class="app-progress-wrap" data-pct="{pct:.4f}">'
        f'<div class="app-progress-track">'
        f'<div class="app-progress-fill"></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )
    st.divider()


# ---------------------------------------------------------------------------
# Step 1 – File upload
# ---------------------------------------------------------------------------


def step1():
    st.header("📂 ステップ 1 : ファイルを選択")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.error(
            "⚠️ **OPENAI_API_KEY** が未設定です。"
            "プロジェクトルートに `.env` ファイルを作成し、キーを設定してください。"
        )
        st.code("OPENAI_API_KEY=sk-...", language="bash")
        return

    st.success("✅ API キー確認済み")

    uploaded = st.file_uploader(
        "Excel または CSV ファイルを選択してください",
        type=["xlsx", "xlsm", "xls", "csv"],
        help="複数シート・複数テーブルを含む Excel ファイルに対応しています",
    )

    if uploaded:
        content = uploaded.getvalue()
        size_kb = len(content) / 1024
        ext = Path(uploaded.name).suffix.lower()

        c1, c2, c3 = st.columns(3)
        c1.metric("ファイル名", uploaded.name)
        c2.metric("サイズ", f"{size_kb:.1f} KB")
        c3.metric("形式", ext.upper())

        if st.button("🔍 テーブル検出を開始", type="primary", use_container_width=True):
            st.session_state.file_content = content
            st.session_state.filename = uploaded.name
            st.session_state.file_ext = ext
            st.session_state.detected_tables = []  # force re-parse
            st.session_state.ai_analysis = None
            st.session_state.final_tables = {}
            st.session_state.selected_ids = set()
            st.session_state.step = 2
            st.rerun()


# ---------------------------------------------------------------------------
# Step 2 – Table detection
# ---------------------------------------------------------------------------


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
            except Exception as e:
                st.error(f"❌ 解析エラー: {e}")
                return

    tables: List[DetectedTable] = st.session_state.detected_tables
    sheets: List[str] = st.session_state.sheet_names

    st.success(
        f"✅ **{len(sheets)} シート** から **{len(tables)} テーブル** を検出しました"
    )

    # Group by sheet
    by_sheet: Dict[str, List[DetectedTable]] = {}
    for t in tables:
        by_sheet.setdefault(t.sheet_name, []).append(t)

    for sheet in sheets:
        sheet_tables = by_sheet.get(sheet, [])
        label = f"📋  {sheet}  （{len(sheet_tables)} テーブル）"
        with st.expander(label, expanded=len(sheet_tables) > 0):
            if not sheet_tables:
                st.info("このシートにはテーブルが検出されませんでした")
                continue
            for t in sheet_tables:
                title_str = f"  🏷️ `{t.title}`" if t.title else ""
                st.markdown(
                    f"**`{t.table_id}`**{title_str}  —  {t.row_count} 行 × {t.col_count} 列"
                    f"  （行 {t.start_row}〜{t.end_row}, 列 {t.start_col}〜{t.end_col}）"
                )
                if t.df is not None and not t.df.empty:
                    st.dataframe(
                        t.df.astype(str),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.divider()

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("← 戻る"):
            st.session_state.step = 1
            st.rerun()
    with c2:
        if not tables:
            st.warning("テーブルが検出されませんでした。別のファイルをお試しください。")
        else:
            if st.button(
                "次へ：テーブル関係分析を開始 →",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.step = 3
                st.rerun()


# ---------------------------------------------------------------------------
# Step 3 – Table relationship analysis
# ---------------------------------------------------------------------------


def step3():
    st.header("🧠 ステップ 3 : テーブル関係分析")

    if st.session_state.ai_analysis is None:
        with st.spinner("テーブルを分析中です..."):
            try:
                result = analyze_tables(
                    st.session_state.detected_tables,
                    st.session_state.sheet_names,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                )
                st.session_state.ai_analysis = result
            except Exception as e:
                st.error(f"❌ テーブル関係分析エラー: {e}")
                return

    analysis: AIAnalysisResult = st.session_state.ai_analysis

    # Summary banner
    st.info(f"📊 **分析サマリー**: {analysis.summary}")

    # Sheet classifications
    with st.expander("📋 シート分類", expanded=True):
        for sc in analysis.sheet_classifications:
            icon = "📊" if sc.is_data_sheet else "📝"
            badge = "データシート" if sc.is_data_sheet else "説明 / 補足シート"
            st.markdown(f"{icon} **{sc.sheet_name}** → {badge}")
            if sc.description:
                st.caption(sc.description)

    # Hierarchy overview
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

    # Minimum granularity
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

    # Master tables
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

    # Integration recommendations preview
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

    # Master generation recommendations preview
    master_irs = [
        ir
        for ir in analysis.integration_recommendations
        if ir.parent_table_id and ir.parent_label_column
    ]
    if master_irs:
        with st.expander(
            f"🗂️ マスタ自動生成推奨（{len(master_irs)} 件） — 次のステップで確認します",
            expanded=False,
        ):
            for ir in master_irs:
                st.markdown(
                    f"**{ir.new_column_name} × {ir.parent_label_column} マスタ**"
                    f"  ←  `{ir.parent_table_id}` と統合テーブル `{ir.recommendation_id}` から生成"
                )
                st.caption(
                    f"統合テーブル（{ir.group_name}）の各行が上位区分（{ir.parent_label_column}）に"
                    "対応するかを示すマスタテーブルを自動生成します。"
                )

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("← 戻る"):
            st.session_state.step = 2
            st.rerun()
    with c2:
        if st.button(
            "次へ：新規テーブル確認 →", type="primary", use_container_width=True
        ):
            st.session_state.step = 4
            st.rerun()


# ---------------------------------------------------------------------------
# Step 4 – Integration review
# ---------------------------------------------------------------------------


def _derive_master_spec(ir, tables_dict):
    """Derive a correct hierarchical master from an integration recommendation.

    The integrated children differ along exactly one axis. We detect that axis
    from the table metadata and map each child to its parent ALONG THE SAME AXIS,
    so a service is never mapped to a branch (cross-axis) and vice versa.

      - sheet axis : children share a section title but live on different sheets
                     (e.g. same service across branches → rolls up to a division)
      - title axis : children share a sheet but have different section titles
                     (e.g. detail services in one branch → roll up to their total)

    Returns dict(child_col, parent_col, mapping{child_label: parent_label},
    parent_id) or None when the children + parent do not form a clean single-axis
    hierarchy (in which case no master should be generated).
    """
    parent = tables_dict.get(ir.parent_table_id)
    members = [tables_dict.get(t) for t in ir.table_ids]
    members = [m for m in members if m is not None]
    if parent is None or len(members) < 2:
        return None

    child_sheets = {m.sheet_name for m in members}
    child_titles = {(m.title or "") for m in members}

    # --- sheet axis: same title, distinct sheets; parent is a different sheet ---
    if len(child_titles) == 1 and len(child_sheets) == len(members):
        shared_title = next(iter(child_titles))
        if parent.sheet_name in child_sheets:
            return None  # parent is a sibling, not an aggregate
        if (parent.title or "") != shared_title:
            return None  # parent not aligned on the shared title → different axis
        mapping = {m.sheet_name: parent.sheet_name for m in members}
        return {
            "child_col": ir.new_column_name or "区分",
            "parent_col": ir.parent_label_column or "上位区分",
            "mapping": mapping,
            "parent_id": ir.parent_table_id,
        }

    # --- title axis: same sheet, distinct titles; parent is the aggregate title ---
    if len(child_sheets) == 1 and len(child_titles) == len(members):
        shared_sheet = next(iter(child_sheets))
        if parent.sheet_name != shared_sheet:
            return None  # parent on a different sheet → cross-axis, reject
        parent_label = parent.title or parent.sheet_name
        if parent_label in child_titles:
            return None  # parent is a sibling, not an aggregate
        child_col = ir.new_column_name or "区分"
        mapping = {(m.title or m.sheet_name): parent_label for m in members}
        return {
            "child_col": child_col,
            "parent_col": f"上位{child_col}",
            "mapping": mapping,
            "parent_id": ir.parent_table_id,
        }

    return None  # ambiguous / cross-axis → no master


def _master_signature(spec):
    """Identity of the master a spec produces, independent of which integration
    it came from. Masters with the same child→parent label map are duplicates."""
    return (
        spec["child_col"],
        spec["parent_col"],
        frozenset(spec["mapping"].items()),
    )


def _dedup_master_irs(master_irs, tables_dict):
    """Keep only recommendations that yield a valid, not-yet-seen master."""
    seen = set()
    result = []
    for ir in master_irs:
        spec = _derive_master_spec(ir, tables_dict)
        if spec is None:
            continue
        sig = _master_signature(spec)
        if sig in seen:
            continue
        seen.add(sig)
        result.append(ir)
    return result


def step4():
    st.header("✅ ステップ 4 : 新規テーブル確認")
    analysis: AIAnalysisResult = st.session_state.ai_analysis
    tables_dict = {t.table_id: t for t in st.session_state.detected_tables}

    has_integrations = bool(analysis.integration_recommendations)
    master_irs = _dedup_master_irs(
        [
            ir
            for ir in analysis.integration_recommendations
            if ir.parent_table_id and ir.parent_label_column
        ],
        tables_dict,
    )
    has_masters = bool(master_irs)

    if not has_integrations and not has_masters:
        st.info("新規テーブルの生成推奨はありません。このステップはスキップします。")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("← 戻る"):
                st.session_state.step = 3
                st.rerun()
        with c2:
            if st.button(
                "次へ：テーブル選択 →", type="primary", use_container_width=True
            ):
                _build_final_tables()
                st.session_state.step = 5
                st.rerun()
        return

    # ── Section 1: Integration recommendations ──────────────────────────────
    if has_integrations:
        st.subheader("🔀 統合テーブル")
        st.caption(
            "AI が以下のテーブル統合を推奨しています。各統合について実施するかどうかをお選びください。"
        )

        for ir in analysis.integration_recommendations:
            if ir.recommendation_id not in st.session_state.integration_decisions:
                st.session_state.integration_decisions[ir.recommendation_id] = True

        for ir in analysis.integration_recommendations:
            with st.container(border=True):
                st.markdown(f"#### {ir.group_name}")
                st.markdown(f"_{ir.description}_")

                _splitter_marker(f"s4-ir-{ir.recommendation_id}")
                c_prev, c_info = st.columns([1, 1])
                with c_prev:
                    st.caption("統合後プレビュー（先頭 2 テーブル × 2 行）")
                    preview_frames = []
                    for tid in ir.table_ids[:2]:
                        t = tables_dict.get(tid)
                        if t and t.df is not None:
                            row = t.df.head(2).copy()
                            row.insert(
                                0, ir.new_column_name, ir.new_column_values.get(tid, "")
                            )
                            preview_frames.append(row)
                    if preview_frames:
                        try:
                            combined_prev = pd.concat(preview_frames, ignore_index=True)
                            st.dataframe(
                                combined_prev.astype(str),
                                use_container_width=True,
                                hide_index=True,
                            )
                        except Exception:
                            st.caption("（プレビュー生成不可）")
                    if len(ir.table_ids) > 2:
                        st.caption(f"他 {len(ir.table_ids) - 2} テーブルも統合されます")

                with c_info:
                    st.markdown(f"**対象テーブル**: {', '.join(ir.table_ids)}")
                    st.markdown(f"**追加列名**: `{ir.new_column_name}`")
                    for tid, val in ir.new_column_values.items():
                        st.markdown(f"  - `{tid}` → **{val}**")
                    st.caption(f"💡 推奨理由: {ir.reasoning}")

                    st.markdown("<br>", unsafe_allow_html=True)
                    decision = st.radio(
                        "この統合を実施しますか？",
                        ["✅ 統合する", "❌ 統合しない"],
                        horizontal=True,
                        key=f"radio_{ir.recommendation_id}",
                        index=(
                            0
                            if st.session_state.integration_decisions.get(
                                ir.recommendation_id, True
                            )
                            else 1
                        ),
                    )
                    st.session_state.integration_decisions[ir.recommendation_id] = (
                        decision == "✅ 統合する"
                    )

    # ── Section 2: Master generation ────────────────────────────────────────
    if has_masters:
        st.divider()
        st.subheader("🗂️ マスタ自動生成")
        st.caption(
            "上位集計テーブルとの階層関係から、下位区分と上位区分の対応マスタを自動生成できます。"
            "統合テーブルに結合することで、どの上位区分に属するかを再現できます。"
        )

        for ir in master_irs:
            spec = _derive_master_spec(ir, tables_dict)
            if spec is None:
                continue
            child_col = spec["child_col"]
            parent_col = spec["parent_col"]
            dm_key = f"dim_master_{ir.recommendation_id}"
            if dm_key not in st.session_state.master_decisions:
                st.session_state.master_decisions[dm_key] = True

            # Build preview DataFrame from the axis-aware child→parent mapping
            master_rows = [
                {child_col: child_label, parent_col: parent_label}
                for child_label, parent_label in spec["mapping"].items()
            ]
            master_preview_df = pd.DataFrame(master_rows)

            with st.container(border=True):
                title = f"{child_col} × {parent_col} マスタ"
                st.markdown(f"#### {title}")
                st.markdown(
                    f"統合テーブル **{ir.group_name}** の `{child_col}` 列と、"
                    f"上位集計テーブル `{spec['parent_id']}` の対応関係を示すマスタテーブル。"
                )

                _splitter_marker(f"s4-dm-{ir.recommendation_id}")
                c_prev, c_info = st.columns([1, 1])
                with c_prev:
                    st.caption("生成されるマスタのプレビュー（全件）")
                    st.dataframe(
                        master_preview_df, use_container_width=True, hide_index=True
                    )

                with c_info:
                    st.markdown(f"**キー列**: `{child_col}`")
                    st.markdown(f"**上位区分列**: `{parent_col}`")
                    st.markdown(f"**上位テーブル**: `{spec['parent_id']}`")
                    st.markdown(f"**行数**: {len(master_rows)} 行")
                    st.caption(
                        f"このマスタと統合テーブルを `{child_col}` で結合すると、"
                        f"各行の `{parent_col}` が参照できます。"
                    )

                    st.markdown("<br>", unsafe_allow_html=True)
                    decision = st.radio(
                        "このマスタを生成しますか？",
                        ["✅ マスタを作成する", "❌ マスタを作成しない"],
                        horizontal=True,
                        key=f"radio_dm_{ir.recommendation_id}",
                        index=(
                            0
                            if st.session_state.master_decisions.get(dm_key, True)
                            else 1
                        ),
                    )
                    st.session_state.master_decisions[dm_key] = (
                        decision == "✅ マスタを作成する"
                    )

    # ── Navigation ───────────────────────────────────────────────────────────
    _inject_splitter_js()
    st.divider()
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("← 戻る"):
            st.session_state.step = 3
            st.rerun()
    with c2:
        if st.button("次へ：テーブル選択 →", type="primary", use_container_width=True):
            _build_final_tables()
            st.session_state.step = 5
            st.rerun()


# ---------------------------------------------------------------------------
# Build final table list (called on leaving step 4)
# ---------------------------------------------------------------------------


def _build_final_tables():
    analysis: AIAnalysisResult = st.session_state.ai_analysis
    tables_dict = {t.table_id: t for t in st.session_state.detected_tables}
    ta_by_id = {ta.table_id: ta for ta in analysis.table_analyses}

    final: Dict[str, dict] = {}
    integrated_ids: Set[str] = set()
    seen_master_sigs: Set = set()  # Dedup masters that map identical child→parent labels

    # Apply approved integrations
    for ir in analysis.integration_recommendations:
        if not st.session_state.integration_decisions.get(ir.recommendation_id, True):
            continue

        frames = []
        for tid in ir.table_ids:
            t = tables_dict.get(tid)
            if t and t.df is not None and not t.df.empty:
                df_copy = t.df.copy()
                df_copy.insert(0, ir.new_column_name, ir.new_column_values.get(tid, ""))
                frames.append(df_copy)
                integrated_ids.add(tid)

        if not frames:
            continue
        try:
            merged_df = pd.concat(frames, ignore_index=True)
        except Exception:
            # Column mismatch – skip integration, revert
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
        }

        # Auto-generate dimension master from the axis-aware hierarchy. Each
        # distinct master (by child→parent label map) is handled once, so sibling
        # integrations that imply the same hierarchy don't produce duplicates.
        dm_key = f"dim_master_{ir.recommendation_id}"
        spec = (
            _derive_master_spec(ir, tables_dict)
            if ir.parent_table_id and ir.parent_label_column
            else None
        )
        if spec is not None:
            master_sig = _master_signature(spec)
            already_handled = master_sig in seen_master_sigs
            seen_master_sigs.add(master_sig)
        else:
            already_handled = True  # not a valid master-producing IR

        if (
            spec is not None
            and not already_handled
            and st.session_state.master_decisions.get(dm_key, True)
        ):
            child_col = spec["child_col"]
            parent_col = spec["parent_col"]
            master_rows = [
                {child_col: child_label, parent_col: parent_label}
                for child_label, parent_label in spec["mapping"].items()
            ]
            if master_rows:
                master_df = pd.DataFrame(master_rows)
                final[dm_key] = {
                    "df": master_df,
                    "display_name": f"{child_col}×{parent_col} マスタ",
                    "description": (
                        f"{child_col}と{parent_col}の対応関係を示すマスタテーブル。"
                        f"統合テーブル「{ir.group_name}」の各行がどの{parent_col}に属するかを再現できる。"
                    ),
                    "reasoning": (
                        f"{spec['parent_id']}（上位集計）と統合テーブルの階層関係から自動生成。"
                        f"統合テーブルに結合することで{parent_col}単位の集計が可能になる。"
                    ),
                    "is_integrated": False,
                    "source_ids": [spec["parent_id"]] + ir.table_ids,
                    "recommended": True,
                    "granularity": "master",
                    "is_minimum": False,
                    "is_master": True,
                }

    # Individual non-integrated tables
    for ta in analysis.table_analyses:
        if ta.table_id in integrated_ids:
            continue
        t = tables_dict.get(ta.table_id)
        if not t or t.df is None or t.df.empty:
            continue
        final[ta.table_id] = {
            "df": t.df,
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

    # Safety net – tables not included in analysis
    analyzed_ids = set(ta_by_id.keys())
    for t in st.session_state.detected_tables:
        if t.table_id in analyzed_ids or t.table_id in integrated_ids:
            continue
        if t.df is None or t.df.empty:
            continue
        final[t.table_id] = {
            "df": t.df,
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
    # Pre-select recommended tables
    st.session_state.selected_ids = {
        tid for tid, info in final.items() if info["recommended"]
    }


# ---------------------------------------------------------------------------
# Step 5 – Table selection
# ---------------------------------------------------------------------------


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


def _table_card(tid: str, info: dict):
    df: pd.DataFrame = info["df"]
    is_sel = tid in st.session_state.selected_ids
    badge = _granularity_badge(info)

    with st.container(border=True):
        _splitter_marker(f"s5-{tid}")
        col_prev, col_info = st.columns([1, 1])

        with col_prev:
            st.dataframe(
                df.astype(str),
                use_container_width=True,
                hide_index=True,
                height=220,
            )

        with col_info:
            st.markdown(
                f"**{info['display_name']}** &nbsp; `{tid}` {badge}",
                unsafe_allow_html=True,
            )
            st.markdown(f"_{info['description']}_")
            st.caption(f"📊 {len(df)} 行 × {len(df.columns)} 列")
            st.caption(f"💡 {info['reasoning']}")
            if info.get("source_ids") and len(info["source_ids"]) > 1:
                st.caption(f"🔗 統合元: {', '.join(info['source_ids'])}")

            st.markdown("<br>", unsafe_allow_html=True)
            if is_sel:
                if st.button(
                    "✅ 選択中",
                    key=f"sel_{tid}",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.selected_ids.discard(tid)
                    st.rerun()
            else:
                if st.button("＋ 選択", key=f"sel_{tid}", use_container_width=True):
                    st.session_state.selected_ids.add(tid)
                    st.rerun()


def step5():
    st.header("📋 ステップ 5 : テーブル選択")

    final: Dict[str, dict] = st.session_state.final_tables

    if not final:
        st.warning("表示できるテーブルがありません")
        if st.button("← 戻る"):
            st.session_state.step = 4
            st.rerun()
        return

    st.info(
        "分析対象とするテーブルを選択してください。"
        "推奨テーブルは初期選択済みです（個別に変更できます）。"
    )

    # Bulk buttons
    c1, c2, c3 = st.columns([1, 1, 5])
    with c1:
        if st.button("✅ 全選択"):
            st.session_state.selected_ids = set(final.keys())
            st.rerun()
    with c2:
        if st.button("❌ 全解除"):
            st.session_state.selected_ids = set()
            st.rerun()
    st.markdown(
        f"**選択中: {len(st.session_state.selected_ids)} / {len(final)} テーブル**"
    )

    st.divider()

    # --- Integrated tables ---
    integrated = {k: v for k, v in final.items() if v["is_integrated"]}
    if integrated:
        st.markdown("### 🔀 統合テーブル")
        for tid, info in integrated.items():
            _table_card(tid, info)

    # --- Minimum granularity ---
    min_tables = {
        k: v for k, v in final.items() if not v["is_integrated"] and v.get("is_minimum")
    }
    if min_tables:
        st.markdown("### ⭐ 最小粒度データ")
        for tid, info in min_tables.items():
            _table_card(tid, info)

    # --- Master tables ---
    master_tables = {
        k: v
        for k, v in final.items()
        if not v["is_integrated"] and not v.get("is_minimum") and v.get("is_master")
    }
    if master_tables:
        st.markdown("### 📚 マスタテーブル")
        for tid, info in master_tables.items():
            _table_card(tid, info)

    # --- Other recommended ---
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

    # --- Non-recommended (collapsed) ---
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
        if st.button("← 戻る"):
            st.session_state.step = 4
            st.rerun()
    with c2:
        n = len(st.session_state.selected_ids)
        if n == 0:
            st.warning("テーブルを 1 件以上選択してください")
        else:
            if st.button(
                f"📥 選択した {n} テーブルをエクスポート →",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.step = 6
                st.rerun()


# ---------------------------------------------------------------------------
# Step 6 – Export
# ---------------------------------------------------------------------------


def step6():
    st.header("📥 ステップ 6 : エクスポート")

    final: Dict[str, dict] = st.session_state.final_tables
    selected = {
        tid: info for tid, info in final.items() if tid in st.session_state.selected_ids
    }

    if not selected:
        st.warning("エクスポート対象のテーブルが選択されていません")
        if st.button("← テーブル選択に戻る"):
            st.session_state.step = 5
            st.rerun()
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

    # Bulk download
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
# Main
# ---------------------------------------------------------------------------


def main():
    _init()

    with st.container():
        _render_header()

    step = st.session_state.step
    if step == 1:
        step1()
    elif step == 2:
        step2()
    elif step == 3:
        step3()
    elif step == 4:
        step4()
    elif step == 5:
        step5()
    elif step == 6:
        step6()
    _inject_splitter_js()


if __name__ == "__main__":
    main()
