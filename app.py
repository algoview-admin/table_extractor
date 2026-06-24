import io
import os
import pickle
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

load_dotenv()

# ---------------------------------------------------------------------------
# Page config & CSS
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

    /* ── Hide Streamlit default header / menu / footer ── */
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer    { visibility: hidden !important; }

    /* ── Remove default top padding (both old and new Streamlit selectors) ── */
    .block-container,
    [data-testid="stMainBlockContainer"] { padding-top: 0 !important; }

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
    "新規テーブル案生成",
    "テーブル選択",
    "エクスポート",
]


def _init():
    defaults = {
        "step": 1,
        "run_mode": "manual",  # "manual" | "semiauto" | "fullauto"
        "auto_processing": False,  # True only during the forward auto-run pass
        "source_mode": None,  # "new_file" | "project" — how data was loaded
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


def _go_to(step: int, stop_auto: bool = True) -> None:
    """Navigation callback — runs before the next render, so the header
    always shows the correct active tab on the very first rerun after a click."""
    if stop_auto:
        st.session_state.auto_processing = False
    st.session_state.step = step


def _build_and_go_step5() -> None:
    _build_final_tables()
    st.session_state.step = 5


# ---------------------------------------------------------------------------
# Project save / load
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
    "final_tables",
    "selected_ids",
    "step",
]


def _serialize_project() -> bytes:
    """Pickle the current session state into a .tep project blob."""
    payload = {
        "__tep_version__": _PROJECT_VERSION,
        "__saved_at__": datetime.now().isoformat(timespec="seconds"),
    }
    for k in _SAVE_KEYS:
        payload[k] = st.session_state.get(k)
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _save_project_to_disk() -> str:
    """Save current session to _PROJECT_DIR. Returns a status message."""
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
    """Restore session state from a .tep blob. Returns a status message."""
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

    # Ensure auto_processing is off and mark source when restoring
    st.session_state["auto_processing"] = False
    st.session_state["source_mode"] = "project"
    return f"✅ プロジェクトを復元しました（保存日時: {payload.get('__saved_at__', '不明')}）"


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
# Header & step indicator
# ---------------------------------------------------------------------------


def _can_navigate_to(target: int) -> bool:
    """Return True if the target step has its prerequisite data available."""
    s = st.session_state
    if target == 1:
        return True
    if target == 2:
        return bool(s.get("file_content")) or bool(s.get("detected_tables"))
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
    title_col, save_col = st.columns([6, 1])
    with title_col:
        st.title("📊 Table Extractor (開発中)")
        st.caption("Excel / CSV ファイルから分析対象とするテーブルを抽出します。")
    with save_col:
        if st.session_state.get("filename"):
            has_data = bool(st.session_state.get("detected_tables"))

            def _on_save():
                st.session_state["_save_result"] = _save_project_to_disk()

            st.markdown("<br>", unsafe_allow_html=True)
            st.button(
                "💾 Save",
                key="hdr_save_btn",
                use_container_width=True,
                disabled=not has_data,
                on_click=_on_save,
                help=f"現在の解析状態を {_PROJECT_DIR} に保存します。",
            )

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
                # Completed step — clickable; stop auto-processing when navigating back
                col.button(
                    f"✅ {label}",
                    key=f"nav_{i}",
                    on_click=_go_to,
                    args=(i,),
                    use_container_width=True,
                )
            elif accessible:
                # Future step with data available — clickable
                col.button(
                    f"○ {label}",
                    key=f"nav_{i}",
                    on_click=_go_to,
                    args=(i,),
                    use_container_width=True,
                )
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

    # Auto-processing banner
    if st.session_state.get("auto_processing"):
        _mode = st.session_state.get("run_mode", "manual")
        _mode_label = "セミオート" if _mode == "semiauto" else "フルオート"
        _step_label = (
            STEP_LABELS[current - 1] if 1 <= current <= len(STEP_LABELS) else ""
        )
        st.info(
            f"⚙️ **{_mode_label} 実行中** — "
            f"ステップ {current} / {len(STEP_LABELS)}「{_step_label}」を処理中...",
            icon=None,
        )

    st.divider()


# ---------------------------------------------------------------------------
# Step 1 – File upload
# ---------------------------------------------------------------------------


def _check_api_config() -> tuple:
    """Validate API credentials from env vars.

    Returns (is_valid: bool, label: str, error_msg: str, hint_code: str).
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

    # Auto-select プロジェクト読込 tab when session was restored from a .tep file.
    # st.tabs always defaults to the first tab on rerun, so we click the button via JS.
    # Using multiple retry delays handles timing variance in Streamlit's React render cycle.
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

    # ── Tab 1: New file ───────────────────────────────────────────────────────
    with tab_new:
        # Show currently loaded file when returning from a later step
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

        # Mode selection
        st.markdown("#### 実行モード")
        mode_labels = {
            "manual": "マニュアル  —  各ステップを手動で確認しながら進む",
            "semiauto": "セミオート  —  テーブル選択画面まで自動実行し、選択のみ手動で行う",
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
                st.session_state.source_mode = "new_file"
                st.session_state.step = 2

            st.button(
                btn_labels[selected_mode],
                type="primary",
                use_container_width=True,
                on_click=_start_file,
                args=(content, uploaded.name, ext, selected_mode),
            )

    # ── Tab 2: Load project (プロジェクト読込) ────────────────────────────────
    with tab_load:
        # Show currently loaded project when returning from a later step
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

    # Auto-advance during the initial forward pass only (before any UI output)
    if st.session_state.auto_processing:
        st.session_state.step = 3
        st.rerun()

    st.success(
        f"✅ **{len(sheets)} シート** から **{len(tables)} テーブル** を検出しました"
    )

    # Group by sheet
    by_sheet: Dict[str, List[DetectedTable]] = {}
    for t in tables:
        by_sheet.setdefault(t.sheet_name, []).append(t)

    # ── Tree view ──────────────────────────────────────────────────────────
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

    # ── Per-sheet expanders (collapsed by default) ─────────────────────────
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
                if t.df is not None and not t.df.empty:
                    st.dataframe(
                        t.df.astype(str),
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
                "次へ：テーブル関係分析を開始 →",
                type="primary",
                use_container_width=True,
                on_click=_go_to,
                args=(3,),
            )


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
                )
                st.session_state.ai_analysis = result
            except Exception as e:
                st.error(f"❌ テーブル関係分析エラー: {e}")
                return

    analysis: AIAnalysisResult = st.session_state.ai_analysis

    # Auto-advance during the initial forward pass only
    if st.session_state.auto_processing:
        st.session_state.step = 4
        st.rerun()

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

    # Master generation recommendations preview (per-axis)
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
        st.button("← 戻る", on_click=_go_to, args=(2,))
    with c2:
        st.button(
            "次へ：新規テーブル案生成 →",
            type="primary",
            use_container_width=True,
            on_click=_go_to,
            args=(4,),
        )


# ---------------------------------------------------------------------------
# Step 4 – Integration review
# ---------------------------------------------------------------------------


def _ir_column_signature(ir, tables_dict) -> frozenset:
    """Column signature used to detect 'similar' integration recommendations.

    Two IRs are considered similar when they share the same discriminator columns
    AND their source tables have the same schema.
    """
    col_names = getattr(ir, "new_column_names", []) or [ir.new_column_name]
    for tid in ir.table_ids:
        t = tables_dict.get(tid)
        if t is not None and t.df is not None:
            return frozenset(col_names + list(t.df.columns))
    return frozenset(col_names)


def _group_irs_by_similarity(irs, tables_dict):
    """Return list-of-lists: each inner list is a group of similar IRs.

    The first element of each group is the representative (shown expanded);
    the rest are collapsed into a dropdown.
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
    """Return 'sheet' or 'title' by checking whether axis values match sheet names or section titles."""
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
    """Return a list of master specs — one per axis that has a parent.

    Each spec is a dict with keys:
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

    # Resolve per-axis parent info; fall back to old single-axis fields for axis 0
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

        # Determine axis type (sheet vs title) by comparing axis values to metadata
        atype = _axis_type(axis_idx, multi_vals, members)

        # Validate: parent must NOT be a sibling (same level) of the children
        if atype == "sheet":
            if parent.sheet_name in child_sheets:
                continue  # parent on same sheet as children → sibling, reject
            parent_label = parent.sheet_name
        else:  # title axis
            child_titles = {(m.title or "") for m in members}
            if (parent.title or parent.sheet_name) in child_titles:
                continue  # parent has same title as a child → sibling, reject
            parent_label = parent.title or parent.sheet_name

        # Build child → parent_label mapping from actual axis values
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
                    parent_label  # duplicates (same branch, diff service) collapse
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
    """Backward-compat shim: return first spec for axis 0 only, or None."""
    specs = _derive_master_specs_for_ir(ir, tables_dict)
    return specs[0] if specs else None


def _master_signature(spec):
    """Identity of the master a spec produces, independent of which integration
    it came from. Masters with the same child→parent label map are duplicates."""
    return (
        spec["child_col"],
        spec["parent_col"],
        frozenset(spec["mapping"].items()),
    )


def _collect_unique_master_specs(irs, tables_dict) -> list:
    """Return deduplicated list of (ir, spec) pairs across all axes of all IRs."""
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
    """Backward-compat shim: return IRs that have at least one valid master spec."""
    seen_ir_ids: set = set()
    result = []
    for ir, _spec in _collect_unique_master_specs(master_irs, tables_dict):
        if ir.recommendation_id not in seen_ir_ids:
            seen_ir_ids.add(ir.recommendation_id)
            result.append(ir)
    return result


def step4():
    st.header("✅ ステップ 4 : 新規テーブル案生成")

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

    # Auto-advance during the initial forward pass only
    if st.session_state.auto_processing:
        with st.spinner("新規テーブル案・マスタ案を自動設定中..."):
            _build_final_tables()
        st.session_state.step = 5
        st.rerun()

    tables_dict = {t.table_id: t for t in st.session_state.detected_tables}

    has_integrations = bool(analysis.integration_recommendations)
    unique_master_specs = _collect_unique_master_specs(
        analysis.integration_recommendations, tables_dict
    )
    has_masters = bool(unique_master_specs)
    # For back-compat (used only in expander count)
    master_irs = list(
        {ir.recommendation_id: ir for ir, _ in unique_master_specs}.values()
    )

    if not has_integrations and not has_masters:
        st.info("新規テーブルの生成推奨はありません。このステップはスキップします。")
        c1, c2 = st.columns([1, 4])
        with c1:
            st.button("← 戻る", on_click=_go_to, args=(3,))
        with c2:
            st.button(
                "次へ：テーブル選択 →",
                type="primary",
                use_container_width=True,
                on_click=_build_and_go_step5,
            )
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

        ir_groups = _group_irs_by_similarity(
            analysis.integration_recommendations, tables_dict
        )

        def _render_ir_card(ir, tables_dict):
            with st.container(border=True):
                st.markdown(f"#### {ir.group_name}")
                st.markdown(f"_{ir.description}_")

                # Full-width before/after preview
                _render_integration_before_after(ir, tables_dict)

                st.divider()

                # Info + decision below the preview
                _splitter_marker(f"s4-ir-{ir.recommendation_id}")
                c_info, c_dec = st.columns([2, 1])
                with c_info:
                    _col_names = getattr(ir, "new_column_names", []) or [
                        ir.new_column_name
                    ]
                    _multi_vals = getattr(ir, "new_column_multi_values", {}) or {}
                    st.markdown(
                        f"**追加列名**: {', '.join(f'`{n}`' for n in _col_names)}"
                    )
                    for tid in ir.table_ids:
                        vals = _multi_vals.get(tid) or [
                            ir.new_column_values.get(tid, "")
                        ]
                        val_str = " / ".join(str(v) for v in vals)
                        st.markdown(f"  - `{tid}` → **{val_str}**")
                    st.caption(f"💡 推奨理由: {ir.reasoning}")

                with c_dec:
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

        for group in ir_groups:
            representative = group[0]
            similar = group[1:]

            _render_ir_card(representative, tables_dict)

            if similar:
                _rep_col_names = getattr(representative, "new_column_names", []) or [
                    representative.new_column_name
                ]
                _axes_label = " × ".join(_rep_col_names)
                with st.expander(
                    f"同様の統合 他 {len(similar)} 件（{_axes_label} 軸）",
                    expanded=False,
                ):
                    for ir in similar:
                        _render_ir_card(ir, tables_dict)

    # ── Section 2: Master generation ────────────────────────────────────────
    if has_masters:
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

    # ── Navigation ───────────────────────────────────────────────────────────
    _inject_splitter_js()
    st.divider()
    c1, c2 = st.columns([1, 4])
    with c1:
        st.button("← 戻る", on_click=_go_to, args=(3,))
    with c2:
        st.button(
            "次へ：テーブル選択 →",
            type="primary",
            use_container_width=True,
            on_click=_build_and_go_step5,
        )


# ---------------------------------------------------------------------------
# Build final table list (called on leaving step 4)
# ---------------------------------------------------------------------------


def _build_final_tables():
    analysis: AIAnalysisResult = st.session_state.ai_analysis
    tables_dict = {t.table_id: t for t in st.session_state.detected_tables}
    ta_by_id = {ta.table_id: ta for ta in analysis.table_analyses}

    final: Dict[str, dict] = {}
    integrated_ids: Set[str] = set()
    seen_master_sigs: Set = (
        set()
    )  # Dedup masters that map identical child→parent labels

    # Apply approved integrations
    for ir in analysis.integration_recommendations:
        if not st.session_state.integration_decisions.get(ir.recommendation_id, True):
            continue

        # Resolve multi-axis column names/values (fall back to single-axis compat fields)
        col_names = getattr(ir, "new_column_names", []) or [ir.new_column_name]
        multi_vals = getattr(ir, "new_column_multi_values", {}) or {}

        frames = []
        for tid in ir.table_ids:
            t = tables_dict.get(tid)
            if t and t.df is not None and not t.df.empty:
                df_copy = t.df.copy()
                vals = multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
                # Insert columns right-to-left so they appear left-to-right in result
                for i in range(len(col_names) - 1, -1, -1):
                    val = vals[i] if i < len(vals) else ""
                    df_copy.insert(0, col_names[i], val)
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
            "new_col_names": col_names,
        }

        # Auto-generate dimension masters for ALL axes that have a parent.
        # Each unique master (by child→parent mapping) is generated only once.
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

# Each new column gets a distinct palette: (cell_bg, cell_fg, hdr_bg, hdr_fg)
# hdr_bg is noticeably darker than cell_bg for visual contrast.
_COL_PALETTES = [
    ("#d0faf0", "#0d4b36", "#2aab87", "#ffffff"),  # teal
    ("#fdebd0", "#7a3a0a", "#d4793a", "#ffffff"),  # orange
    ("#d8e8fd", "#0d2f7f", "#4a7de0", "#ffffff"),  # blue
    ("#f0d5f8", "#5b0f7f", "#9b4fc4", "#ffffff"),  # purple
]

# Per-axis color families for integration previews.
# Each family has 8 DISTINCT colors (not shades): (hdr_bg, hdr_fg, cell_bg, cell_fg)
# Axis 0 = WARM family (red/orange/pink/amber — visually distinct per value)
# Axis 1 = COOL family (blue/teal/purple/cyan — visually distinct per value)
# Axis 2 = NATURE family (green/lime/olive/forest)
# Axis 3 = ACCENT family (deep-orange/indigo/rose/mint)
_AXIS_FAMILIES: list = [
    [  # axis-0: WARM — each value is a clearly different warm hue
        ("#e74c3c", "#fff", "#fadbd8", "#7b0c0c"),  # red
        ("#e67e22", "#fff", "#fae5d3", "#7a3a0a"),  # orange
        ("#e91e63", "#fff", "#fce4ec", "#7c0024"),  # magenta/pink
        ("#f39c12", "#1a1a1a", "#fef9e7", "#5d3a00"),  # amber
        ("#c0392b", "#fff", "#f5b7b1", "#6e1006"),  # dark red
        ("#d35400", "#fff", "#fad5b0", "#6b2800"),  # burnt orange
        ("#ec407a", "#fff", "#fce8ef", "#7a0036"),  # rose
        ("#f57c00", "#fff", "#fff0d9", "#6b3800"),  # deep orange
    ],
    [  # axis-1: COOL — each value is a clearly different cool hue
        ("#2980b9", "#fff", "#d6eaf8", "#0d2f6e"),  # blue
        ("#1abc9c", "#fff", "#d1f2eb", "#0a4038"),  # teal
        ("#8e44ad", "#fff", "#e8daef", "#4a1a72"),  # purple
        ("#00acc1", "#fff", "#e0f7fa", "#00474f"),  # cyan
        ("#3f51b5", "#fff", "#e8eaf6", "#1a237e"),  # indigo
        ("#16a085", "#fff", "#cde8e4", "#0a3630"),  # dark teal
        ("#6c3483", "#fff", "#e4d0ef", "#3b1260"),  # deep purple
        ("#0288d1", "#fff", "#e1f1fb", "#013d6e"),  # light blue
    ],
    [  # axis-2: NATURE — greens/lime/forest
        ("#27ae60", "#fff", "#d5f5e3", "#0b3c2e"),  # green
        ("#8bc34a", "#1a1a1a", "#f1f8e9", "#33691e"),  # lime green
        ("#00695c", "#fff", "#cce5e2", "#00332e"),  # forest
        ("#558b2f", "#fff", "#dcedc8", "#2a4200"),  # olive
        ("#2e7d32", "#fff", "#c8e6c9", "#0a2d0b"),  # dark green
        ("#76ff03", "#1a1a1a", "#f4ffe0", "#3a5f00"),  # neon lime
        ("#1b5e20", "#fff", "#c3e8c4", "#0a1c0a"),  # deep forest
        ("#aed581", "#1a1a1a", "#ecf6dc", "#3a5f00"),  # light olive
    ],
    [  # axis-3: ACCENT — misc vivid
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
    """Return a Pandas Styler with new_cols highlighted; each column gets a distinct palette,
    with the column header rendered darker than the cell background."""
    df_str = df.astype(str)
    valid = [c for c in new_cols if c in df_str.columns]
    styler = df_str.style

    if not valid:
        return styler

    col_pal = {c: _COL_PALETTES[i % len(_COL_PALETTES)] for i, c in enumerate(valid)}

    # ── Cell background ──
    for col, (cbg, cfg, _, _) in col_pal.items():
        styler = styler.set_properties(
            subset=[col], **{"background-color": cbg, "color": cfg}
        )

    # ── Column header background ──
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

    # Method B: set_table_styles – targets Pandas HTML class selectors
    # (th.col_heading.colN).  Works in some Streamlit versions that ignore
    # apply_index but still honour table-level CSS injected via set_table_styles.
    tbl_styles = []
    for col, (_, _, hbg, hfg) in col_pal.items():
        try:
            col_idx = list(df_str.columns).index(col)
            tbl_styles.append({
                "selector": f"th.col_heading.col{col_idx}",
                "props": f"background-color: {hbg} !important; color: {hfg} !important;",
            })
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
    """Render a structured before → after integration preview.

    Multi-axis coloring: each axis dimension gets a distinct color family
    (green / blue / purple / orange). Within each family, unique axis values
    each receive a distinct shade. Source table cards show per-axis colored
    pills; the integrated table colors each axis column's cells by value.
    First 3 source tables are shown directly; additional tables are in an
    expander.

    full_df_size: (n_rows, n_cols) of the actual full integrated table;
                  shown at the bottom-right of the 統合後 preview.
    source_ids:   list of source table IDs to display after the expander.
    """
    col_names = getattr(ir, "new_column_names", []) or [ir.new_column_name]
    multi_vals = getattr(ir, "new_column_multi_values", {}) or {}
    n_preview = 2 if compact else 3

    # ── Build per-axis unique-value order ───────────────────────────────────
    axis_val_order: list = [[] for _ in col_names]
    for tid in ir.table_ids:
        vals = multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
        for ai, v in enumerate(vals):
            if ai < len(axis_val_order):
                sv = str(v)
                if sv not in axis_val_order[ai]:
                    axis_val_order[ai].append(sv)

    def _axis_color(ai: int, val: str) -> tuple:
        """Return (hdr_bg, hdr_fg, cell_bg, cell_fg) for the given axis+value."""
        family = _AXIS_FAMILIES[ai % len(_AXIS_FAMILIES)]
        vals_list = axis_val_order[ai] if ai < len(axis_val_order) else []
        vi = vals_list.index(val) if val in vals_list else 0
        return family[vi % len(family)]

    # ── Helpers: seamless header row (HTML) + interactive dataframes ────────
    # Headers rendered as one HTML flex row → zero gap between cards.
    # Dataframes rendered with st.dataframe() → scroll + fullscreen work.
    import html as _html

    def _src_header_row_html(tids: list) -> str:
        cells = []
        for j, tid in enumerate(tids):
            vals = multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
            pills = "".join(
                f'<span style="background:{_axis_color(ai, str(v))[0]};'
                f'color:{_axis_color(ai, str(v))[1]};'
                f'padding:2px 9px;border-radius:12px;font-size:0.72rem;font-weight:600;'
                f'margin-right:4px;display:inline-block;margin-bottom:3px;">'
                f'{_html.escape(cn)}:&nbsp;{_html.escape(str(v))}</span>'
                for ai, (cn, v) in enumerate(zip(col_names, vals))
            )
            bl = "border-left:1px solid #2d3748;" if j > 0 else ""
            cells.append(
                f'<div style="flex:1;min-width:0;background:#161c2c;{bl}padding:7px 10px;">'
                f'<div style="font-size:0.7rem;color:#7a8599;margin-bottom:4px;">'
                f'📋&nbsp;{_html.escape(tid)}</div>{pills}</div>'
            )
        return (
            '<div style="display:flex;border:1px solid #2d3748;'
            'border-radius:6px 6px 0 0;overflow:hidden;margin-bottom:0;">'
            + "".join(cells) + '</div>'
        )

    def _src_dataframe_row(tids: list) -> None:
        cols = st.columns(len(tids), gap="small")
        for i, tid in enumerate(tids):
            t = tables_dict.get(tid)
            with cols[i]:
                if t is not None and t.df is not None and not t.df.empty:
                    prev = t.df.head(n_preview)
                    st.dataframe(
                        prev.astype(str),
                        use_container_width=True,
                        hide_index=True,
                        height=min(len(prev) * 35 + 38, 128),
                    )
                else:
                    st.caption("（データなし）")

    # ════════════════════════════════════════════════════════════════════════
    # 統合前
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:6px 0 14px;">'
        '<div style="width:5px;height:22px;background:#4a7de0;border-radius:3px;flex-shrink:0;"></div>'
        '<span style="font-size:1.05rem;font-weight:800;color:#c8d4e8;letter-spacing:.04em;">統合前</span>'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,rgba(74,125,224,.4),transparent);"></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    SAMPLE_LIMIT = 3
    preview_tids = ir.table_ids[:SAMPLE_LIMIT]
    extra_tids = ir.table_ids[SAMPLE_LIMIT:]

    if preview_tids:
        st.markdown(_src_header_row_html(preview_tids), unsafe_allow_html=True)
        _src_dataframe_row(preview_tids)

    if extra_tids:
        with st.expander(f"他 {len(extra_tids)} テーブルを見る", expanded=False):
            n_ex = min(len(extra_tids), 3)
            for start in range(0, len(extra_tids), n_ex):
                chunk = extra_tids[start:start + n_ex]
                st.markdown(_src_header_row_html(chunk), unsafe_allow_html=True)
                _src_dataframe_row(chunk)

    # ── 統合元テーブル一覧 (エクスパンダー下) ──────────────────────────────
    # Avoid st.columns() inside the expander — nested columns in the same
    # vertical container can cause unequal width distribution in the source
    # table grid rendered above.
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

    # ── 統合処理 separator ───────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0;margin:22px 0 18px;">'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,transparent,rgba(39,174,96,.5));"></div>'
        '<div style="border:1.5px solid rgba(39,174,96,.7);border-radius:24px;'
        'padding:6px 22px;margin:0 16px;font-size:1.05rem;font-weight:800;'
        'color:#7FFFD4;letter-spacing:.08em;'
        'background:linear-gradient(135deg,rgba(39,174,96,.12),rgba(26,188,156,.08));'
        'display:flex;align-items:center;gap:8px;white-space:nowrap;">'
        '↓&nbsp;&nbsp;統合処理'
        '</div>'
        '<div style="flex:1;height:1px;background:linear-gradient(to left,transparent,rgba(39,174,96,.5));"></div>'
        '</div>',
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
        '</div>',
        unsafe_allow_html=True,
    )

    frames = []
    for tid in ir.table_ids:
        t = tables_dict.get(tid)
        if t is not None and t.df is not None and not t.df.empty:
            row = t.df.head(n_preview).copy()
            vals = multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
            for ci in range(len(col_names) - 1, -1, -1):
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

        # Color each axis cell by its VALUE within that axis's color family
        def _color_row(row):
            s = pd.Series("", index=row.index)
            for ai, c in enumerate(valid_cols):
                if c in row.index:
                    _, _, cbg, cfg = _axis_color(ai, str(row[c]))
                    s[c] = f"background-color:{cbg};color:{cfg};"
            return s

        styler = df_str.style.apply(_color_row, axis=1) if valid_cols else df_str.style

        # Axis column headers: use each axis family's base (shade 0) header color
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
                tbl_styles.append({
                    "selector": f"th.col_heading.col{ci}",
                    "props": (
                        f"background-color:{hbg} !important;"
                        f"color:{hfg} !important;font-weight:bold !important;"
                    ),
                })
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
            height=min(n_data * 35 + 38, 200 if compact else 320),
        )
        # Full integrated table size – shown bottom-right of the preview
        if full_df_size:
            n_rows, n_cols_full = full_df_size
            st.markdown(
                f'<div style="text-align:right;font-size:0.72rem;'
                f'color:#7a8599;margin-top:2px;">'
                f'📊 {n_rows:,} 行 × {n_cols_full} 列</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.caption("（プレビュー生成不可）")


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
            if st.button("✅ 選択中", key=f"sel_{tid}", use_container_width=True, type="primary"):
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
            # Description block ABOVE the before/after preview so users read
            # what the integration does before seeing the table comparison.
            st.markdown(f"_{info['description']}_")
            if info.get("reasoning"):
                st.caption(f"💡 {info['reasoning']}")

            # Before/after view — source list and size are rendered inside
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
            # Standard view for non-integrated tables
            _splitter_marker(f"s5-{tid}")
            col_prev, col_info = st.columns([1, 1])
            with col_prev:
                _new_cols = info.get("new_col_names") or []
                st.dataframe(
                    _styled_df(df, _new_cols) if _new_cols else df.astype(str),
                    use_container_width=True,
                    hide_index=True,
                    height=220,
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
    st.header("📋 ステップ 5 : テーブル選択")

    # Reset splitter positions in localStorage whenever Step 5 loads so that
    # stale drag positions from a previous visit cannot collapse the right column.
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
        st.button("← 戻る", on_click=_go_to, args=(4,))
        return

    # Backfill new_col_names for entries that were loaded from an older .tep file
    # which was saved before this field was added.  Derive from ai_analysis so
    # that column highlighting works even after a project restore.
    _analysis = st.session_state.get("ai_analysis")
    if _analysis:
        _ir_map = {
            f"integrated_{ir.recommendation_id}": ir
            for ir in _analysis.integration_recommendations
        }
        for k, info in final.items():
            if info.get("is_integrated") and not info.get("new_col_names") and k in _ir_map:
                ir = _ir_map[k]
                info["new_col_names"] = (
                    getattr(ir, "new_column_names", []) or [ir.new_column_name]
                )

    # Auto-processing terminates at step 5
    if st.session_state.auto_processing:
        if st.session_state.run_mode == "fullauto":
            # Full-auto: auto-select recommended tables and proceed to export
            recommended = {
                tid for tid, info in final.items() if info.get("recommended", False)
            }
            st.session_state.selected_ids = (
                recommended if recommended else set(final.keys())
            )
            st.session_state.auto_processing = False
            st.session_state.step = 6
            st.rerun()
        else:
            # Semi-auto: forward pass ends here — user selects tables manually
            st.session_state.auto_processing = False

    st.info(
        "分析対象とするテーブルを選択してください。"
        "推奨テーブルは初期選択済みです（個別に変更できます）。"
    )

    st.markdown(
        f"**選択中: {len(st.session_state.selected_ids)} / {len(final)} テーブル**"
    )

    # Bulk buttons
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

    # Build lookups for the integrated table before/after display
    _s5_analysis = st.session_state.get("ai_analysis")
    _s5_ir_by_rec: dict = {}
    if _s5_analysis:
        _s5_ir_by_rec = {
            ir.recommendation_id: ir
            for ir in _s5_analysis.integration_recommendations
        }
    _s5_tbls = {t.table_id: t for t in st.session_state.get("detected_tables", [])}

    def _get_ir_for(k: str):
        if k.startswith("integrated_"):
            return _s5_ir_by_rec.get(k[len("integrated_"):])
        return None

    # --- Integrated tables (grouped by column signature) ---
    integrated = {k: v for k, v in final.items() if v["is_integrated"]}
    if integrated:
        st.markdown("### 🔀 統合テーブル")

        def _integrated_col_sig(info: dict) -> frozenset:
            df = info.get("df")
            if df is not None:
                return frozenset(df.columns)
            return frozenset()

        # Group by column signature: representative first, similar ones collapsed
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

            _table_card(rep_tid, rep_info, ir=_get_ir_for(rep_tid), tables_dict=_s5_tbls)

            if similar_int:
                with st.expander(
                    f"同様の統合テーブル 他 {len(similar_int)} 件",
                    expanded=False,
                ):
                    for tid, info in similar_int:
                        _table_card(tid, info, ir=_get_ir_for(tid), tables_dict=_s5_tbls)

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
        st.button("← 戻る", on_click=_go_to, args=(4,))
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
                args=(6,),
            )


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
        st.button("← テーブル選択に戻る", on_click=_go_to, args=(5,))
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

    # Save callback stores result in session_state so it survives the rerun
    # and can be shown as a toast from outside the header container.
    if "_save_result" in st.session_state:
        _msg = st.session_state.pop("_save_result")
        st.toast(_msg, icon="💾" if _msg.startswith("✅") else "❌")

    with st.container():
        _render_header()

    # Fire portal-sync before step functions so the header updates immediately,
    # even during long operations like the Step 3 AI call (which sends intermediate
    # state via st.spinner before _inject_splitter_js at the end of main() runs).
    _inject_splitter_js()

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
