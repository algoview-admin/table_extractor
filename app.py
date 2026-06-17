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
        st.title("📊 Table Extractor AI")
        st.caption("Excel / CSV ファイルから分析対象とするテーブルを抽出します。")
    with save_col:
        if st.session_state.get("filename"):
            has_data = bool(st.session_state.get("detected_tables"))
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "💾 Save",
                key="hdr_save_btn",
                use_container_width=True,
                disabled=not has_data,
                help=f"現在の解析状態を {_PROJECT_DIR} に保存します。",
            ):
                msg = _save_project_to_disk()
                if msg.startswith("✅"):
                    st.toast(msg, icon="💾")
                else:
                    st.toast(msg, icon="❌")

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
                if col.button(f"✅ {label}", key=f"nav_{i}", use_container_width=True):
                    st.session_state.auto_processing = False
                    st.session_state.step = i
                    st.rerun()
            elif accessible:
                # Future step with data available — clickable
                if col.button(f"○ {label}", key=f"nav_{i}", use_container_width=True):
                    st.session_state.auto_processing = False
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
            if st.button(
                btn_labels[selected_mode], type="primary", use_container_width=True
            ):
                st.session_state.file_content = content
                st.session_state.filename = uploaded.name
                st.session_state.file_ext = ext
                st.session_state.detected_tables = []  # force re-parse
                st.session_state.ai_analysis = None
                st.session_state.final_tables = {}
                st.session_state.selected_ids = set()
                st.session_state.auto_processing = selected_mode != "manual"
                st.session_state.source_mode = "new_file"
                st.session_state.step = 2
                st.rerun()

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
        if st.button("← 戻る"):
            st.session_state.auto_processing = False
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
        if st.button("← 戻る"):
            st.session_state.auto_processing = False
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
    st.header("✅ ステップ 4 : 新規テーブル確認")

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
        with st.spinner("新規テーブル・マスタを自動設定中..."):
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
            if st.button("← 戻る"):
                st.session_state.auto_processing = False
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

        ir_groups = _group_irs_by_similarity(
            analysis.integration_recommendations, tables_dict
        )

        def _render_ir_card(ir, tables_dict):
            with st.container(border=True):
                st.markdown(f"#### {ir.group_name}")
                st.markdown(f"_{ir.description}_")

                _splitter_marker(f"s4-ir-{ir.recommendation_id}")
                c_prev, c_info = st.columns([1, 1])
                with c_prev:
                    st.caption("統合後プレビュー（先頭 2 テーブル × 2 行）")
                    _pv_col_names = getattr(ir, "new_column_names", []) or [
                        ir.new_column_name
                    ]
                    _pv_multi_vals = getattr(ir, "new_column_multi_values", {}) or {}
                    preview_frames = []
                    for tid in ir.table_ids[:2]:
                        t = tables_dict.get(tid)
                        if t and t.df is not None:
                            row = t.df.head(2).copy()
                            vals = _pv_multi_vals.get(tid) or [
                                ir.new_column_values.get(tid, "")
                            ]
                            for i in range(len(_pv_col_names) - 1, -1, -1):
                                val = vals[i] if i < len(vals) else ""
                                row.insert(0, _pv_col_names[i], val)
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
        if st.button("← 戻る"):
            st.session_state.auto_processing = False
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
        # Title: always above the table/description split
        st.markdown(
            f"**{info['display_name']}** &nbsp; `{tid}` {badge}",
            unsafe_allow_html=True,
        )

        # Marker placed immediately before st.columns so JS finds the right block
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
        if st.button("← 戻る"):
            st.session_state.step = 4
            st.rerun()
        return

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

            _table_card(rep_tid, rep_info)

            if similar_int:
                with st.expander(
                    f"同様の統合テーブル 他 {len(similar_int)} 件",
                    expanded=False,
                ):
                    for tid, info in similar_int:
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
            st.session_state.auto_processing = False
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
            st.session_state.auto_processing = False
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
