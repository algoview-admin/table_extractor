import io
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import streamlit as st
import streamlit.components.v1 as components

from src.models import AIAnalysisResult, DetectedTable

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
        "auto_completed": False,  # True when auto-run naturally reached its stop point
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
    from steps.step4_analyze import _build_final_tables
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
    st.caption(
        "保存された `.tep` ファイルは Step 1「プロジェクト読込」で再開できます。"
    )
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
        if st.session_state.get("filename") and bool(
            st.session_state.get("detected_tables")
        ):
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
        slot_half = 100.0 / (2 * n_steps)  # half a slot width in %
        fill_w = 2 * (current - 1) * slot_half  # fill ends at current dot center
        bg_style = f"left:{slot_half:.3f}%;right:{slot_half:.3f}%"
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
            f"</div>",
            unsafe_allow_html=True,
        )

    # 自動処理バナーはコンテナ外（固定ヘッダーのクローン対象外）に配置し
    # プログレスバーとの重なりを防ぐ。
    # run_mode で表示有無を判定することで、タブ移動後も消えないようにする。
    _mode = st.session_state.get("run_mode", "manual")
    pass  # 処理中表示は進捗バーで代替するため削除

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
