import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from streamlit_ui.shared import _go_to, _load_project
from src.models import DetectedTable


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
        model = os.getenv("OPENAI_MODEL", "gpt-5.4")
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
        # ウィジェットkeyは"run_mode_widget"とし、論理的な状態("run_mode")
        # とは分離する。両者を同じキーにすると_load_project()の一括復元と
        # 衝突してStreamlitAPIExceptionになる。
        if "run_mode_widget" not in st.session_state:
            st.session_state.run_mode_widget = st.session_state.run_mode
        selected_mode = st.radio(
            "モードを選択してください",
            options=mode_keys,
            format_func=lambda k: mode_labels[k],
            key="run_mode_widget",
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
                st.session_state.tables_normalized = False
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
