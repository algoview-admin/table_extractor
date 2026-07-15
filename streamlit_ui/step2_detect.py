from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from streamlit_ui.shared import _go_to
from src.step1_upload import load_csv, load_excel
from src.step2_detect import (
    detect_tables,
    detect_from_csv,
    get_original_df,
    build_tree_text,
    group_tables_by_sheet,
)
from src.models import DetectedTable


def step2():
    st.header("🔍 ステップ 2 : テーブル検出")

    if not st.session_state.detected_tables:
        with st.spinner("ファイルを解析中..."):
            try:
                ext = st.session_state.file_ext
                if ext == ".csv":
                    df = load_csv(st.session_state.file_content)
                    tables, sheets = detect_from_csv(df, st.session_state.filename)
                else:
                    sheet_grids, sheets = load_excel(
                        st.session_state.file_content, st.session_state.filename
                    )
                    tables, _ = detect_tables(sheet_grids)
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

    by_sheet = group_tables_by_sheet(tables)

    # ── ツリービュー ───────────────────────────────────────────────────────
    st.code(build_tree_text(st.session_state.filename, sheets, by_sheet), language=None)

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
                orig = get_original_df(t)
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

