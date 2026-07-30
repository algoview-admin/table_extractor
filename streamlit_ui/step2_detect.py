import html as _html
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from streamlit_ui.shared import _go_to
from src.step1_upload import load_csv, load_excel
from src.step2_detect import (
    detect_tables,
    get_original_df,
    build_tree_text,
    group_tables_by_sheet,
)
from src.models import DetectedTable


def _render_original_df(df: "pd.DataFrame") -> None:
    """検出直後の生データを表示する。

    st.dataframe はグリッド描画時にセル先頭の半角スペースを視覚的に
    保持しない（階層圧縮カラムの展開機能が検出対象とする「先頭空白の
    段差」がここでは見えなくなり、元データの確認という目的を果たせない）
    ため、先頭の半角スペースを &nbsp; に変換したHTMLテーブルで表示する。
    """
    has_leading_space = any(
        isinstance(v, str) and v[:1] == " "
        for col in df.columns
        for v in df[col].astype(str)
    )
    if not has_leading_space:
        st.dataframe(df.astype(str), use_container_width=True, hide_index=True)
        return

    def _cell_html(v: object) -> str:
        s = str(v)
        stripped = s.lstrip(" ")
        n_leading = len(s) - len(stripped)
        return "&nbsp;" * n_leading + _html.escape(stripped)

    headers = "".join(
        f"<th style='padding:4px 10px;text-align:left;border-bottom:2px solid rgba(66,153,225,0.5);"
        f"white-space:nowrap;font-weight:600;font-size:13px'>{_html.escape(str(c))}</th>"
        for c in df.columns
    )
    rows = "".join(
        "<tr>" + "".join(
            f"<td style='padding:4px 10px;font-size:13px;white-space:nowrap;"
            f"border-bottom:1px solid rgba(255,255,255,0.06)'>{_cell_html(v)}</td>"
            for v in row
        ) + "</tr>"
        for _, row in df.iterrows()
    )
    st.markdown(
        f"<div style='overflow-x:auto'><table style='border-collapse:separate;border-spacing:0;width:100%'>"
        f"<thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def step2():
    st.header("🔍 ステップ 2 : テーブル検出")

    if not st.session_state.detected_tables:
        with st.spinner("ファイルを解析中..."):
            try:
                ext = st.session_state.file_ext
                if ext == ".csv":
                    sheet_grids, sheets = load_csv(st.session_state.file_content)
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

    # Step3（テーブル整形）が「うち」書き分離等で新規生成したテーブルは、
    # ファイルから実際に検出されたものではないため、このステップの表示
    # （件数・ツリー・シート別一覧）からは除外する。st.session_state.detected_tables
    # 自体は Step4 以降が参照する共有リストのためここではフィルタのみ行い、
    # 変更・上書きはしない。
    tables: List[DetectedTable] = [
        t for t in st.session_state.detected_tables if not getattr(t, "is_step3_derived", False)
    ]
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
                    _render_original_df(orig)
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

