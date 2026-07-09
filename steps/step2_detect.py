from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from steps.shared import _go_to
from src.step1_parser import parse_csv, parse_excel
from src.models import DetectedTable

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

