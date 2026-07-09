import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from streamlit_ui.shared import _go_to, _reset
from streamlit_ui.step6_select import _granularity_badge
from src.models import DetectedTable
from src.step7_export import (
    build_export_zip,
    df_to_csv_bytes,
    metadata_to_json_bytes,
    safe_filename,
)


def _collect_agg_removed_metadata(info: dict) -> Tuple[List[dict], List[dict]]:
    """info["source_ids"] を辿り、元 DetectedTable の集計除去メタデータを集約する。

    Returns:
        (row_metadata, col_metadata) のタプル。
    """
    tables_by_id = {t.table_id: t for t in st.session_state.detected_tables}
    row_combined: List[dict] = []
    col_combined: List[dict] = []
    for sid in info.get("source_ids", []):
        src = tables_by_id.get(sid)
        if src is not None:
            row_combined.extend(getattr(src, "agg_removed_row_metadata", []))
            col_combined.extend(getattr(src, "agg_removed_col_metadata", []))
    return row_combined, col_combined


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

    # 元テーブルの集計除去メタデータを、選択テーブルの info に付与する
    # （build_export_zip / ダウンロードボタンの両方から参照するため事前に計算）
    for info in selected.values():
        row_meta, col_meta = _collect_agg_removed_metadata(info)
        if row_meta:
            info["agg_removed_row_metadata"] = row_meta
        if col_meta:
            info["agg_removed_col_metadata"] = col_meta

    st.success(f"✅ **{len(selected)} テーブル** のエクスポート準備が完了しました")

    zip_bytes, export_files = build_export_zip(selected)

    # 一括ダウンロード
    stem = Path(st.session_state.filename).stem
    st.download_button(
        "📦 全テーブルを ZIP でまとめてダウンロード",
        data=zip_bytes,
        file_name=f"{stem}_抽出テーブル.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.divider()
    st.markdown("### 個別ダウンロード")

    for tid, info in selected.items():
        df: pd.DataFrame = info["df"]
        safe_name = safe_filename(info["display_name"])
        fname = export_files.get(f"{safe_name}.csv", b"")
        row_meta = info.get("agg_removed_row_metadata", [])
        col_meta = info.get("agg_removed_col_metadata", [])
        has_agg_meta = bool(row_meta or col_meta)

        with st.container(border=True):
            c_info, c_dl = st.columns([4, 1])
            with c_info:
                badge = _granularity_badge(info)
                st.markdown(
                    f"**{info['display_name']}** {badge}",
                    unsafe_allow_html=True,
                )
                _caption = f"📊 {len(df)} 行 × {len(df.columns)} 列  →  `{safe_name}.csv`"
                if has_agg_meta:
                    _caption += f"  ＋  `{safe_name}_metadata.json`"
                st.caption(_caption)
                with st.expander("プレビュー"):
                    st.dataframe(
                        df.astype(str),
                        use_container_width=True,
                        hide_index=True,
                        height=210,
                    )
                    if has_agg_meta:
                        n_total = len(row_meta) + len(col_meta)
                        st.caption(f"📋 集計除去メタデータ（計{n_total}件）")
                        st.json(
                            {
                                "aggregate_rows_removed": row_meta,
                                "aggregate_columns_removed": col_meta,
                            }
                        )
            with c_dl:
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    "⬇️ CSV",
                    data=df_to_csv_bytes(df),
                    file_name=f"{safe_name}.csv",
                    mime="text/csv",
                    key=f"dl_{tid}",
                    use_container_width=True,
                )
                if has_agg_meta:
                    st.download_button(
                        "⬇️ メタデータ",
                        data=metadata_to_json_bytes(row_meta, col_meta),
                        file_name=f"{safe_name}_metadata.json",
                        mime="application/json",
                        key=f"dl_meta_{tid}",
                        use_container_width=True,
                    )

    st.divider()
    if st.button("🔄 新しいファイルを分析する", use_container_width=True):
        _reset()


