import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from steps.shared import _go_to, _reset
from steps.step6_select import _granularity_badge
from src.models import DetectedTable
from src.step7_export import build_export_zip, df_to_csv_bytes, safe_filename

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
                st.download_button(
                    "⬇️ CSV",
                    data=df_to_csv_bytes(df),
                    file_name=f"{safe_name}.csv",
                    mime="text/csv",
                    key=f"dl_{tid}",
                    use_container_width=True,
                )

    st.divider()
    if st.button("🔄 新しいファイルを分析する", use_container_width=True):
        _reset()


