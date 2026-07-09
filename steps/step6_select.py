from typing import Dict, List, Optional, Set

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from steps.shared import _go_to, _inject_splitter_js, _splitter_marker, _build_and_go_step6
from steps.step4_analyze import (
    _render_integration_before_after,
    _render_derived_before_after,
    _COL_PALETTES,
)
from src.models import AIAnalysisResult, DetectedTable, DerivedLatentTable, IntegrationRecommendation
from src.step6_select import (
    granularity_badge as _granularity_badge,
    group_final_tables,
    group_integrated_by_columns,
    safe_table_filename,
)



def _styled_df(df: "pd.DataFrame", new_cols: list) -> "pd.io.formats.style.Styler":
    """new_colsをハイライトした Pandas Styler を返す。各列に異なるパレットを割り当て、
    列ヘッダーはセル背景より暗くレンダリングされる。"""
    df_str = df.astype(str)
    valid = [c for c in new_cols if c in df_str.columns]
    styler = df_str.style

    if not valid:
        return styler

    col_pal = {c: _COL_PALETTES[i % len(_COL_PALETTES)] for i, c in enumerate(valid)}

    # ── セル背景 ──
    for col, (cbg, cfg, _, _) in col_pal.items():
        styler = styler.set_properties(
            subset=[col], **{"background-color": cbg, "color": cfg}
        )

    # ── 列ヘッダー背景 ──
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

    # Method B: set_table_styles — Pandas HTMLクラスセレクター（th.col_heading.colN）を対象とする。
    # apply_indexを無視するが set_table_styles 経由で注入されたテーブルレベルの
    # CSSには対応しているStreamlitバージョンで機能する。
    tbl_styles = []
    for col, (_, _, hbg, hfg) in col_pal.items():
        try:
            col_idx = list(df_str.columns).index(col)
            tbl_styles.append(
                {
                    "selector": f"th.col_heading.col{col_idx}",
                    "props": f"background-color: {hbg} !important; color: {hfg} !important;",
                }
            )
        except ValueError:
            pass
    if tbl_styles:
        try:
            styler = styler.set_table_styles(tbl_styles, overwrite=False)
        except Exception:
            pass

    return styler



def _table_card(tid: str, info: dict, ir=None, tables_dict=None):
    df: pd.DataFrame = info["df"]
    is_sel = tid in st.session_state.selected_ids
    badge = _granularity_badge(info)

    def _sel_button():
        if is_sel:
            if st.button(
                "✅ 選択中", key=f"sel_{tid}", use_container_width=True, type="primary"
            ):
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
            # 統合前/統合後プレビューの上に説明ブロックを表示し、
            # テーブル比較を見る前に統合内容をユーザーが読めるようにする。
            st.markdown(f"_{info['description']}_")
            if info.get("reasoning"):
                st.caption(f"💡 {info['reasoning']}")

            # 統合前/統合後ビュー — ソース一覧とサイズは内部でレンダリングする
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
            # 非統合テーブルの標準ビュー
            _splitter_marker(f"s5-{tid}")
            col_prev, col_info = st.columns([1, 1])
            with col_prev:
                _new_cols = info.get("new_col_names") or []
                _trig_cols = info.get("trigger_col_names") or []
                _row_px, _hdr_px, _max_visible = 35, 38, 10
                # 軸列 → 軸カラー、trigger列（年軸省略理由）→ 琥珀色
                _df_disp = _styled_df(df, _new_cols) if _new_cols else df.astype(str)
                if _trig_cols:
                    _df_str = df.astype(str)
                    _valid_trig = [c for c in _trig_cols if c in _df_str.columns]
                    if _valid_trig:

                        def _hl_trig_s5(row, _vtc=_valid_trig):
                            s = pd.Series("", index=row.index)
                            for _c in _vtc:
                                s[_c] = (
                                    "background-color:#3d2e00;color:#ffd369;font-weight:600;"
                                )
                            return s

                        if isinstance(_df_disp, pd.DataFrame):
                            _df_disp = _df_disp.style.apply(_hl_trig_s5, axis=1)
                        else:
                            _df_disp = _df_disp.apply(_hl_trig_s5, axis=1)
                st.dataframe(
                    _df_disp,
                    use_container_width=True,
                    hide_index=True,
                    height=min(
                        len(df) * _row_px + _hdr_px, _max_visible * _row_px + _hdr_px
                    ),
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
    st.header("📋 ステップ 6 : テーブル選択")

    # Step 5 が読み込まれるたびに localStorage のsplitter位置をリセットし、
    # 前回の古いドラッグ位置で右列が折りたたまれないようにする。
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
        st.button("← 戻る", on_click=_go_to, args=(5,))
        return

    # このフィールドが追加される前に保存された古い .tep ファイルから読み込まれたエントリに
    # new_col_names を補完する。プロジェクト復元後も列ハイライトが機能するよう
    # ai_analysis から派生させる。
    _analysis = st.session_state.get("ai_analysis")
    if _analysis:
        _ir_map = {
            f"integrated_{ir.recommendation_id}": ir
            for ir in _analysis.integration_recommendations
        }
        for k, info in final.items():
            if (
                info.get("is_integrated")
                and not info.get("new_col_names")
                and k in _ir_map
            ):
                ir = _ir_map[k]
                info["new_col_names"] = getattr(ir, "new_column_names", []) or [
                    ir.new_column_name
                ]

    # 自動処理はStep 5で終了する
    if st.session_state.auto_processing:
        if st.session_state.run_mode == "fullauto":
            # フルオート: 推奨テーブルを自動選択してエクスポートへ進む
            recommended = {
                tid for tid, info in final.items() if info.get("recommended", False)
            }
            st.session_state.selected_ids = (
                recommended if recommended else set(final.keys())
            )
            st.session_state.auto_processing = False
            st.session_state.auto_completed = True
            st.session_state.step = 7
            st.rerun()
        else:
            # セミオート: 前進パスはここで終了 — ユーザーが手動でテーブルを選択する
            st.session_state.auto_processing = False
            st.session_state.auto_completed = True

    st.info(
        "分析対象とするテーブルを選択してください。"
        "推奨テーブルは初期選択済みです（個別に変更できます）。"
    )

    st.markdown(
        f"**選択中: {len(st.session_state.selected_ids)} / {len(final)} テーブル**"
    )

    # 一括操作ボタン
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

    # 統合テーブルの前/後表示用のルックアップを構築する
    _s5_analysis = st.session_state.get("ai_analysis")
    _s5_ir_by_rec: dict = {}
    if _s5_analysis:
        _s5_ir_by_rec = {
            ir.recommendation_id: ir for ir in _s5_analysis.integration_recommendations
        }
    _s5_tbls = {t.table_id: t for t in st.session_state.get("detected_tables", [])}

    def _get_ir_for(k: str):
        if k.startswith("integrated_"):
            return _s5_ir_by_rec.get(k[len("integrated_") :])
        return None

    integrated, min_tables, master_tables, other_rec, non_rec = group_final_tables(final)

    # --- 統合テーブル（列シグネチャでグループ化） ---
    if integrated:
        st.markdown("### 🔀 統合テーブル")
        for group in group_integrated_by_columns(integrated):
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

    # --- 最小粒度テーブル ---
    if min_tables:
        st.markdown("### ⭐ 最小粒度データ")
        for tid, info in min_tables.items():
            _table_card(tid, info)

    # --- マスタテーブル ---
    if master_tables:
        st.markdown("### 📚 マスタテーブル")
        for tid, info in master_tables.items():
            _table_card(tid, info)

    # --- その他の推奨テーブル ---
    if other_rec:
        st.markdown("### 📊 その他の推奨テーブル")
        for tid, info in other_rec.items():
            _table_card(tid, info)

    # --- 非推奨テーブル（折りたたみ） ---
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
        st.button("← 戻る", on_click=_go_to, args=(5,))
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
                args=(7,),
            )


# ---------------------------------------------------------------------------
# Step 6 — エクスポート
# ---------------------------------------------------------------------------


