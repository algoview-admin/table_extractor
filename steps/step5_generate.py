from typing import Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

from steps.shared import _go_to, _inject_splitter_js, _splitter_marker, _build_and_go_step6
from steps.step4_analyze import (
    _build_final_tables,
    _dlt_virtual_table,
    _clip_ai_irs_by_latent_groups,
    _build_auto_irs_from_latent,
    _render_latent_group_card,
    _render_unified_ir_card,
    _collect_unique_master_specs,
    _group_irs_by_similarity,
)
from src.step5_generate import find_latent_tables, derive_latent_tables, group_latent_proposals
from src.models import AIAnalysisResult, DetectedTable, DerivedLatentTable

def step4():
    st.header("✅ ステップ 5 : 新規テーブル案生成")

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

    if (
        st.session_state.auto_processing
        and st.session_state.get("run_mode") == "fullauto"
    ):
        with st.spinner("新規テーブル案・マスタ案を自動設定中..."):
            _build_final_tables()
        st.session_state.step = 6
        st.rerun()
    elif st.session_state.auto_processing:
        # セミオート: ユーザーが提案を確認できるようここで停止する
        st.session_state.auto_processing = False
        st.session_state.auto_completed = True

    tables_dict = {t.table_id: t for t in st.session_state.detected_tables}

    # ── 潜在グループを計算する（複数のセクションで使用） ───────────────────
    _latent_proposals = find_latent_tables(st.session_state.detected_tables)
    _derived_latent = derive_latent_tables(st.session_state.detected_tables)
    _latent_groups = group_latent_proposals(_latent_proposals, _derived_latent)

    # 拡張 tables_dict: 各DLT用の仮想 DetectedTable エントリ
    _ext_tables = dict(tables_dict)
    for _dlt in _derived_latent:
        _vt = _dlt_virtual_table(_dlt, tables_dict)
        if _vt is not None:
            _ext_tables[_dlt.proposal_id] = _vt

    # 暫定刈り込み: 早期の「表示するものなし」チェックと has_masters の計算にのみ使用する。
    # 完全な動的計算は後で、潜在グループカードがレンダリングされた後（決定が更新された後）に実行する。
    _ai_irs_prelim = _clip_ai_irs_by_latent_groups(
        analysis.integration_recommendations, _latent_groups, tables_dict
    )
    has_integrations = bool(_ai_irs_prelim) or bool(_latent_groups)
    has_masters = bool(_collect_unique_master_specs(_ai_irs_prelim, tables_dict))
    has_latent = bool(_latent_groups)

    if not has_integrations and not has_masters and not has_latent:
        st.info("新規テーブルの生成推奨はありません。このステップはスキップします。")
        c1, c2 = st.columns([1, 4])
        with c1:
            st.button("← 戻る", on_click=_go_to, args=(4,))
        with c2:
            st.button(
                "次へ：テーブル選択 →",
                type="primary",
                use_container_width=True,
                on_click=_build_and_go_step6,
            )
        return

    # ── セクション 1: 潜在テーブル推定（最初に表示） ──────────────────────────
    if has_latent:
        st.subheader("🔢 潜在テーブル推定")
        st.caption(
            "テーブルの注記に記載された集計関係を分析し、未検出テーブルを差分計算で推定しました。"
            "「追加する」を選択すると下の統合テーブル案に自動反映されます。"
        )
        for _group in _latent_groups:
            _render_latent_group_card(_group, tables_dict)

    # ── セクション 2: 統合テーブル ────────────────────────────────────────────
    # auto-IRは現在の潜在グループ決定を動的に反映する
    _auto_irs_flat = _build_auto_irs_from_latent(
        _latent_groups, _ext_tables, tables_dict
    )

    # AI IRを刈り込む: 検証済み兄弟セットに属さない同一sheet内テーブルを除外する。
    # 例: 潜在検出がT2+T3を兄弟と判定した場合、{T2,T3,T4}からX_D（T4）を除外する。
    _ai_irs_trimmed = _clip_ai_irs_by_latent_groups(
        analysis.integration_recommendations, _latent_groups, tables_dict
    )

    # 実検出テーブルが全て承認済みauto-IRに含まれるAI IRを抑制する。
    # 潜在X-3が承認された場合: auto-IRの実={T2,T3} ⊆ AI IR {T2,T3} → AI IRを抑制する。
    # （auto-IRが潜在テーブルを追加することでAI IRを置き換える。）
    _superseded_ai_ids: set = set()
    for _sup_ir, _ in _auto_irs_flat:
        # auto-IR内の実（仮想でない）テーブルID
        _sup_real = {t for t in _sup_ir.table_ids if t in tables_dict}
        if len(_sup_real) < 2:
            continue
        for _ai_ir in _ai_irs_trimmed:
            # AI IRの全テーブルがauto-IRの実テーブルに含まれる場合は抑制する
            # （auto-IRがDLTを加えてAI IRを上書きするため）
            if set(_ai_ir.table_ids).issubset(_sup_real):
                _superseded_ai_ids.add(_ai_ir.recommendation_id)

    # マージ: auto-IRを先頭に（潜在テーブルが承認された際に上部に表示されるよう）、
    # 次に抑制されていないAI IRを続ける。
    _all_irs_flagged: list = [(ir, True) for ir, _ in _auto_irs_flat]
    _all_irs_flagged += [
        (ir, False)
        for ir in _ai_irs_trimmed
        if ir.recommendation_id not in _superseded_ai_ids
    ]
    _is_auto_map: dict = {ir.recommendation_id: flag for ir, flag in _all_irs_flagged}
    _all_irs = [ir for ir, _ in _all_irs_flagged]

    if _all_irs:
        if has_latent:
            st.divider()
        st.subheader("🔀 統合テーブル")
        _int_caption = "各統合について実施するかどうかをお選びください。"
        if has_latent:
            _int_caption += (
                "潜在テーブルを「追加する」にすると、関連する統合提案が自動で追加されます。"
            )
        st.caption(_int_caption)

        # 結合リスト全体で列シグネチャによりグループ化する
        _ir_groups = _group_irs_by_similarity(_all_irs, _ext_tables)

        for _grp in _ir_groups:
            _rep = _grp[0]
            _similar = _grp[1:]
            _render_unified_ir_card(
                _rep,
                _ext_tables,
                is_auto=_is_auto_map.get(_rep.recommendation_id, False),
            )
            if _similar:
                _rep_cols = _rep.new_column_names or [_rep.new_column_name]
                _axes_lbl = " × ".join(_rep_cols)
                with st.expander(
                    f"同様の統合 他 {len(_similar)} 件（{_axes_lbl} 軸）",
                    expanded=False,
                ):
                    for _ir in _similar:
                        _render_unified_ir_card(
                            _ir,
                            _ext_tables,
                            is_auto=_is_auto_map.get(_ir.recommendation_id, False),
                        )

    # ── セクション 3: マスタ自動生成 ─────────────────────────────────────────
    # 動的に再計算する: 刈り込み済みAI IR（X_D除外）+ 承認済みauto-IR
    # （axis_parent_table_idsが設定されているため、マスタ生成がX_3マッピングを生成できる）を使用する。
    # 仮想DLTテーブル（X_3）を解決するため _ext_tables を使用する。
    _master_ir_list = [ir for ir, _ in _auto_irs_flat] + [
        ir for ir in _ai_irs_trimmed if ir.recommendation_id not in _superseded_ai_ids
    ]
    unique_master_specs = _collect_unique_master_specs(_master_ir_list, _ext_tables)
    _has_masters_now = bool(unique_master_specs)

    if _has_masters_now:
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

    # ── ナビゲーション ────────────────────────────────────────────────────────
    _inject_splitter_js()
    st.divider()
    c1, c2 = st.columns([1, 4])
    with c1:
        st.button("← 戻る", on_click=_go_to, args=(4,))
    with c2:
        st.button(
            "次へ：テーブル選択 →",
            type="primary",
            use_container_width=True,
            on_click=_build_and_go_step6,
        )


# ---------------------------------------------------------------------------
# 最終テーブルリストを構築する（Step 4 離脱時に呼び出す）
# ---------------------------------------------------------------------------


