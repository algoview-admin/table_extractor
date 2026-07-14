import html as _html
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from streamlit_ui.shared import _go_to, _inject_splitter_js, _splitter_marker, _build_and_go_step6
from src.step4_analyze import analyze_tables
from src.step5_suggest import (
    find_latent_tables, derive_latent_tables, LatentTableGroup, group_latent_proposals,
)
from src.models import (
    AIAnalysisResult, DetectedTable, DerivedLatentTable,
    IntegrationRecommendation, MasterTableInfo,
)

def step3():
    st.header("🧠 ステップ 4 : テーブル関係分析")

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

    # 初回の前進パス中のみ自動的に次のステップへ進む
    if st.session_state.auto_processing:
        st.session_state.step = 5
        st.rerun()

    # サマリーバナー
    st.info(f"📊 **分析サマリー**: {analysis.summary}")

    # シート分類
    with st.expander("📋 シート分類", expanded=True):
        for sc in analysis.sheet_classifications:
            icon = "📊" if sc.is_data_sheet else "📝"
            badge = "データシート" if sc.is_data_sheet else "説明 / 補足シート"
            st.markdown(f"{icon} **{sc.sheet_name}** → {badge}")
            if sc.description:
                st.caption(sc.description)

    # 階層概要
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

    # 最小粒度
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

    # マスタテーブル
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

    # 統合推奨プレビュー
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

    # マスタ生成推奨プレビュー（軸ごと）
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
        st.button("← 戻る", on_click=_go_to, args=(3,))
    with c2:
        st.button(
            "次へ：新規テーブル案生成 →",
            type="primary",
            use_container_width=True,
            on_click=_go_to,
            args=(5,),
        )


def _ir_column_signature(ir, tables_dict) -> frozenset:
    """「類似」統合推奨を検出するために使用する列シグネチャ。

    同じ識別列を持ち、かつソーステーブルのスキーマが同一である場合、
    2つのIRは類似とみなされる。
    """
    col_names = getattr(ir, "new_column_names", []) or [ir.new_column_name]
    for tid in ir.table_ids:
        t = tables_dict.get(tid)
        if t is not None and t.effective_df is not None:
            return frozenset(col_names + list(t.effective_df.columns))
    return frozenset(col_names)


def _detect_redundant_axes(
    col_names: list, multi_vals: dict, ir, tables_dict: dict
) -> dict:
    """統合で追加する新カラムのうち、既存データから導出可能な冗長な軸を検出する。

    新カラム値に含まれる4桁年（YYYY）が、各テーブルの既存カラムのいずれかに
    すでに含まれている場合、そのカラムは冗長と判定する。

    Returns: {col_names_index: {triggering_col_name, ...}} — 冗長と判定した軸インデックスと
             その判断理由となった既存カラム名のセット。in 演算子でインデックス確認可能。
    """
    import re as _r

    redundant: dict = {}
    for ci, col_name in enumerate(col_names):
        # 全テーブルについて「新カラム値の年が既存カラムに存在するか」を確認
        all_found = True
        any_table = False
        trigger_cols: set = set()  # 判断理由となった既存カラム名
        for tid in ir.table_ids:
            t = tables_dict.get(tid)
            if t is None or t.effective_df is None:
                all_found = False
                break
            val = (
                (multi_vals.get(tid) or [])[ci]
                if ci < len(multi_vals.get(tid) or [])
                else ir.new_column_values.get(tid, "")
            )
            year_m = _r.search(r"(19|20)\d{2}", str(val))
            if not year_m:
                all_found = False
                break
            year = year_m.group(0)
            df = t.effective_df
            found_in_col = False
            for ec in df.columns:
                # 新カラム名と同一のカラムは除外（自己参照防止）
                if str(ec) == str(col_name):
                    continue
                non_null = [
                    str(v)
                    for v in df[ec].dropna()
                    if str(v).lower() not in ("nan", "none", "")
                ]
                if not non_null:
                    continue
                match_ratio = sum(1 for v in non_null if year in v) / len(non_null)
                if match_ratio >= 0.5:
                    found_in_col = True
                    trigger_cols.add(str(ec))
                    break
            if not found_in_col:
                all_found = False
                break
            any_table = True
        if all_found and any_table and len(ir.table_ids) >= 2:
            # 統合後も各テーブルの行が既存カラムで自然に区別できるか確認する
            # → 各テーブルで検出された「年を含むカラム」の値が互いに重複しない場合のみ冗長とみなす
            year_val_sets: list = []
            distinguishable = True
            for tid in ir.table_ids:
                t = tables_dict.get(tid)
                val = (
                    (multi_vals.get(tid) or [])[ci]
                    if ci < len(multi_vals.get(tid) or [])
                    else ir.new_column_values.get(tid, "")
                )
                year_m = _r.search(r"(19|20)\d{2}", str(val))
                if not year_m:
                    distinguishable = False
                    break
                year = year_m.group(0)
                df = t.effective_df
                vals_with_year: set = set()
                for ec in df.columns:
                    if str(ec) == str(col_name):
                        continue
                    non_null = [
                        str(v)
                        for v in df[ec].dropna()
                        if str(v).lower() not in ("nan", "none", "")
                    ]
                    if not non_null:
                        continue
                    if sum(1 for v in non_null if year in v) / len(non_null) >= 0.5:
                        vals_with_year.update(non_null)
                year_val_sets.append(vals_with_year)
            # 各テーブルの値集合が互いに重複しなければ自然に区別可能
            if distinguishable and len(year_val_sets) >= 2:
                for i in range(len(year_val_sets)):
                    for j in range(i + 1, len(year_val_sets)):
                        if year_val_sets[i] & year_val_sets[j]:
                            distinguishable = False
                            break
                    if not distinguishable:
                        break
            if distinguishable:
                redundant[ci] = trigger_cols
    return redundant


def _group_irs_by_similarity(irs, tables_dict):
    """リストのリストを返す: 各内部リストは類似IRのグループ。

    各グループの先頭要素が代表（展開表示）となり、
    残りはドロップダウンに折りたたまれる。
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
    """軸の値がsheet名またはセクションタイトルと一致するかを確認し、'sheet'または'title'を返す。"""
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
    """親を持つ軸ごとに1つずつ、マスタspecのリストを返す。

    各specは以下のキーを持つdict:
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

    # 軸ごとの親情報を解決する; axis 0については旧シングル軸フィールドにフォールバック
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

        # 軸の値をメタデータと比較して軸タイプ（sheet vs title）を判定する
        atype = _axis_type(axis_idx, multi_vals, members)

        # 検証: 親は子の兄弟（同レベル）であってはならない
        if atype == "sheet":
            if parent.sheet_name in child_sheets:
                continue  # 親が子と同じsheetにある → 兄弟のため除外
            parent_label = parent.sheet_name
        else:  # title軸
            child_titles = {(m.title or "") for m in members}
            if (parent.title or parent.sheet_name) in child_titles:
                continue  # 親が子と同じタイトルを持つ → 兄弟のため除外
            parent_label = parent.title or parent.sheet_name

        # 実際の軸の値から child → parent_label のマッピングを構築する
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
                    parent_label  # 重複（同じ枝、異なるサービス）は集約する
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
    """後方互換シム: axis 0 の最初のspecのみを返す。存在しない場合はNone。"""
    specs = _derive_master_specs_for_ir(ir, tables_dict)
    return specs[0] if specs else None


def _master_signature(spec):
    """specが生成するマスタの同一性を表す。どの統合由来かに関わらず、
    同じ child→parent ラベルマップを持つマスタは重複とみなす。"""
    return (
        spec["child_col"],
        spec["parent_col"],
        frozenset(spec["mapping"].items()),
    )


def _collect_unique_master_specs(irs, tables_dict) -> list:
    """全IRの全軸にわたって重複排除した (ir, spec) ペアのリストを返す。"""
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
    """後方互換シム: 有効なmaster specを少なくとも1つ持つIRを返す。"""
    seen_ir_ids: set = set()
    result = []
    for ir, _spec in _collect_unique_master_specs(master_irs, tables_dict):
        if ir.recommendation_id not in seen_ir_ids:
            seen_ir_ids.add(ir.recommendation_id)
            result.append(ir)
    return result


# ---------------------------------------------------------------------------
# 潜在テーブルヘルパー（Step 4）
# ---------------------------------------------------------------------------


def _dlt_virtual_table(
    dlt: DerivedLatentTable, tables_dict: dict
) -> Optional[DetectedTable]:
    """統合プレビュー用に DerivedLatentTable を仮想 DetectedTable としてラップする。"""
    parent = tables_dict.get(dlt.parent_table_id)
    if parent is None or dlt.df is None:
        return None
    return DetectedTable(
        table_id=dlt.proposal_id,
        sheet_name=parent.sheet_name,
        start_row=0,
        end_row=len(dlt.df),
        start_col=0,
        end_col=len(dlt.df.columns),
        df=dlt.df.copy(),
        title=dlt.derived_name,
    )


def _infer_col_name_from_values(values: List[str]) -> str:
    """値ラベルから妥当な列名を推定する。

    共通prefixの末尾にある区切り文字・数字を除去した語幹を返す。
    意味のある共通プレフィックスが見つからない場合は '種別' にフォールバックする。
    """
    import os.path as _osp

    if len(values) < 2:
        return "種別"
    prefix = _osp.commonprefix(values)
    prefix = re.sub(r"[-_・\s\d]+$", "", prefix).strip()
    return prefix if len(prefix) >= 2 else "種別"


def _infer_axis_name(values: List[str]) -> str:
    """値ラベル群から軸名称を動的に推定する。

    Step 1: 共通prefix（末尾の区切り・数字・英字を除去）で語幹を取り出す。
    Step 2: Step1 が失敗した場合、全値に共通して含まれる最長の部分文字列を探す。
            意味のある文字（日本語等）を含まない候補は除外する。
    どちらも失敗した場合は '種別' を返す。
    """
    import os.path as _osp

    if not values:
        return "種別"
    if len(values) == 1:
        return values[0]

    # Step 1: 共通prefix（末尾の区切り文字・数字・英字を除去）
    prefix = _osp.commonprefix(values)
    prefix_clean = re.sub(r"[-_・\s\dA-Za-zＡ-Ｚａ-ｚ０-９]+$", "", prefix).strip()
    if len(prefix_clean) >= 2:
        return prefix_clean

    # Step 2: 全値に含まれる最長の共通部分文字列を探す
    # 数字・ASCII英字・記号のみで構成される部分文字列は意味が薄いため除外する
    _HAS_MEANINGFUL = re.compile(r"[^\d\s\-_・A-Za-zＡ-Ｚａ-ｚ０-９（）()【】「」]")
    shortest = min(values, key=len)
    for length in range(len(shortest), 1, -1):
        for start in range(len(shortest) - length + 1):
            substr = shortest[start : start + length]
            if not _HAS_MEANINGFUL.search(substr):
                continue
            if all(substr in v for v in values):
                return substr.strip()

    return "種別"


def _strip_common_suffix(labels: List[str]) -> List[str]:
    """全ラベルに共通する末尾部分を除去して識別部分のみを返す。

    各ラベルから共通末尾（区切り文字含む）を取り除き、
    集計元を識別する語幹部分のみを返す。
    意味のある共通末尾が2文字未満の場合はそのまま返す。
    """
    import os.path as _osp

    if len(labels) < 2:
        return list(labels)

    rev = [lbl[::-1] for lbl in labels]
    suf_raw = _osp.commonprefix(rev)[::-1]

    # 区切り文字を除いた実質的な共通末尾が2文字以上あるか確認する
    suf_meaningful = re.sub(r"^[-_・\s（）()「」「」]+", "", suf_raw)
    if len(suf_meaningful) < 2:
        return list(labels)

    result = []
    for lbl in labels:
        if lbl.endswith(suf_raw):
            stripped = lbl[: -len(suf_raw)].rstrip("-_・ （）()「」「」").strip()
            result.append(stripped if stripped else lbl)
        else:
            result.append(lbl)
    return result


def _build_auto_irs_from_latent(
    latent_groups: List[LatentTableGroup],
    ext_tables_dict: dict,
    tables_dict: dict,
    _for_build: bool = False,
) -> List[tuple]:
    """承認された潜在グループに対して IntegrationRecommendation オブジェクトを構築する。

    (IntegrationRecommendation, group_key) タプルのフラットリストを返す。
    latent_group_decisions[group_key] が True かつDLTが少なくとも1つ存在するグループのみ
    IRを生成する。

    優先度ルール（グループに2軸統合IRが生成できる場合）:
      - 2軸IRは常に result に含める（ユーザーがいつでも選択を変更できるようにする）
      - 2軸統合が「統合しない」の場合のみ、1軸IRも追加する
    2軸統合が生成できない（メンバーが1件のみ）場合は常に1軸IRを返す。
    """
    result = []
    lg_dec = st.session_state.get("latent_group_decisions", {})
    if "latent_auto_int_decisions" not in st.session_state:
        st.session_state.latent_auto_int_decisions = {}
    auto_int_dec = st.session_state.latent_auto_int_decisions

    for group in latent_groups:
        gk = group.group_key
        if not lg_dec.get(gk, True):
            continue
        if not group.has_derived:
            continue

        members_with_dlt = [(lp, dlt) for lp, dlt in group.members if dlt is not None]

        # ── クロスシート集計メンバーを2軸統合から除外する ────────────────────────
        # あるメンバーの親テーブルの数値合計が他の全メンバーの合計に近い場合（許容誤差5%以内）、
        # そのメンバーは他シートを束ねたクロスシート集計であると判定し除外する。
        # 例: 拠点A + 拠点B = 全拠点集計 の場合、全拠点集計メンバーは冗長として除外する。
        if len(members_with_dlt) >= 3:

            def _num_total(tid: str) -> float:
                t = tables_dict.get(tid)
                if t is None or t.effective_df is None or t.effective_df.empty:
                    return 0.0
                try:
                    return float(
                        t.effective_df.select_dtypes(include="number")
                        .abs()
                        .values.sum()
                    )
                except Exception:
                    return 0.0

            _parent_totals = [
                _num_total(dlt.parent_table_id) for _, dlt in members_with_dlt
            ]
            _non_summary = []
            for _i, (lp, dlt) in enumerate(members_with_dlt):
                _others = sum(v for _j, v in enumerate(_parent_totals) if _j != _i)
                _this = _parent_totals[_i]
                if _others > 1e-9 and _this > 1e-9:
                    _ratio = abs(_this - _others) / max(_this, _others)
                    if _ratio < 0.05:
                        continue  # クロスシート集計 → 除外
                _non_summary.append((lp, dlt))
            if len(_non_summary) >= 2:
                members_with_dlt = _non_summary

        # ── 2軸統合IR を構築する（メンバーが2件以上の場合）─────────────────────
        cross_ir = None
        cross_rec_id = f"latent_auto_cross2_{gk}"

        if len(members_with_dlt) >= 2:
            member_sheets = [
                (
                    ext_tables_dict.get(lp.source_table_id)
                    or tables_dict.get(lp.source_table_id)
                )
                for lp, _ in members_with_dlt
            ]
            sheet_names_of_members = [t.sheet_name if t else "" for t in member_sheets]
            is_multi_sheet = len(set(sheet_names_of_members)) > 1

            # 軸2の値: 複数シートはsheet_name、同一シートはsource_titleから識別部分を抽出
            if is_multi_sheet:
                raw_axis2_vals = sheet_names_of_members
            else:
                raw_axis2_vals = [
                    (t.title or lp.source_table_id)
                    for (lp, _), t in zip(members_with_dlt, member_sheets)
                ]
                raw_axis2_vals = _strip_common_suffix(raw_axis2_vals)

            cross_ids: list = []
            cross_multi_vals: dict = {}
            for (lp, dlt), axis2_val in zip(members_with_dlt, raw_axis2_vals):
                for i, tid in enumerate(lp.detected_table_ids):
                    cat_val = (
                        lp.detected_names[i] if i < len(lp.detected_names) else tid
                    )
                    cross_ids.append(tid)
                    cross_multi_vals[tid] = [cat_val, axis2_val]
                cross_ids.append(dlt.proposal_id)
                cross_multi_vals[dlt.proposal_id] = [dlt.derived_name, axis2_val]

            if len(cross_ids) >= 4:
                cat_names_all = group.detected_names + group.missing_names
                cat_col_name = _infer_col_name_from_values(cat_names_all)
                axis2_unique = list(
                    dict.fromkeys(cross_multi_vals[tid][1] for tid in cross_ids)
                )
                axis2_col_name = _infer_axis_name(axis2_unique)
                if not axis2_col_name or axis2_col_name == "種別":
                    axis2_col_name = "シート" if is_multi_sheet else "区分"

                # デフォルトは「統合する」(True)
                if cross_rec_id not in auto_int_dec:
                    auto_int_dec[cross_rec_id] = True

                cross_ir = IntegrationRecommendation(
                    recommendation_id=cross_rec_id,
                    group_name=(
                        f"{'・'.join(group.missing_names)} × {axis2_col_name} "
                        f"2軸統合テーブル"
                    ),
                    description=(
                        f"差分推定した「{'・'.join(group.missing_names)}」を含む"
                        f"「{'・'.join(cat_names_all)}」を"
                        f"全{axis2_col_name}横断で統合した2軸テーブル"
                    ),
                    table_ids=cross_ids,
                    new_column_name=cat_col_name,
                    new_column_values={
                        tid: cross_multi_vals[tid][0] for tid in cross_ids
                    },
                    reasoning=(
                        f"潜在テーブル「{'・'.join(group.missing_names)}」の差分推定により、"
                        f"「{cat_col_name}」軸と「{axis2_col_name}」軸の2軸で"
                        f"全{axis2_col_name}のテーブルを統合できます。"
                    ),
                    new_column_names=[cat_col_name, axis2_col_name],
                    new_column_multi_values={
                        tid: list(v) for tid, v in cross_multi_vals.items()
                    },
                    # 構成要素軸(axis 0)の親を設定することでマスタ生成が機能する。
                    # 集計元軸(axis 1)は親なし（集計元自体が軸なのでマスタ不要）。
                    axis_parent_table_ids=[
                        members_with_dlt[0][0].source_table_id,
                        None,
                    ],
                    axis_parent_label_columns=[None, None],
                )

        # ── 1軸統合IR を構築する（集計元ごと）───────────────────────────────
        per_sheet_irs: list = []
        for lp, dlt in group.members:
            if dlt is None:
                continue
            table_ids = list(lp.detected_table_ids) + [dlt.proposal_id]
            col_vals: dict = {}
            for i, tid in enumerate(lp.detected_table_ids):
                col_vals[tid] = (
                    lp.detected_names[i] if i < len(lp.detected_names) else tid
                )
            col_vals[dlt.proposal_id] = dlt.derived_name

            col_name = _infer_col_name_from_values(list(col_vals.values()))
            rec_id = f"latent_auto_{gk}_{dlt.proposal_id}"

            # 1軸IRは2軸が「統合しない」に切り替えられた時に初めて表示される。
            # デフォルトを True にして、表示時に「統合する」が選択済みになるようにする。
            if rec_id not in auto_int_dec:
                auto_int_dec[rec_id] = True

            ir = IntegrationRecommendation(
                recommendation_id=rec_id,
                group_name=f"{'・'.join(group.missing_names)} 統合テーブル（{lp.source_title}）",
                description=(
                    f"差分推定した「{'・'.join(group.missing_names)}」を含む "
                    f"「{'・'.join(group.detected_names + group.missing_names)}」の統合テーブル"
                ),
                table_ids=table_ids,
                new_column_name=col_name,
                new_column_values={tid: v for tid, v in col_vals.items()},
                reasoning=(
                    f"潜在テーブル「{'・'.join(group.missing_names)}」の差分推定により、"
                    f"「{'・'.join(group.detected_names + group.missing_names)}」を"
                    f"1つの統合テーブルにまとめることができます。"
                ),
                new_column_names=[col_name],
                new_column_multi_values={tid: [v] for tid, v in col_vals.items()},
                # 親を設定することで、マスタテーブル生成がこのauto-IRに対して
                # 正しいマッピング（child値 → 親集計タイトル）を生成できるようにする。
                axis_parent_table_ids=[lp.source_table_id],
                axis_parent_label_columns=[None],
            )
            per_sheet_irs.append((ir, gk))

        # ── 優先度ルール: 2軸IRは常に表示し、「統合しない」の場合のみ1軸IRを追加 ──
        # _for_build=True（最終ビルド時）は決定に関わらず常に全IRを返し、
        # 呼び出し側がそれぞれの決定に基づいてフィルタリングする。
        if cross_ir is not None:
            result.append((cross_ir, gk))  # 常に2軸カードを表示
            if _for_build or not auto_int_dec.get(cross_rec_id, True):
                result.extend(
                    per_sheet_irs
                )  # ビルド時は常に追加、表示時は2軸「統合しない」の場合のみ
        else:
            result.extend(per_sheet_irs)  # 2軸なし → 常に1軸

    return result


def _clip_ai_irs_by_latent_groups(
    irs: list,
    latent_groups: List[LatentTableGroup],
    tables_dict: dict,
) -> list:
    """潜在グループの検証済み兄弟関係と一致しないテーブルをAI統合提案から除去する。

    2段階の刈り込みを行う：

    ステップ1（同一シート刈り込み）:
      潜在グループがD1, D2を兄弟（検証済み集計構成要素）と認識している場合、
      {D1, D2, ...extras}を含むAI IRのうち、extrasがD1/D2と同一シートにある
      場合は {D1, D2} のみに刈り込む。

    ステップ2（クロスシート刈り込み）:
      ステップ1で刈り込まれなかったIRが、兄弟テーブル以外のテーブルを含む場合、
      全グループ横断の兄弟集合（all_sib_tables）外のテーブルを除去する。
      これにより、複数シートにまたがる誤ったIR（例: 兄弟X_1+X_2に加えて
      別集計軸テーブルX_Dを含むクロスシートIR）も正しく刈り込まれる。
    """
    # 全潜在グループメンバーから兄弟セットを収集する
    sibling_sets: List[frozenset] = []
    for grp in latent_groups:
        for lp, _ in grp.members:
            if len(lp.detected_table_ids) >= 2:
                sibling_sets.append(frozenset(lp.detected_table_ids))

    if not sibling_sets:
        return irs

    # 全兄弟セットの和集合（クロスシート刈り込みに使用）
    all_sib_tables: set = set()
    for ss in sibling_sets:
        all_sib_tables.update(ss)

    def _make_trimmed(base_ir, keep_set):
        """keep_set のテーブルのみ残した IntegrationRecommendation を返す。"""
        new_ids = [t for t in base_ir.table_ids if t in keep_set]
        if len(new_ids) < 2:
            return None
        return IntegrationRecommendation(
            recommendation_id=base_ir.recommendation_id,
            group_name=base_ir.group_name,
            description=base_ir.description,
            table_ids=new_ids,
            new_column_name=base_ir.new_column_name,
            new_column_values={
                k: v for k, v in base_ir.new_column_values.items() if k in keep_set
            },
            reasoning=base_ir.reasoning,
            parent_table_id=base_ir.parent_table_id,
            parent_label_column=base_ir.parent_label_column,
            user_decision=base_ir.user_decision,
            new_column_names=base_ir.new_column_names,
            new_column_multi_values={
                k: v
                for k, v in base_ir.new_column_multi_values.items()
                if k in keep_set
            },
            axis_parent_table_ids=base_ir.axis_parent_table_ids,
            axis_parent_label_columns=base_ir.axis_parent_label_columns,
        )

    result = []
    for ir in irs:
        ir_set = frozenset(ir.table_ids)
        trimmed_ir = ir

        # ── ステップ1: 同一シート内の余分なテーブルを刈り込む ────────────────
        for sib_set in sibling_sets:
            if not (sib_set < ir_set and len(sib_set) >= 2):
                continue
            extras = ir_set - sib_set
            sib_sheets = {
                tables_dict[t].sheet_name for t in sib_set if t in tables_dict
            }
            extra_sheets = {
                tables_dict[t].sheet_name for t in extras if t in tables_dict
            }
            if not sib_sheets or not (extra_sheets <= sib_sheets):
                continue  # クロスsheetのextras — ステップ2で処理
            clipped = _make_trimmed(ir, sib_set)
            if clipped is not None:
                trimmed_ir = clipped
            break  # 最初に一致した兄弟セットを使用する

        # ── ステップ2: クロスシートIRから非兄弟テーブルを除去 ────────────────
        # ステップ1で刈り込まれていない場合のみ実行する。
        # IRが兄弟テーブルと非兄弟テーブルの両方を含む場合、
        # 全グループ横断の兄弟集合に含まれないテーブルを除去する。
        # （例: 複数シートにまたがる {X_1_A, X_2_A, X_D_A, X_1_B, X_2_B, X_D_B}
        #        → 非兄弟の X_D_A, X_D_B を除去して {X_1_A, X_2_A, X_1_B, X_2_B} にする）
        if trimmed_ir is ir:
            cur_set = frozenset(trimmed_ir.table_ids)
            in_sib = cur_set & all_sib_tables
            out_sib = cur_set - all_sib_tables
            if in_sib and out_sib:
                clipped = _make_trimmed(trimmed_ir, all_sib_tables)
                if clipped is not None:
                    trimmed_ir = clipped

        result.append(trimmed_ir)
    return result


def _render_latent_group_card(group: LatentTableGroup, tables_dict: dict) -> None:
    """潜在テーブルグループ（全sheet横断の提案 + 派生）の統合カードをレンダリングする。"""
    gk = group.group_key
    if "latent_group_decisions" not in st.session_state:
        st.session_state.latent_group_decisions = {}
    if gk not in st.session_state.latent_group_decisions:
        st.session_state.latent_group_decisions[gk] = True

    # st.radio() が呼ばれる前にradioウィジェットのキーを事前初期化する。
    # これにより、Streamlitの「index vs. key」競合による1rerun遅延を回避する:
    # `key=` と共に `index=` を使用すると、レンダリング時に古い latent_group_decisions
    # 値からindexが再計算されるため、ユーザーが変更した後もStreamlitがウィジェットを
    # index値にリセットしてしまう可能性がある。
    _radio_key = f"radio_latent_group_{gk}"
    if _radio_key not in st.session_state:
        st.session_state[_radio_key] = (
            "✅ 追加する"
            if st.session_state.latent_group_decisions.get(gk, True)
            else "❌ 追加しない"
        )

    _NOTE_TYPE_LABEL = {
        "aggregation": ("集計注記", "🧮"),
        "exclusion": ("除外注記", "➖"),
        "reference": ("参照注記", "🔗"),
        "general": ("注記", "📝"),
    }
    type_label, type_icon = _NOTE_TYPE_LABEL.get(group.note_type, ("注記", "📝"))
    missing_label = "・".join(group.missing_names)
    n_sheets = len(group.members)
    n_derived = sum(1 for _, dlt in group.members if dlt is not None)

    with st.container(border=True):
        sheet_badge = (
            f"<span style='font-size:0.72rem;color:#607090;margin-left:8px;'>"
            f"（{n_sheets} シート）</span>"
            if n_sheets > 1
            else ""
        )
        st.markdown(
            f"#### {type_icon} {missing_label}（差分推定）{sheet_badge}",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='background:#0d1520;border-left:3px solid #2a4060;"
            f"border-radius:0 6px 6px 0;padding:8px 14px;"
            f"font-size:0.82rem;color:#7090b0;margin:6px 0'>"
            f"{group.note_text}</div>",
            unsafe_allow_html=True,
        )

        # 代表メンバーの集計テーブル名（ソーステーブル = 注記の記載元 = 集計親）
        _rep_source_title = group.members[0][0].source_title if group.members else ""
        c_det, c_miss = st.columns(2)
        with c_det:
            st.markdown("**検出テーブル**")
            if _rep_source_title:
                st.markdown(f"📊 {_rep_source_title}（集計対象）")
            for name in group.detected_names:
                st.markdown(f"✅ {name}")
        with c_miss:
            st.markdown("**潜在テーブル候補**")
            for name in group.missing_names:
                st.markdown(
                    f"<span style='color:#e08080'>❓ {name}</span>",
                    unsafe_allow_html=True,
                )

        if n_derived > 0:
            derived_members = [
                (lp, dlt) for lp, dlt in group.members if dlt is not None
            ]
            # 代表: 最初のDLTを直接表示
            _render_derived_before_after(derived_members[0][1], tables_dict)
            # その他: 折りたたみexpander
            if len(derived_members) > 1:
                others = derived_members[1:]
                with st.expander(f"同様の推定 他 {len(others)} 件", expanded=False):
                    for lp, dlt in others:
                        st.markdown(f"**{lp.source_title}**")
                        _render_derived_before_after(dlt, tables_dict)
                        st.divider()

        st.divider()
        _splitter_marker(f"s4-lg-{gk}")
        c_info, c_dec = st.columns([2, 1])
        with c_info:
            st.markdown("**💡 推奨理由**")
            st.caption(group.members[0][0].reasoning)
        with c_dec:
            st.markdown("<br>", unsafe_allow_html=True)
            decision = st.radio(
                "この潜在テーブルを追加しますか？",
                ["✅ 追加する", "❌ 追加しない"],
                horizontal=True,
                key=_radio_key,
            )
            st.session_state.latent_group_decisions[gk] = decision == "✅ 追加する"
            if decision == "✅ 追加する" and n_derived > 0:
                st.caption("💡 下の統合テーブル案に自動反映されます")


def _render_unified_ir_card(
    ir: IntegrationRecommendation,
    tables_for_render: dict,
    is_auto: bool = False,
) -> None:
    """統合推奨カードをレンダリングする — AI提案と自動生成の両方に対応する。"""
    rec_id = ir.recommendation_id
    dec_store = "latent_auto_int_decisions" if is_auto else "integration_decisions"
    if dec_store not in st.session_state:
        st.session_state[dec_store] = {}
    if rec_id not in st.session_state[dec_store]:
        st.session_state[dec_store][rec_id] = True

    with st.container(border=True):
        st.markdown(f"#### {ir.group_name}")
        if is_auto:
            st.markdown(
                "<span style='display:inline-block;background:#1a3a1a;border:1px solid #2a6a2a;"
                "border-radius:12px;padding:2px 10px;font-size:0.72rem;color:#7aba7a;"
                "margin-bottom:8px;'>💡 潜在テーブル追加による自動提案</span>",
                unsafe_allow_html=True,
            )
        st.markdown(f"_{ir.description}_")

        _render_integration_before_after(ir, tables_for_render)

        st.divider()
        _splitter_marker(f"s4-ir-{rec_id}")
        c_info, c_dec = st.columns([2, 1])
        with c_info:
            _col_names = ir.new_column_names or [ir.new_column_name]
            _multi_vals = ir.new_column_multi_values or {}
            st.markdown(f"**追加列名**: {', '.join(f'`{n}`' for n in _col_names)}")
            for tid in ir.table_ids:
                vals = _multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
                val_str = " / ".join(str(v) for v in vals)
                if is_auto:
                    t_entry = tables_for_render.get(tid)
                    lbl = f"（{t_entry.title}）" if t_entry and t_entry.title else ""
                    st.markdown(f"  - `{tid}`{lbl} → **{val_str}**")
                else:
                    st.markdown(f"  - `{tid}` → **{val_str}**")
            st.caption(f"💡 推奨理由: {ir.reasoning}")
        with c_dec:
            st.markdown("<br>", unsafe_allow_html=True)
            _ir_radio_key = f"radio_{rec_id}"
            # index= を使うと 2クリック問題が起きるため、事前初期化 + index なし で対応する
            if _ir_radio_key not in st.session_state:
                st.session_state[_ir_radio_key] = (
                    "✅ 統合する"
                    if st.session_state[dec_store].get(rec_id, True)
                    else "❌ 統合しない"
                )
            decision = st.radio(
                "この統合を実施しますか？",
                ["✅ 統合する", "❌ 統合しない"],
                horizontal=True,
                key=_ir_radio_key,
            )
            st.session_state[dec_store][rec_id] = decision == "✅ 統合する"



# ---------- 共有ディスプレイヘルパー（step6_selectもimport） ----------

_COL_PALETTES = [
    ("#d0faf0", "#0d4b36", "#2aab87", "#ffffff"),  # teal
    ("#fdebd0", "#7a3a0a", "#d4793a", "#ffffff"),  # orange
    ("#d8e8fd", "#0d2f7f", "#4a7de0", "#ffffff"),  # blue
    ("#f0d5f8", "#5b0f7f", "#9b4fc4", "#ffffff"),  # purple
]

# 統合プレビュー用の軸ごとのカラーファミリー。
# 各ファミリーは8つの明確に区別できる色を持つ（シェードではなく個別の色）: (hdr_bg, hdr_fg, cell_bg, cell_fg)
# Axis 0 = WARM ファミリー（赤/オレンジ/ピンク/アンバー — 値ごとに視覚的に区別可能）
# Axis 1 = COOL ファミリー（青/ティール/紫/シアン — 値ごとに視覚的に区別可能）
# Axis 2 = NATURE ファミリー（緑/ライム/オリーブ/フォレスト）
# Axis 3 = ACCENT ファミリー（ディープオレンジ/インディゴ/ローズ/ミント）
_AXIS_FAMILIES: list = [
    [  # axis-0: WARM — 各値は明確に異なるウォームな色相
        ("#e74c3c", "#fff", "#fadbd8", "#7b0c0c"),  # red
        ("#e67e22", "#fff", "#fae5d3", "#7a3a0a"),  # orange
        ("#e91e63", "#fff", "#fce4ec", "#7c0024"),  # magenta/pink
        ("#f39c12", "#1a1a1a", "#fef9e7", "#5d3a00"),  # amber
        ("#c0392b", "#fff", "#f5b7b1", "#6e1006"),  # dark red
        ("#d35400", "#fff", "#fad5b0", "#6b2800"),  # burnt orange
        ("#ec407a", "#fff", "#fce8ef", "#7a0036"),  # rose
        ("#f57c00", "#fff", "#fff0d9", "#6b3800"),  # deep orange
    ],
    [  # axis-1: COOL — 各値は明確に異なるクールな色相
        ("#2980b9", "#fff", "#d6eaf8", "#0d2f6e"),  # blue
        ("#1abc9c", "#fff", "#d1f2eb", "#0a4038"),  # teal
        ("#8e44ad", "#fff", "#e8daef", "#4a1a72"),  # purple
        ("#00acc1", "#fff", "#e0f7fa", "#00474f"),  # cyan
        ("#3f51b5", "#fff", "#e8eaf6", "#1a237e"),  # indigo
        ("#16a085", "#fff", "#cde8e4", "#0a3630"),  # dark teal
        ("#6c3483", "#fff", "#e4d0ef", "#3b1260"),  # deep purple
        ("#0288d1", "#fff", "#e1f1fb", "#013d6e"),  # light blue
    ],
    [  # axis-2: NATURE — グリーン/ライム/フォレスト
        ("#27ae60", "#fff", "#d5f5e3", "#0b3c2e"),  # green
        ("#8bc34a", "#1a1a1a", "#f1f8e9", "#33691e"),  # lime green
        ("#00695c", "#fff", "#cce5e2", "#00332e"),  # forest
        ("#558b2f", "#fff", "#dcedc8", "#2a4200"),  # olive
        ("#2e7d32", "#fff", "#c8e6c9", "#0a2d0b"),  # dark green
        ("#76ff03", "#1a1a1a", "#f4ffe0", "#3a5f00"),  # neon lime
        ("#1b5e20", "#fff", "#c3e8c4", "#0a1c0a"),  # deep forest
        ("#aed581", "#1a1a1a", "#ecf6dc", "#3a5f00"),  # light olive
    ],
    [  # axis-3: ACCENT — 各種鮮やかな色
        ("#ff5722", "#fff", "#fbe9e7", "#bf360c"),  # deep orange
        ("#9c27b0", "#fff", "#f3e5f5", "#4a148c"),  # deep purple
        ("#009688", "#fff", "#e0f2f1", "#004d40"),  # teal accent
        ("#ffc107", "#1a1a1a", "#fff8e1", "#6b4900"),  # yellow accent
        ("#5c6bc0", "#fff", "#e8eaf6", "#1a237e"),  # indigo accent
        ("#ef5350", "#fff", "#ffebee", "#7b0000"),  # red accent
        ("#26a69a", "#fff", "#e0f2f1", "#00352f"),  # teal light
        ("#ab47bc", "#fff", "#f3e5f5", "#4a0072"),  # purple accent
    ],
]


def _render_integration_before_after(
    ir,
    tables_dict: dict,
    compact: bool = False,
    full_df_size: tuple = None,
    source_ids: list = None,
) -> None:
    """構造化された統合前 → 統合後のプレビューをレンダリングする。

    マルチ軸カラーリング: 各軸ディメンションには異なるカラーファミリー
    （緑/青/紫/オレンジ）を割り当てる。各ファミリー内では、ユニークな軸値ごとに
    異なる色を割り当てる。ソーステーブルカードは軸ごとに色付きのピルを表示し、
    統合テーブルは各軸列のセルを値によって色分けする。
    最初の3つのソーステーブルは直接表示され、追加テーブルはexpanderに収納する。

    full_df_size: 実際の統合済みテーブルの (n_rows, n_cols)。
                  統合後プレビューの右下に表示される。
    source_ids:   expander の後に表示するソーステーブルIDのリスト。
    """
    col_names = getattr(ir, "new_column_names", []) or [ir.new_column_name]
    multi_vals = getattr(ir, "new_column_multi_values", {}) or {}

    # インラインプレビューボックスの可視行の高さ。
    # (a) スクロールで全行を表示し、(b) ネイティブのFullscreenボタンで全レコードを
    # 表示できるよう、常に完全なDataFrameを渡す。
    _BEFORE_VISIBLE = 3  # 統合前カード: スクロール前に表示する行数
    _AFTER_VISIBLE = 10  # 統合後プレビュー: スクロール前に表示する行数
    _ROW_PX = 35  # st.dataframe の1データ行のおよそのpx高さ
    _HDR_PX = 38  # ヘッダー行の高さ

    # ── 冗長軸の検出（先頭で実行してソースカードのハイライトに利用）────────────
    redundant_axes_info = _detect_redundant_axes(col_names, multi_vals, ir, tables_dict)
    # 全軸の triggering col を集約（ソースカード共通ハイライト用）
    _all_trigger_cols: set = set()
    for _tc_set in redundant_axes_info.values():
        _all_trigger_cols |= _tc_set

    # ── 軸ごとのユニーク値の順序を構築する ──────────────────────────────────
    axis_val_order: list = [[] for _ in col_names]
    for tid in ir.table_ids:
        vals = multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
        for ai, v in enumerate(vals):
            if ai < len(axis_val_order):
                sv = str(v)
                if sv not in axis_val_order[ai]:
                    axis_val_order[ai].append(sv)

    def _axis_color(ai: int, val: str) -> tuple:
        """指定された軸+値に対する (hdr_bg, hdr_fg, cell_bg, cell_fg) を返す。"""
        family = _AXIS_FAMILIES[ai % len(_AXIS_FAMILIES)]
        vals_list = axis_val_order[ai] if ai < len(axis_val_order) else []
        vi = vals_list.index(val) if val in vals_list else 0
        return family[vi % len(family)]

    # ── ヘルパー: ソーステーブルカード（ヘッダー + DataFrame）を1つレンダリングする ──
    import html as _html

    def _src_card(tid: str) -> None:
        t = tables_dict.get(tid)
        vals = multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
        pills = "".join(
            f'<span style="background:{_axis_color(ai, str(v))[0]};'
            f"color:{_axis_color(ai, str(v))[1]};"
            f"padding:2px 9px;border-radius:12px;font-size:0.72rem;font-weight:600;"
            f'margin-right:4px;display:inline-block;margin-bottom:3px;">'
            f"{_html.escape(cn)}:&nbsp;{_html.escape(str(v))}</span>"
            for ai, (cn, v) in enumerate(zip(col_names, vals))
        )
        st.markdown(
            f'<div style="background:#161c2c;border:1px solid #2d3748;'
            f'border-radius:6px 6px 0 0;padding:7px 10px;margin-bottom:0;">'
            f'<div style="font-size:0.7rem;color:#7a8599;margin-bottom:4px;">'
            f"📋&nbsp;{_html.escape(tid)}</div>{pills}</div>",
            unsafe_allow_html=True,
        )
        if t is not None and t.effective_df is not None and not t.effective_df.empty:
            n_rows = len(t.effective_df)
            df_disp = t.effective_df.astype(str)
            # 冗長軸の判断理由となった既存カラムをそのテーブルの軸色でハイライト
            _trig_col_styles: dict = {}
            for _ci, _trig_set in redundant_axes_info.items():
                _t_vals = multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
                _aval = str(_t_vals[_ci] if _ci < len(_t_vals) else "")
                _, _, _cbg, _cfg = _axis_color(_ci, _aval)
                for _tc in _trig_set:
                    _trig_col_styles[_tc] = (
                        f"background-color:{_cbg};color:{_cfg};font-weight:600;"
                    )
            _valid_trig_map = {
                c: sty for c, sty in _trig_col_styles.items() if c in df_disp.columns
            }
            if _valid_trig_map:

                def _hl_trig(row, _vtc=_valid_trig_map):
                    s = pd.Series("", index=row.index)
                    for _c, _sty in _vtc.items():
                        s[_c] = _sty
                    return s

                df_disp = df_disp.style.apply(_hl_trig, axis=1)
            st.dataframe(
                df_disp,
                use_container_width=True,
                hide_index=True,
                height=min(
                    n_rows * _ROW_PX + _HDR_PX, _BEFORE_VISIBLE * _ROW_PX + _HDR_PX
                ),
            )
        else:
            st.caption("（データなし）")

    def _src_card_grid(tids: list) -> None:
        cols = st.columns(len(tids), gap="small")
        for i, tid in enumerate(tids):
            with cols[i]:
                _src_card(tid)

    # ════════════════════════════════════════════════════════════════════════
    # 統合前
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:6px 0 14px;">'
        '<div style="width:5px;height:22px;background:#4a7de0;border-radius:3px;flex-shrink:0;"></div>'
        '<span style="font-size:1.05rem;font-weight:800;color:#c8d4e8;letter-spacing:.04em;">統合前</span>'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,rgba(74,125,224,.4),transparent);"></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    SAMPLE_LIMIT = 3
    preview_tids = ir.table_ids[:SAMPLE_LIMIT]
    extra_tids = ir.table_ids[SAMPLE_LIMIT:]

    if preview_tids:
        _src_card_grid(preview_tids)

    if extra_tids:
        with st.expander(f"他 {len(extra_tids)} テーブルを見る", expanded=False):
            n_ex = min(len(extra_tids), 3)
            for start in range(0, len(extra_tids), n_ex):
                _src_card_grid(extra_tids[start : start + n_ex])

    # ── 統合元テーブル一覧 (expander下) ─────────────────────────────────────
    # expander内のst.columns()は避ける — 同じ縦方向コンテナ内のネストした columns は
    # 上部でレンダリングされたソーステーブルグリッドの幅が不均等になる可能性がある。
    if source_ids:
        _INLINE_LIMIT = 4
        if len(source_ids) <= _INLINE_LIMIT:
            st.caption(f"🔗 統合元テーブル一覧: {' ／ '.join(source_ids)}")
        else:
            with st.expander(
                f"🔗 統合元テーブル一覧 （{len(source_ids)} テーブル）", expanded=False
            ):
                for sid in source_ids:
                    st.caption(f"• {sid}")

    # ── 統合処理セパレーター ─────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0;margin:22px 0 18px;">'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,transparent,rgba(39,174,96,.5));"></div>'
        '<div style="border:1.5px solid rgba(39,174,96,.7);border-radius:24px;'
        "padding:6px 22px;margin:0 16px;font-size:1.05rem;font-weight:800;"
        "color:#7FFFD4;letter-spacing:.08em;"
        "background:linear-gradient(135deg,rgba(39,174,96,.12),rgba(26,188,156,.08));"
        'display:flex;align-items:center;gap:8px;white-space:nowrap;">'
        "↓&nbsp;&nbsp;統合処理"
        "</div>"
        '<div style="flex:1;height:1px;background:linear-gradient(to left,transparent,rgba(39,174,96,.5));"></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════════════
    # 統合後
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:6px 0 14px;">'
        '<div style="width:5px;height:22px;background:#27ae60;border-radius:3px;flex-shrink:0;"></div>'
        '<span style="font-size:1.05rem;font-weight:800;color:#c8d4e8;letter-spacing:.04em;">統合後</span>'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,rgba(39,174,96,.4),transparent);"></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # redundant_axes_info は関数先頭で計算済み（ソースカードハイライトにも使用）
    redundant_preview = redundant_axes_info

    # 冗長軸が存在する場合は注釈ボックスを表示
    if redundant_preview:
        _suppressed_notes = []
        for _ci, _tc_set in sorted(redundant_preview.items()):
            if _ci < len(col_names):
                _trig_str = "・".join(sorted(_tc_set)) if _tc_set else "既存列"
                _suppressed_notes.append(
                    f"「{col_names[_ci]}」軸は既存の「{_trig_str}」列から導出可能なため追加を省略しました"
                )
        if _suppressed_notes:
            st.markdown(
                '<div style="background:rgba(255,180,0,0.1);border-left:3px solid #ffd369;'
                "border-radius:0 4px 4px 0;padding:7px 12px;margin:0 0 12px;"
                'font-size:0.81rem;color:#ffd369;">'
                + "ℹ️ "
                + "。".join(_suppressed_notes)
                + "。</div>",
                unsafe_allow_html=True,
            )

    frames = []
    for tid in ir.table_ids:
        t = tables_dict.get(tid)
        if t is not None and t.effective_df is not None and not t.effective_df.empty:
            row = (
                t.effective_df.copy()
            )  # full data — scroll + Fullscreen reveal all rows
            vals = multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
            for ci in range(len(col_names) - 1, -1, -1):
                if ci in redundant_preview:
                    continue
                val = vals[ci] if ci < len(vals) else ""
                row.insert(0, col_names[ci], val)
            frames.append(row)

    if not frames:
        st.caption("（プレビュー生成不可）")
        return

    try:
        combined = pd.concat(frames, ignore_index=True)
        df_str = combined.astype(str)
        valid_cols = [c for c in col_names if c in df_str.columns]

        # 各軸セルをそのカラーファミリー内の値によって色付けする
        def _color_row(row):
            s = pd.Series("", index=row.index)
            for ai, c in enumerate(valid_cols):
                if c in row.index:
                    _, _, cbg, cfg = _axis_color(ai, str(row[c]))
                    s[c] = f"background-color:{cbg};color:{cfg};"
            return s

        styler = df_str.style.apply(_color_row, axis=1) if valid_cols else df_str.style

        # 軸列ヘッダー: 各軸ファミリーのベース（shade 0）ヘッダー色を使用する
        def _hdr_fn(idx):
            out = []
            for c in idx:
                if c in valid_cols:
                    ai = valid_cols.index(c)
                    hbg, hfg = _AXIS_FAMILIES[ai % len(_AXIS_FAMILIES)][0][:2]
                    out.append(f"background-color:{hbg};color:{hfg};font-weight:bold;")
                else:
                    out.append("")
            return out

        try:
            styler = styler.apply_index(_hdr_fn, axis="columns")
        except Exception:
            pass

        tbl_styles = []
        for c in valid_cols:
            try:
                ci = list(df_str.columns).index(c)
                ai = valid_cols.index(c)
                hbg, hfg = _AXIS_FAMILIES[ai % len(_AXIS_FAMILIES)][0][:2]
                tbl_styles.append(
                    {
                        "selector": f"th.col_heading.col{ci}",
                        "props": (
                            f"background-color:{hbg} !important;"
                            f"color:{hfg} !important;font-weight:bold !important;"
                        ),
                    }
                )
            except ValueError:
                pass
        if tbl_styles:
            try:
                styler = styler.set_table_styles(tbl_styles, overwrite=False)
            except Exception:
                pass

        # 軸が省略された場合、判断理由となった既存列（trigger_cols）を各テーブルの軸色でハイライト
        if _all_trigger_cols:
            _after_trig = [c for c in _all_trigger_cols if c in df_str.columns]
            if _after_trig:
                # ソーステーブルの軸値ごとに trigger 列の値→軸色マッピングを構築する
                _trig_val_color: dict = {}  # (col_name, cell_value) → CSS style string
                for _ci, _trig_set in redundant_axes_info.items():
                    for _tid in ir.table_ids:
                        _t = tables_dict.get(_tid)
                        if _t is None or _t.effective_df is None:
                            continue
                        _t_vals = multi_vals.get(_tid) or [
                            ir.new_column_values.get(_tid, "")
                        ]
                        _aval = str(_t_vals[_ci] if _ci < len(_t_vals) else "")
                        _, _, _cbg, _cfg = _axis_color(_ci, _aval)
                        _sty = f"background-color:{_cbg};color:{_cfg};font-weight:600;"
                        for _tc in _trig_set:
                            if _tc in _t.effective_df.columns:
                                for _v in _t.effective_df[_tc].astype(str).unique():
                                    _trig_val_color[(_tc, _v)] = _sty

                _fallback_sty = "background-color:#3d2e00;color:#ffd369;font-weight:600;"

                def _hl_after_trig(
                    row, _vtc=_after_trig, _vmap=_trig_val_color, _fb=_fallback_sty
                ):
                    s = pd.Series("", index=row.index)
                    for _c in _vtc:
                        if _c not in row.index:
                            continue
                        _val = str(row[_c])
                        s[_c] = _vmap.get((_c, _val), _fb)
                    return s

                try:
                    styler = styler.apply(_hl_after_trig, axis=1)
                except Exception:
                    pass
                # ヘッダー — 最初の suppressed 軸の先頭色 (shade 0) を使用
                try:
                    _prev_tbl_styles = tbl_styles[:]
                    _first_ci = next(iter(sorted(redundant_axes_info.keys())), None)
                    if _first_ci is not None:
                        _hdr_bg, _hdr_fg = _AXIS_FAMILIES[
                            _first_ci % len(_AXIS_FAMILIES)
                        ][0][:2]
                    else:
                        _hdr_bg, _hdr_fg = "#7d5a00", "#ffd369"
                    for _ac in _after_trig:
                        try:
                            _aci = list(df_str.columns).index(_ac)
                            _prev_tbl_styles.append(
                                {
                                    "selector": f"th.col_heading.col{_aci}",
                                    "props": (
                                        f"background-color:{_hdr_bg} !important;"
                                        f"color:{_hdr_fg} !important;"
                                        "font-weight:bold !important;"
                                    ),
                                }
                            )
                        except ValueError:
                            pass
                    styler = styler.set_table_styles(_prev_tbl_styles, overwrite=True)
                except Exception:
                    pass

        n_data = len(df_str)
        st.dataframe(
            styler,
            use_container_width=True,
            hide_index=True,
            height=min(n_data * _ROW_PX + _HDR_PX, _AFTER_VISIBLE * _ROW_PX + _HDR_PX),
        )
        # 統合済みテーブルのサイズ — プレビューの右下に表示する
        if full_df_size:
            n_rows, n_cols_full = full_df_size
            st.markdown(
                f'<div style="text-align:right;font-size:0.72rem;'
                f'color:#7a8599;margin-top:2px;">'
                f"📊 {n_rows:,} 行 × {n_cols_full} 列</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.caption("（プレビュー生成不可）")



def _render_derived_before_after(
    dlt: "DerivedLatentTable",
    tables_dict: dict,
) -> None:
    """派生潜在テーブルの推定前/推定後ビューをレンダリングする。

    「推定前」は集計親テーブルと各検出済み構成要素を表示する。
    「推定後」は数値的に派生（差し引き）されたテーブルを表示する。
    """
    _ROW_PX = 35
    _HDR_PX = 38
    _BEFORE_VISIBLE = 3
    _AFTER_VISIBLE = 4

    import html as _html

    # ── 推定前 ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:6px 0 14px;">'
        '<div style="width:5px;height:22px;background:#4a7de0;border-radius:3px;flex-shrink:0;"></div>'
        '<span style="font-size:1.05rem;font-weight:800;color:#c8d4e8;letter-spacing:.04em;">推定前（検出テーブル）</span>'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,rgba(74,125,224,.4),transparent);"></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    def _table_card_mini(tid: str, role_label: str, color: str) -> None:
        t = tables_dict.get(tid)
        title = (t.title if t else None) or tid
        st.markdown(
            f'<div style="background:#161c2c;border:1px solid #2d3748;'
            f'border-radius:6px 6px 0 0;padding:7px 10px;margin-bottom:0;">'
            f'<div style="font-size:0.7rem;color:#7a8599;margin-bottom:4px;">📋&nbsp;{_html.escape(tid)}</div>'
            f'<span style="background:{color};color:#fff;padding:2px 9px;border-radius:12px;'
            f'font-size:0.72rem;font-weight:600;">{_html.escape(role_label)}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if t is not None and t.effective_df is not None and not t.effective_df.empty:
            n = len(t.effective_df)
            st.dataframe(
                t.effective_df.astype(str),
                use_container_width=True,
                hide_index=True,
                height=min(n * _ROW_PX + _HDR_PX, _BEFORE_VISIBLE * _ROW_PX + _HDR_PX),
            )
        else:
            st.caption("（データなし）")

    # 親（集計）+ 子をcolumnsに表示する
    all_display = dlt.source_display_order  # [parent_id, child1_id, ...]
    n_disp = min(len(all_display), 4)
    cols = st.columns(n_disp, gap="small")
    for i, tid in enumerate(all_display[:n_disp]):
        with cols[i]:
            if tid == dlt.parent_table_id:
                _table_card_mini(tid, f"集計テーブル：{dlt.parent_title}", "#2a607a")
            else:
                _child_t = tables_dict.get(tid)
                _child_title = (_child_t.title if _child_t else None) or tid
                _table_card_mini(tid, f"集計元テーブル：{_child_title}", "#4a5a3a")
    if len(all_display) > n_disp:
        with st.expander(
            f"他 {len(all_display) - n_disp} テーブルを見る", expanded=False
        ):
            for tid in all_display[n_disp:]:
                if tid == dlt.parent_table_id:
                    _table_card_mini(
                        tid, f"集計テーブル：{dlt.parent_title}", "#2a607a"
                    )
                else:
                    _child_t = tables_dict.get(tid)
                    _child_title = (_child_t.title if _child_t else None) or tid
                    _table_card_mini(tid, f"集計元テーブル：{_child_title}", "#4a5a3a")

    # ── 差分計算セパレーター ─────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0;margin:18px 0 14px;">'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,transparent,rgba(229,168,60,.5));"></div>'
        '<div style="border:1.5px solid rgba(229,168,60,.7);border-radius:24px;'
        "padding:6px 22px;margin:0 16px;font-size:0.95rem;font-weight:800;"
        "color:#f5d06a;letter-spacing:.06em;"
        "background:linear-gradient(135deg,rgba(229,168,60,.12),rgba(200,140,40,.08));"
        'white-space:nowrap;">↓&nbsp;&nbsp;差分計算</div>'
        '<div style="flex:1;height:1px;background:linear-gradient(to left,transparent,rgba(229,168,60,.5));"></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#1a1508;border-left:3px solid rgba(229,168,60,.6);'
        f"border-radius:0 6px 6px 0;padding:6px 14px;font-size:0.82rem;"
        f'color:#c8a030;font-family:monospace;margin-bottom:14px;">'
        f"{_html.escape(dlt.derivation_formula)}</div>",
        unsafe_allow_html=True,
    )

    # ── 推定後 ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:6px 0 14px;">'
        '<div style="width:5px;height:22px;background:#e5a83c;border-radius:3px;flex-shrink:0;"></div>'
        '<span style="font-size:1.05rem;font-weight:800;color:#c8d4e8;letter-spacing:.04em;">推定後（潜在テーブル）</span>'
        '<div style="flex:1;height:1px;background:linear-gradient(to right,rgba(229,168,60,.4),transparent);"></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if dlt.df is not None and not dlt.df.empty:
        n = len(dlt.df)
        st.dataframe(
            dlt.df.astype(str),
            use_container_width=True,
            hide_index=True,
            height=min(n * _ROW_PX + _HDR_PX, _AFTER_VISIBLE * _ROW_PX + _HDR_PX),
        )
        st.caption(
            f"📐 {len(dlt.df)} 行 × {len(dlt.df.columns)} 列  "
            f"（{dlt.parent_title} から {', '.join(dlt.detected_child_ids)} を差し引いた推定値）"
        )
    else:
        st.caption("（推定データなし）")



# ---------- 最終テーブルリスト構築 ----------

def _build_final_tables():
    analysis: AIAnalysisResult = st.session_state.ai_analysis
    tables_dict = {t.table_id: t for t in st.session_state.detected_tables}
    ta_by_id = {ta.table_id: ta for ta in analysis.table_analyses}

    final: Dict[str, dict] = {}
    integrated_ids: Set[str] = set()
    seen_master_sigs: Set = (
        set()
    )  # 同一の child→parent ラベルマップを持つマスタを重複排除する

    # ── 潜在グループの検出（1回のみ）────────────────────────────────────────
    _bft_proposals = find_latent_tables(st.session_state.detected_tables)
    _bft_derived = derive_latent_tables(st.session_state.detected_tables)
    _bft_groups = group_latent_proposals(_bft_proposals, _bft_derived)
    _bft_ext = dict(tables_dict)
    for _dlt in _bft_derived:
        _vt = _dlt_virtual_table(_dlt, tables_dict)
        if _vt:
            _bft_ext[_dlt.proposal_id] = _vt

    # ── 承認済みauto-IRによって置き換えられるAI IRを計算する（step4と同じロジック）──
    _bft_ai_irs_trimmed = _clip_ai_irs_by_latent_groups(
        analysis.integration_recommendations, _bft_groups, tables_dict
    )
    _bft_auto_irs = _build_auto_irs_from_latent(_bft_groups, _bft_ext, tables_dict)
    _superseded_in_bft: set = set()
    for _bft_auto_ir, _ in _bft_auto_irs:
        if not st.session_state.get("latent_auto_int_decisions", {}).get(
            _bft_auto_ir.recommendation_id, True
        ):
            continue
        _auto_real = {t for t in _bft_auto_ir.table_ids if t in tables_dict}
        if len(_auto_real) < 2:
            continue
        for _ai_check in _bft_ai_irs_trimmed:
            # AI IRの全テーブルがauto-IRの実テーブルに含まれる場合は抑制する
            if set(_ai_check.table_ids).issubset(_auto_real):
                _superseded_in_bft.add(_ai_check.recommendation_id)

    _lg_dec = st.session_state.get("latent_group_decisions", {})
    _lai_dec = st.session_state.get("latent_auto_int_decisions", {})

    # ── 潜在グループ: DLT + auto-IR統合（step4と同じ順序で先に処理）────────
    for _grp in _bft_groups:
        _gk = _grp.group_key
        # グループ決定: 新しいキーを優先し、DLTごとの derived_decisions にフォールバック
        if _gk in _lg_dec:
            _grp_accepted = _lg_dec[_gk]
        else:
            _grp_accepted = any(
                st.session_state.get("derived_decisions", {}).get(dlt.proposal_id, True)
                for _, dlt in _grp.members
                if dlt is not None
            )
        if not _grp_accepted:
            continue

        for _lp, _dlt in _grp.members:
            if _dlt is None:
                continue
            # DLTは統合の統合元として使用する中間テーブルのため、
            # 単体では最小粒度データではなく非推奨テーブルとして扱う。
            # クロスシート集計除外で auto-IR の table_ids に含まれないDLTも
            # 同様に非推奨とする。
            final[_dlt.proposal_id] = {
                "df": _dlt.df,
                "display_name": _dlt.derived_name,
                "description": _dlt.derivation_formula,
                "reasoning": _dlt.reasoning,
                "is_integrated": False,
                "source_ids": _dlt.source_display_order,
                "recommended": False,
                "granularity": "detail",
                "is_minimum": False,
                "is_master": False,
            }

        # 承認済みauto-IR統合テーブル
        _auto_irs_bft = _build_auto_irs_from_latent([_grp], _bft_ext, tables_dict)
        for _auto_ir, _ in _auto_irs_bft:
            _rec_id = _auto_ir.recommendation_id
            if not _lai_dec.get(_rec_id, True):
                continue

            _col_names_bft = _auto_ir.new_column_names or [_auto_ir.new_column_name]
            _multi_vals_bft = _auto_ir.new_column_multi_values or {}
            _redundant_bft = _detect_redundant_axes(
                _col_names_bft,
                _multi_vals_bft,
                _auto_ir,
                {t.table_id: t for t in st.session_state.get("detected_tables", [])},
            )

            _frames_bft = []
            for _tid in _auto_ir.table_ids:
                _t = _bft_ext.get(_tid)
                if _t and _t.effective_df is not None and not _t.effective_df.empty:
                    _df_copy = _t.effective_df.copy()
                    _vals = _multi_vals_bft.get(_tid) or [
                        _auto_ir.new_column_values.get(_tid, "")
                    ]
                    for _ci in range(len(_col_names_bft) - 1, -1, -1):
                        if _ci in _redundant_bft:
                            continue
                        _val = _vals[_ci] if _ci < len(_vals) else ""
                        _df_copy.insert(0, _col_names_bft[_ci], _val)
                    _frames_bft.append(_df_copy)

            if not _frames_bft:
                continue
            try:
                _merged_bft = pd.concat(_frames_bft, ignore_index=True)
            except Exception:
                continue

            # 統合元テーブルを「統合済み」として扱う
            # （「統合する」選択時は最小粒度データとして個別表示しない）
            for _tid in _auto_ir.table_ids:
                if _tid in tables_dict:
                    integrated_ids.add(_tid)
                elif _tid in final:
                    # DLT（仮想テーブル）は tables_dict にないため integrated_ids で管理できない。
                    # 代わりに final エントリを非推奨・非最小粒度に降格させ、
                    # 非推奨テーブルの折りたたみセクションに移動させる。
                    final[_tid]["is_minimum"] = False
                    final[_tid]["recommended"] = False

            _int_key = f"latent_auto_int_{_rec_id}"
            _bft_reasoning = _auto_ir.reasoning
            if _redundant_bft:
                # 軸を追加しなかった場合は推奨理由の文言を動的に変換する
                for _bci in _redundant_bft:
                    if _bci < len(_col_names_bft):
                        _anm = _col_names_bft[_bci]
                        _bft_reasoning = _bft_reasoning.replace(
                            f"{_anm}軸を追加することで", f"{_anm}軸で統合することで"
                        ).replace(
                            f"{_anm}を追加することで", f"{_anm}で統合することで"
                        )
                _bft_supp = []
                for _bci, _btc in sorted(_redundant_bft.items()):
                    if _bci < len(_col_names_bft):
                        _btrig = "・".join(sorted(_btc)) if _btc else "既存列"
                        _bft_supp.append(
                            f"「{_col_names_bft[_bci]}」軸は既存の「{_btrig}」列から導出可能なため追加を省略しました"
                        )
                if _bft_supp:
                    _bft_reasoning = _bft_reasoning + "。" + "。".join(_bft_supp) + "。"
            _bft_trigger_cols: list = sorted(
                set(c for tc in _redundant_bft.values() for c in tc)
            )
            final[_int_key] = {
                "df": _merged_bft,
                "display_name": _auto_ir.group_name,
                "description": _auto_ir.description,
                "reasoning": _bft_reasoning,
                "is_integrated": True,
                "source_ids": _auto_ir.table_ids,
                "recommended": True,
                "granularity": "detail",
                "is_minimum": True,
                "is_master": False,
                "new_col_names": [
                    c for i, c in enumerate(_col_names_bft) if i not in _redundant_bft
                ],
                "trigger_col_names": _bft_trigger_cols,
            }

            # auto-IRに対してもマスタを生成する（AI IRと同じロジック）
            for _spec in _derive_master_specs_for_ir(_auto_ir, tables_dict):
                _master_sig = _master_signature(_spec)
                if _master_sig in seen_master_sigs:
                    continue
                seen_master_sigs.add(_master_sig)
                _axis_idx = _spec.get("axis_idx", 0)
                _dm_key = f"dim_master_{_rec_id}_ax{_axis_idx}"
                if not st.session_state.master_decisions.get(_dm_key, True):
                    continue
                _child_col = _spec["child_col"]
                _parent_col = _spec["parent_col"]
                _master_rows = [
                    {_child_col: ck, _parent_col: pv}
                    for ck, pv in _spec["mapping"].items()
                ]
                if _master_rows:
                    _master_df = pd.DataFrame(_master_rows)
                    final[_dm_key] = {
                        "df": _master_df,
                        "display_name": f"{_child_col} × {_parent_col} マスタ",
                        "description": (
                            f"元データの上位集計テーブル（{_spec['parent_id']}）と各子テーブルの"
                            f"「{_child_col}」値の対応関係から生成したマスタテーブル。"
                            f"統合テーブルを「{_child_col}」で結合すると「{_parent_col}」単位での再集計が可能になる。"
                        ),
                        "reasoning": (
                            f"元データ内の上位集計テーブル {_spec['parent_id']} の階層関係から自動生成。"
                            f"統合テーブル自体ではなく、ソースデータの親子関係をもとにした対応表。"
                        ),
                        "is_integrated": False,
                        "source_ids": [_spec["parent_id"]] + _auto_ir.table_ids,
                        "recommended": True,
                        "granularity": "master",
                        "is_minimum": False,
                        "is_master": True,
                    }

    # ── AI IR統合（抑制・不承認のものはスキップ）────────────────────────────
    for ir in _bft_ai_irs_trimmed:
        if ir.recommendation_id in _superseded_in_bft:
            continue  # 派生テーブルを含むauto-IRによって置き換えられた
        if not st.session_state.integration_decisions.get(ir.recommendation_id, True):
            continue

        col_names = getattr(ir, "new_column_names", []) or [ir.new_column_name]
        multi_vals = getattr(ir, "new_column_multi_values", {}) or {}
        redundant_axes = _detect_redundant_axes(col_names, multi_vals, ir, tables_dict)

        frames = []
        for tid in ir.table_ids:
            t = tables_dict.get(tid)
            if t and t.effective_df is not None and not t.effective_df.empty:
                df_copy = t.effective_df.copy()
                vals = multi_vals.get(tid) or [ir.new_column_values.get(tid, "")]
                for i in range(len(col_names) - 1, -1, -1):
                    if i in redundant_axes:
                        continue
                    val = vals[i] if i < len(vals) else ""
                    df_copy.insert(0, col_names[i], val)
                frames.append(df_copy)
                integrated_ids.add(tid)

        if not frames:
            continue
        try:
            merged_df = pd.concat(frames, ignore_index=True)
        except Exception:
            for tid in ir.table_ids:
                integrated_ids.discard(tid)
            continue

        src_ta = next((ta_by_id[tid] for tid in ir.table_ids if tid in ta_by_id), None)
        key = f"integrated_{ir.recommendation_id}"
        _ir_reasoning = ir.reasoning
        if redundant_axes:
            # 軸を追加しなかった場合は推奨理由の文言を動的に変換する
            for _rci in redundant_axes:
                if _rci < len(col_names):
                    _anm = col_names[_rci]
                    _ir_reasoning = _ir_reasoning.replace(
                        f"{_anm}軸を追加することで", f"{_anm}軸で統合することで"
                    ).replace(
                        f"{_anm}を追加することで", f"{_anm}で統合することで"
                    )
            _ir_supp = []
            for _ici, _itc in sorted(redundant_axes.items()):
                if _ici < len(col_names):
                    _itrig = "・".join(sorted(_itc)) if _itc else "既存列"
                    _ir_supp.append(
                        f"「{col_names[_ici]}」軸は既存の「{_itrig}」列から導出可能なため追加を省略しました"
                    )
            if _ir_supp:
                _ir_reasoning = _ir_reasoning + "。" + "。".join(_ir_supp) + "。"
        _ir_trigger_cols: list = sorted(
            set(c for tc in redundant_axes.values() for c in tc)
        )
        final[key] = {
            "df": merged_df,
            "display_name": ir.group_name,
            "description": ir.description,
            "reasoning": _ir_reasoning,
            "is_integrated": True,
            "source_ids": ir.table_ids,
            "recommended": True,
            "granularity": src_ta.granularity_level if src_ta else "detail",
            "is_minimum": src_ta.is_minimum_granularity_candidate if src_ta else False,
            "is_master": src_ta.is_master_table if src_ta else False,
            "new_col_names": [
                c for i, c in enumerate(col_names) if i not in redundant_axes
            ],
            "trigger_col_names": _ir_trigger_cols,
        }

        # 親を持つ全軸に対してディメンションマスタを自動生成する。
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

    # 個別の非統合テーブル
    for ta in analysis.table_analyses:
        if ta.table_id in integrated_ids:
            continue
        t = tables_dict.get(ta.table_id)
        if not t or t.effective_df is None or t.effective_df.empty:
            continue
        final[ta.table_id] = {
            "df": t.effective_df,
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

    # セーフティネット — 分析対象外のテーブル
    analyzed_ids = set(ta_by_id.keys())
    for t in st.session_state.detected_tables:
        if t.table_id in analyzed_ids or t.table_id in integrated_ids:
            continue
        if t.effective_df is None or t.effective_df.empty:
            continue
        final[t.table_id] = {
            "df": t.effective_df,
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

    # 単位混在分離（Step3）で生成された指標マスタを、LLM分析を介さず
    # 機械的に「マスタ」として登録する（ディメンションマスタと同じ強制分類パターン）。
    for t in st.session_state.detected_tables:
        if t.unit_master_df is None or t.unit_master_df.empty:
            continue
        um_key = f"{t.table_id}_unit_master"
        if um_key in final:
            continue
        src_ta = ta_by_id.get(t.table_id)
        src_final = final.get(t.table_id)
        src_name = (
            (src_ta.display_name if src_ta else None)
            or (src_final.get("display_name") if src_final else None)
            or t.title
            or t.table_id
        )
        label_col = (t.unit_split_info or {}).get("label_col", "指標")
        final[um_key] = {
            "df": t.unit_master_df,
            "display_name": f"{src_name} 指標マスタ",
            "description": (
                f"「{src_name}」の {label_col} 列に混在していた単位を分離して"
                f"生成した指標マスタ（{label_col}・単位の対応表）。"
            ),
            "reasoning": "テーブル整形（Step3 単位混在の分離）で自動生成されたマスタテーブルです。",
            "is_integrated": False,
            "source_ids": [t.table_id],
            "recommended": True,
            "granularity": "master",
            "is_minimum": False,
            "is_master": True,
        }

    st.session_state.final_tables = final
    # 推奨テーブルを事前選択する
    st.session_state.selected_ids = {
        tid for tid, info in final.items() if info["recommended"]
    }


# ---------------------------------------------------------------------------
# Step 5 — テーブル選択
# ---------------------------------------------------------------------------

# 各新規列にはそれぞれ異なるパレットを割り当てる: (cell_bg, cell_fg, hdr_bg, hdr_fg)
# hdr_bgは視覚的なコントラストのためcell_bgより明らかに暗い色を使用する。
