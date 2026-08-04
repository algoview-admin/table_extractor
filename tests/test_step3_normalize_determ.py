"""Step3 決定論的パイプラインのテスト（C1〜C14、LLM不使用）。

対応する実装チェックリスト（src/step3_normalize_determ.py）:
  C1.  多段ヘッダー単純統合（merge_header_rows）
  C2.  Pivot検出と変換
  C3.  多段ヘッダー軸展開（detect_multi_axis_header、構造検出のみ）
  C4.  カラム名内階層区切りの分離
  C5.  括弧書き注釈の分離（検出のみ。命名はD3でLLMモック使用）
  C6.  階層圧縮カラムの展開（インデント方式・区切り文字方式）
  C7.  ロールアップ行の除去と階層整合性検証
  C8.  グルーピング列の前方補完
  C9.  「うち」書き識別と別テーブル分離
  C10. 集計行・集計列の除去 + 階層整合性検証
  C11. 単位混在の分離
  C12. 無効カラムの検出
  C13. Wide_to_long検出（Tier1時系列語彙、Tier2区切り文字）
  C14. クロス集計形式の検出
"""

from conftest import load_fixture_tables, tables_by_sheet
import src.step3_normalize_determ as det


def _table(filename, sheet=None):
    tables = load_fixture_tables(filename)
    if sheet is None:
        return tables[0]
    return next(t for t in tables if t.sheet_name == sheet)


# --- C1: 多段ヘッダー単純統合 ------------------------------------------------


def test_merge_header_rows_name_unit_pair():
    t = _table("step2_structure_test.xlsx", "ヘッダー種別")
    tables = [x for x in load_fixture_tables("step2_structure_test.xlsx") if x.sheet_name == "ヘッダー種別"]
    name_unit = next(x for x in tables if x.raw_header_roles == ["name", "unit"])
    cols, used = det.merge_header_rows(name_unit.raw_header_rows, name_unit.raw_header_roles, 2)
    assert cols == ["支店", "従業者数 [人]"]
    assert used == {0, 1}


def test_merge_header_rows_merged_span_type():
    tables = [x for x in load_fixture_tables("step2_structure_test.xlsx") if x.sheet_name == "ヘッダー種別"]
    merged = next(x for x in tables if x.raw_header_roles == ["name", "name"])
    cols, used = det.merge_header_rows(merged.raw_header_rows, merged.raw_header_roles, 5)
    assert cols == ["商品", "東京_売上", "東京_原価", "大阪_売上", "大阪_原価"]


# --- C2: Pivot検出と変換 -----------------------------------------------------


def test_detect_and_apply_pivot_kv():
    t = _table("step3_pivot_test.xlsx")
    info = det.detect_pivot_kv(t.df)
    assert info is not None
    assert info["attributes"] == ["売上", "利益"]
    out = det.apply_pivot_kv(t.df, info)
    assert list(out.columns) == ["支店", "売上", "利益"]
    assert out.loc[out["支店"] == "東京", "売上"].iloc[0] == 100
    assert out.loc[out["支店"] == "大阪", "利益"].iloc[0] == 15


# --- 変換規模の事前予測機能: evaluate_pivot_scale / rank・limit_pivot_attributes --


def test_evaluate_pivot_scale_small_no_warning():
    info = {"key_cols": ["支店"], "attributes": ["売上", "利益"]}
    scale = det.evaluate_pivot_scale(info)
    assert scale == {
        "n_attrs": 2, "output_columns": 3,
        "exceeds_excel_limit": False, "needs_warning": False,
    }


def test_evaluate_pivot_scale_performance_warning_boundary():
    # 閾値ちょうどは警告しない、閾値+1は警告する（境界値）
    at_threshold = {"key_cols": [], "attributes": [f"a{i}" for i in range(det.PIVOT_PERFORMANCE_WARNING_THRESHOLD)]}
    assert det.evaluate_pivot_scale(at_threshold)["needs_warning"] is False

    over_threshold = {"key_cols": [], "attributes": [f"a{i}" for i in range(det.PIVOT_PERFORMANCE_WARNING_THRESHOLD + 1)]}
    scale = det.evaluate_pivot_scale(over_threshold)
    assert scale["needs_warning"] is True
    assert scale["exceeds_excel_limit"] is False


def test_evaluate_pivot_scale_exceeds_excel_limit():
    info = {"key_cols": [], "attributes": [f"a{i}" for i in range(det.PIVOT_EXCEL_MAX_COLUMNS + 1)]}
    scale = det.evaluate_pivot_scale(info)
    assert scale["exceeds_excel_limit"] is True
    assert scale["needs_warning"] is True


def test_rank_pivot_attributes_by_frequency():
    import pandas as pd
    df = pd.DataFrame({"attr": ["A", "B", "A", "C", "A", "B"], "value": [1, 2, 3, 4, 5, 6]})
    info = {"attr_col": "attr", "attributes": ["A", "B", "C"]}
    ranked = det.rank_pivot_attributes_by_frequency(df, info)
    assert ranked == [("A", 3), ("B", 2), ("C", 1)]


def test_limit_pivot_attributes_preserves_original_order():
    info = {"attributes": ["A", "B", "C", "D"]}
    ranked = [("B", 5), ("D", 3), ("A", 2), ("C", 1)]
    limited = det.limit_pivot_attributes(info, ranked, top_n=2)
    # 頻度上位2件はB,Dだが、列順は元のattributes初出順（A,B,C,D）の中で
    # B,Dのみ残した順序になる（頻度順に並べ替えないことを確認）
    assert limited["attributes"] == ["B", "D"]


def test_normalize_tables_blocks_large_pivot(monkeypatch):
    # 変換規模の事前予測機能: PIVOT_PERFORMANCE_WARNING_THRESHOLDを超えるPivotは
    # normalize_tables()内で自動適用されず、以降のStep3処理（うち分離〜
    # クロス集計検出）が完全に保留されることをend-to-endで検証する。
    # このテーブルはブロックされるため実際のLLM呼び出しは発生しない
    # （client未設定を許容するのはこの理由のため、テスト時はmake_transpose_client
    # 自体をモックしてAPI認証情報を不要にする）。
    monkeypatch.setattr(det, "make_transpose_client", lambda: (None, None))

    tables = tables_by_sheet("step3_pivot_scale_test.xlsx")
    t = tables["PivotScale"]
    all_tables = [t]
    det.normalize_tables(all_tables, "step3_pivot_scale_test.xlsx")

    assert t.pivot_scale_warning is not None
    assert t.pivot_scale_warning["n_attrs"] == 201
    assert t.pivot_scale_warning["needs_warning"] is True
    assert t.pivot_decision is None
    assert list(t.df.columns) == ["拠点", "属性", "値"]
    assert len(t.df) == 402
    assert t.uchi_split_info is None
    assert t.invalid_col_candidates is None
    assert t.wide_to_long_info is None
    assert t.stack_info is None
    assert len(all_tables) == 1  # うち分離等が走っていないため派生テーブルも増えない


# --- C3: 多段ヘッダー軸展開（構造検出のみ） ----------------------------------


def test_detect_multi_axis_header_structural():
    t = _table("step3_multiaxis_header_test.xlsx")
    result = det.detect_multi_axis_header(t.raw_header_rows, t.raw_header_roles)
    assert result is not None
    assert result["kinds"] == ["candidate", "candidate"]
    assert result["values"] == [["売上", "原価"], ["東京", "大阪"]]
    assert result["candidate_idxs"] == [0, 1]


def test_apply_multi_axis_header_with_hand_built_axis_result():
    t = _table("step3_multiaxis_header_test.xlsx")
    info = det.detect_multi_axis_header(t.raw_header_rows, t.raw_header_roles)
    # LLMが確定させる axis_result を手構築してテストする（D-groupの
    # detect_dimension_axes 自体はLLMモックで別途テスト）
    axis_result = {
        "axis_names": ["科目", "支店"],
        "value_name": "値",
        "indicator_axis_index": 0,
    }
    out = det.apply_multi_axis_header(t.df, info, axis_result)
    assert "支店" in out.columns
    assert "売上" in out.columns and "原価" in out.columns
    # ラベル列（年）は両ヘッダー行で空白のため列名は "列"（プレースホルダ）になる
    row = out[out["支店"] == "東京"]
    assert set(row["列"]) == {"2024年", "2025年"}


# --- C4: カラム名内階層区切りの分離 -------------------------------------------


def test_detect_column_hierarchy_split():
    t = _table("step3_column_hierarchy_split_test.csv")
    result = det.detect_column_hierarchy_split(t.df)
    assert result is not None
    names = [c["name"] for c in result["columns"]]
    assert "帳票プラン1 / 帳票プラン2" in names
    # T/O比（数値列、区切り文字を含まない）は候補にならない
    assert "T/O比" not in names


def test_apply_column_hierarchy_split():
    t = _table("step3_column_hierarchy_split_test.csv")
    result = det.detect_column_hierarchy_split(t.df)
    out = det.apply_column_hierarchy_split(t.df, result["columns"])
    assert "帳票プラン1" in out.columns and "帳票プラン2" in out.columns
    assert out.loc[0, "帳票プラン1"] == "フレッツ光"
    assert out.loc[0, "帳票プラン2"] == "おまかせLAN構築"
    assert out.loc[0, "T/O比"] == 12.5


# --- C5: 括弧書き注釈の分離（検出のみ） --------------------------------------


def test_detect_paren_annotations():
    t = _table("step3_paren_annotation_test.csv")
    result = det.detect_paren_annotations(t.df)
    assert result is not None
    cols = {c["source_col"]: c for c in result["columns"]}
    assert "支店" in cols
    assert set(cols["支店"]["distinct_annotations"]) == {"本社", "支店"}
    # 「人口」列の(人)(％)は単位語彙のため対象外
    assert "人口" not in cols


def test_apply_paren_annotations_with_hand_named_column():
    t = _table("step3_paren_annotation_test.csv")
    result = det.detect_paren_annotations(t.df)
    named = [{**c, "new_col": "拠点区分"} for c in result["columns"] if c["source_col"] == "支店"]
    out = det.apply_paren_annotations(t.df, named)
    assert "拠点区分" in out.columns
    assert out.loc[0, "支店"] == "東京"
    assert out.loc[0, "拠点区分"] == "本社"


# --- C6/C7: 階層圧縮カラムの展開・ロールアップ行除去・整合性検証 -------------


def test_detect_hierarchy_expansion_indent_mode():
    t = _table("step3_hierarchy_expand_test.xlsx", "インデント方式")
    result = det.detect_hierarchy_expansion(t.df)
    assert result["mode"] == "indent"
    assert result["depth"] == 3


def test_detect_hierarchy_expansion_delimiter_mode():
    t = _table("step3_hierarchy_expand_test.xlsx", "区切り文字方式")
    result = det.detect_hierarchy_expansion(t.df)
    assert result["mode"] == "delimiter"
    assert result["delimiter"] == "＞"


def test_detect_hierarchy_expansion_cell_format_indent_mode():
    # Step1のセル書式インデント→先頭空白変換(A2)を経由した列でも
    # インデント方式として検出できること
    t = _table("step3_hierarchy_expand_test.xlsx", "セル書式インデント方式")
    result = det.detect_hierarchy_expansion(t.df)
    assert result["mode"] == "indent"
    assert result["depth"] == 3


def test_expand_hierarchy_column_and_rollup_removal():
    t = _table("step3_hierarchy_expand_test.xlsx", "インデント方式")
    detection = det.detect_hierarchy_expansion(t.df)
    expanded = det.expand_hierarchy_column(t.df, detection, ["地域", "事業部", "支店"])
    assert list(expanded.columns) == ["地域", "事業部", "支店", "売上"]
    rollup_idx, metadata, integrity = det.find_hierarchy_rollup_rows(expanded, detection)
    # 東日本(1600=1100+500)・神奈川事業部(1100=700+400)はロールアップ行として検出される
    final = expanded.drop(index=rollup_idx).reset_index(drop=True)
    assert len(final) == 3  # 横浜支店・川崎支店・千葉事業部の3行のみ残る
    leaves = set(zip(final["事業部"], final["支店"]))
    assert leaves == {("神奈川事業部", "横浜支店"), ("神奈川事業部", "川崎支店"), ("千葉事業部", "")}


# --- C8: グルーピング列の前方補完 --------------------------------------------


def test_fill_grouping_cols():
    t = _table("step3_fill_uchi_aggregate_test.xlsx", "前方補完と集計除去")
    filled_df, filled_cols = det.fill_grouping_cols(t.df)
    assert "支店" in filled_cols
    assert filled_df["支店"].tolist()[:3] == ["東京", "東京", "東京"]


# --- C9: 「うち」書き識別と別テーブル分離 ------------------------------------


def test_detect_and_apply_uchi_breakdown():
    t = _table("step3_fill_uchi_aggregate_test.xlsx", "うち分離")
    filled_df, _ = det.fill_grouping_cols(t.df)
    info = det.detect_uchi_breakdown(filled_df)
    assert info is not None
    assert info["label_col"] == "区分"
    assert info["match_count"] == 4

    main_df, breakdown_df, protected, integrity = det.apply_uchi_split(filled_df, info)
    # 内訳行(うち男性・うち女性)は本体から消え、内訳テーブルへ分離される
    assert "うち男性" not in main_df["区分"].tolist()
    assert "うち女性" not in main_df["区分"].tolist()
    assert breakdown_df.shape[0] == 4
    assert set(breakdown_df["子区分"]) == {"男性", "女性"}


# --- C10: 集計行・集計列の除去 + 階層整合性検証 -------------------------------


def test_remove_aggregates_removes_exact_keyword_rows():
    t = _table("step3_fill_uchi_aggregate_test.xlsx", "前方補完と集計除去")
    filled_df, _ = det.fill_grouping_cols(t.df)
    cleaned, rows_info, cols_removed, row_idx, row_meta, col_meta, integrity = det.remove_aggregates(filled_df)
    assert "合計" not in cleaned["部門"].tolist()
    assert len(rows_info) == 3  # 東京・大阪・福岡の3つの合計行


def test_remove_aggregates_integrity_check_pass_and_fail_causes():
    t = _table("step3_fill_uchi_aggregate_test.xlsx", "前方補完と集計除去")
    filled_df, _ = det.fill_grouping_cols(t.df)
    _, _, _, _, _, _, integrity = det.remove_aggregates(filled_df)
    by_branch = {r["支店"]: r for r in integrity}
    assert by_branch["東京"]["status"] == "PASS"
    assert by_branch["大阪"]["status"] == "FAIL"
    assert by_branch["大阪"]["cause"] == "二重計上または単位ミス"
    assert by_branch["福岡"]["status"] == "FAIL"
    assert by_branch["福岡"]["cause"] == "内訳あり（未表示項目）"


# --- C11: 単位混在の分離 -----------------------------------------------------


def test_detect_and_split_units():
    t = _table("step3_unit_split_test.xlsx")
    result = det.detect_and_split_units(t.df)
    assert result is not None
    assert result["mapping"] == {"人口": "人", "労働力率": "％"}
    assert result["cleaned_df"]["指標"].tolist()[0] == "人口"
    assert list(result["master_df"].columns) == ["指標", "単位"]


# --- C12: 無効カラムの検出 ---------------------------------------------------


def test_detect_invalid_columns():
    # detect_invalid_columns はStep3の純粋関数（既に確定したDataFrameの
    # 列を評価するだけ）のため、Step2の生グリッド検出を経由せず直接構築した
    # DataFrameで検証する。「無名＋全欠損」列は列名・データのどちらにも
    # 手掛かりが一切無く、複数テーブルの1列ギャップ区切り（本セッションで
    # gap_threshold=1 に変更）と raw grid 上で本質的に区別が付かないため、
    # Step2の検出結果に依存させると意図せず別テーブルとして分離されうる。
    import pandas as pd

    df = pd.DataFrame({
        "支店": ["東京", "大阪"],
        "支店_1": [1001, 1002],
        "値": [None, None],
        "列": [None, None],
        "Unnamed: 3": ["参考A", None],
        "件数": [100, 200],
    })
    result = det.detect_invalid_columns(df)
    assert result is not None
    by_name = {c["name"]: c for c in result["columns"]}
    assert by_name["値"]["reason"] == "全欠損"
    assert by_name["値"]["default_selected"] is True
    assert by_name["列"]["reason"] == "無名＋全欠損"
    assert by_name["Unnamed: 3"]["reason"] == "無名（データあり）"
    assert by_name["Unnamed: 3"]["default_selected"] is False


def test_invalid_columns_fixture_end_to_end():
    # フィクスチャ自体（Step2経由）でも、「全欠損だが列名がある」列と
    # 「無名だがデータがある」列が1つのテーブルとして正しく検出できる
    # ことを確認する（「無名＋全欠損」列は上のテストで個別に検証済み）。
    t = _table("step3_invalid_columns_test.xlsx")
    result = det.detect_invalid_columns(t.df)
    assert result is not None
    by_name = {c["name"]: c for c in result["columns"]}
    assert by_name["値"]["reason"] == "全欠損"
    assert by_name["Unnamed: 3"]["reason"] == "無名（データあり）"


# --- C13: Wide_to_long検出（Tier1・Tier2） -----------------------------------


def test_wide_to_long_tier1_time_vocabulary():
    t = _table("step3_wide_to_long_test.xlsx", "時系列語彙")
    result = det.detect_wide_to_long(t.df, t.title, "step3_wide_to_long_test.xlsx")
    assert result is not None
    assert set(result["axis_tokens"]) == {"2023", "2024"}
    assert set(result["indicators"]) == {"売上", "原価"}


def test_wide_to_long_tier2_delimiter():
    t = _table("step3_wide_to_long_test.xlsx", "区切り文字")
    result = det.detect_wide_to_long(t.df, t.title, "step3_wide_to_long_test.xlsx")
    assert result is not None
    assert len(result["axis_tokens"]) == 5
    assert set(result["indicators"]) == {"売上", "原価"}


def test_wide_to_long_tier3_requires_client():
    # client未指定の場合、Tier3（LLM要）は検出されない
    t = _table("step3_wide_to_long_test.xlsx", "LLM要")
    result = det.detect_wide_to_long(t.df, t.title, "step3_wide_to_long_test.xlsx")
    assert result is None


def test_wide_to_long_tier3_deterministic_prefilter_finds_candidates():
    # LLM確認より前の、決定論的な反復セグメント検出自体は動作すること
    t = _table("step3_wide_to_long_test.xlsx", "LLM要")
    candidate_cols = [c for c in map(str, t.df.columns) if det._classify_col_time(c) is None]
    concat = det._find_concatenated_axis_candidates(candidate_cols)
    assert concat is not None
    assert set(v[0] for v in concat.values()) == {"ゴールド", "シルバー", "ブロンズ"}


# --- C14: クロス集計形式の検出 -----------------------------------------------


def test_detect_cross_table():
    t = _table("step3_crosstab_test.xlsx")
    result = det.detect_cross_table(t.df, t.title, "step3_crosstab_test.xlsx")
    assert result is not None
    assert result["label_cols"] == ["支店"]
    assert result["time_cols"] == ["2023年", "2024年", "2025年"]
    assert result["var_name"] == "年"


def test_stack_cross_table():
    t = _table("step3_crosstab_test.xlsx")
    info = det.detect_cross_table(t.df, t.title, "step3_crosstab_test.xlsx")
    out = det.stack_cross_table(t.df, info)
    assert list(out.columns) == ["支店", "年", "値"]
    assert out.shape[0] == 6  # 2支店 × 3年


# --- 列名生成規則機能: resolve_axis_var_name / resolve_value_col_name ---------


def test_resolve_axis_var_name_dictionary_source():
    name, source, reason = det.resolve_axis_var_name(time_kind="year")
    assert name == "年"
    assert source == "dictionary"
    assert reason == ""


def test_resolve_axis_var_name_computed_source():
    name, source, reason = det.resolve_axis_var_name(
        tokens=["東京支社", "大阪支社", "名古屋支社"]
    )
    assert name == "支社"
    assert source == "computed"


def test_resolve_axis_var_name_fallback_with_time_kind():
    # time_kindが辞書に存在しない場合は"期間"（時系列文脈の既定名）
    name, source, reason = det.resolve_axis_var_name(time_kind="__unknown_kind__")
    assert name == det._VAR_NAME_FALLBACK
    assert source == "fallback"
    assert reason


def test_resolve_axis_var_name_fallback_without_time_kind():
    # 共通接辞も取れない非時系列トークンは"区分"（汎用カテゴリの既定名）
    name, source, reason = det.resolve_axis_var_name(tokens=["ABC", "XYZ"])
    assert name == det._AXIS_GENERIC_VAR_NAME
    assert source == "fallback"


def test_resolve_value_col_name_dictionary_source():
    name, source, reason = det.resolve_value_col_name(title="2024年度 売上実績")
    assert name == "売上"
    assert source == "dictionary"
    assert reason == ""


def test_resolve_value_col_name_fallback_without_client():
    name, source, reason = det.resolve_value_col_name(title="意味の無いタイトル")
    assert name == "値"
    assert source == "fallback"
    assert reason


def test_detect_cross_table_records_naming_provenance():
    t = _table("step3_crosstab_test.xlsx")
    result = det.detect_cross_table(t.df, t.title, "step3_crosstab_test.xlsx")
    assert result["var_name_source"] == "dictionary"
    assert result["value_name_source"] in ("dictionary", "fallback")


def test_wide_to_long_tier1_records_naming_provenance():
    t = _table("step3_wide_to_long_test.xlsx", "時系列語彙")
    result = det.detect_wide_to_long(t.df, t.title, "step3_wide_to_long_test.xlsx")
    assert result["axis_var_name_source"] == "dictionary"


def test_wide_to_long_tier2_records_naming_provenance():
    t = _table("step3_wide_to_long_test.xlsx", "区切り文字")
    result = det.detect_wide_to_long(t.df, t.title, "step3_wide_to_long_test.xlsx")
    assert result["axis_var_name_source"] in ("computed", "fallback")
