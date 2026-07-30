"""Step4 決定論的事前検証のテスト（E1〜E2、LLM不使用の純粋関数のみ）。

対応する実装チェックリスト:
  E1. テーブル間合計関係の検証（detect_sum_relations）
  E2. シート間集計構造の検出（detect_sheet_levels）
"""

from conftest import load_fixture_tables
import src.step4_analyze as s4


# --- E1: テーブル間合計関係の検証 --------------------------------------------


def test_detect_sum_relations_parent_child():
    tables = load_fixture_tables("step4_sum_relations_test.xlsx")
    relations = s4.detect_sum_relations(tables)
    assert len(relations) == 1
    rel = relations[0]
    parent_table = next(t for t in tables if t.table_id == rel["parent_id"])
    assert parent_table.sheet_name == "全社計"
    child_sheets = {next(t for t in tables if t.table_id == cid).sheet_name for cid in rel["child_ids"]}
    assert child_sheets == {"東京", "大阪", "名古屋"}


# --- E2: シート間集計構造の検出 ----------------------------------------------


def test_detect_sheet_levels_scale_only_not_flagged():
    # 系統1: AP数・プラン数・ID数はスケールが違うだけで合計関係がないため、
    # いずれも集計シート候補として検出されてはならない（誤検出防止の回帰）
    tables = load_fixture_tables("step4_sheet_levels_test.xlsx")
    detail_tables = [t for t in tables if t.sheet_name in ("AP数", "プラン数", "ID数")]
    hints = s4.detect_sheet_levels(detail_tables)
    assert hints == []


def test_detect_sheet_levels_true_aggregate_detected():
    # 系統2: 全社計2 = 東京2+大阪2+名古屋2 という真の合計関係は検出される
    tables = load_fixture_tables("step4_sheet_levels_test.xlsx")
    agg_tables = [t for t in tables if t.sheet_name in ("全社計2", "東京2", "大阪2", "名古屋2")]
    hints = s4.detect_sheet_levels(agg_tables)
    assert len(hints) == 1
    assert hints[0]["aggregate_sheet"] == "全社計2"
    assert hints[0]["source_sheets"] == sorted(["東京2", "大阪2", "名古屋2"])


def test_detect_sheet_levels_mixed_groups_isolated():
    # 系統1・系統2は列スキーマが異なるため別グループとして扱われ、
    # 系統1側の誤検出が系統2の検出結果に影響しないこと（回帰確認）
    tables = load_fixture_tables("step4_sheet_levels_test.xlsx")
    hints = s4.detect_sheet_levels(tables)
    aggregate_sheets = {h["aggregate_sheet"] for h in hints}
    assert "ID数" not in aggregate_sheets
    assert "全社計2" in aggregate_sheets
