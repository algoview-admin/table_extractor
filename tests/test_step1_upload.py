"""Step1 パース機能のテスト（A1〜A3）。

対応する実装チェックリスト:
  A1. Excel結合セルの値伝播
  A2. Excelセル書式インデント→先頭空白変換（バグ回帰）
  A3. CSV型復元・先頭空白保持（.rstrip() バグ回帰）
"""

from conftest import load_fixture_tables, tables_by_sheet
from src.step1_upload import _parse_csv_cell


# --- A3: CSV型復元・先頭空白保持 -------------------------------------------


def test_csv_cell_type_coercion():
    assert _parse_csv_cell(" 123") == 123
    assert _parse_csv_cell("123 ") == 123
    assert _parse_csv_cell("12.5") == 12.5
    assert _parse_csv_cell("") is None
    assert _parse_csv_cell("   ") is None


def test_csv_cell_preserves_leading_whitespace():
    # .strip() ではなく .rstrip() であること（先頭空白は階層検出のシグナル）
    assert _parse_csv_cell("  abc") == "  abc"
    assert _parse_csv_cell("abc  ") == "abc"


def test_csv_parsing_fixture_end_to_end():
    t = load_fixture_tables("step1_csv_parsing_test.csv")[0]
    df = t.df
    assert list(df.columns) == ["支店", "売上", "備考"]
    # 数値文字列は int/float に復元されている
    assert df["売上"].tolist() == [1000, 800, 600.5]
    # 先頭空白を持つセルは保持される
    assert df["支店"].tolist()[2] == " 名古屋 " or df["支店"].tolist()[2].startswith(" ")


# --- A1: Excel結合セルの値伝播 -----------------------------------------------


def test_merged_cell_value_propagation():
    t = tables_by_sheet("step2_structure_test.xlsx")["ヘッダー種別"]
    # ヘッダー種別シートの2番目のテーブル（結合セルスパン型ヘッダー）は
    # T1 ではなく検出順で2番目になる。シート内の全テーブルを見て探す。
    tables = [x for x in load_fixture_tables("step2_structure_test.xlsx") if x.sheet_name == "ヘッダー種別"]
    merged_table = next(x for x in tables if x.raw_header_rows and len(x.raw_header_rows) == 2 and x.raw_header_roles == ["name", "name"])
    # 結合セル「東京」「大阪」がそれぞれ2列に複製されていること
    assert merged_table.raw_header_rows[0].count("東京") == 2
    assert merged_table.raw_header_rows[0].count("大阪") == 2


# --- A2: Excelセル書式インデント→先頭空白変換（バグ回帰） -------------------


def test_excel_cell_indent_converted_to_leading_space():
    t = load_fixture_tables("step1_excel_cell_indent_test.xlsx")[0]
    values = t.df["地域"].tolist()
    # インデント0（東日本相当の関東地方等）は空白なし
    assert values[0] == "関東地方"
    # インデント1・2は半角スペース2つ×段数が先頭に付与されている
    assert values[1] == "  東京都"
    assert values[2] == "    新宿区"
    assert values[3] == "    渋谷区"


def test_excel_cell_indent_does_not_affect_plain_columns():
    t = load_fixture_tables("step1_excel_cell_indent_test.xlsx")[0]
    # インデント書式を設定していない列は影響を受けない（回帰確認）
    values = t.df["備考"].tolist()
    assert all(not str(v).startswith(" ") for v in values if v)
