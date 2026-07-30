"""Step2 テーブル検出機能のテスト（B1〜B5）。

対応する実装チェックリスト:
  B1. タイトル行・ヘッダー行数/役割判定
  B2. 同一シート内の複数テーブル検出（横並び・縦並び）
  B3. 表以外データ（注釈・メモ）の除外／末尾注釈の分離
  B4. 低品質領域の除外（誤discardの解消を含む）
  B5. 重複カラム名の一意化、無名列判定
  B6. 1列ギャップの横並び・1行だけのテーブル・幅広テーブル中の全文字列行
      混在（tests/README.md で洗い出した4つの罠のうち3つを実装で解消した
      ことの回帰テスト。詳細はREADMEおよびsrc/step2_detect.pyの該当コメント参照）
"""

from conftest import load_fixture_tables
from src.step3_normalize_determ import _is_placeholder_col_name


def _tables_in(sheet_name):
    return [t for t in load_fixture_tables("step2_structure_test.xlsx") if t.sheet_name == sheet_name]


# --- B1: タイトル行・ヘッダー行数/役割判定 -----------------------------------


def test_title_row_detected_and_excluded_from_data():
    tables = _tables_in("ヘッダー種別")
    single_header = next(t for t in tables if t.raw_header_rows is None and list(t.df.columns) == ["支店", "売上"])
    # タイトル行「2024年度 支店別売上実績」がデータ行に混入していないこと
    assert "2024年度 支店別売上実績" not in single_header.df["支店"].tolist()
    assert single_header.df.shape[0] == 2


def test_two_row_header_name_unit_pair_roles():
    tables = _tables_in("ヘッダー種別")
    name_unit_table = next(t for t in tables if t.raw_header_roles == ["name", "unit"])
    assert name_unit_table.raw_header_rows == [["支店", "従業者数"], [None, "人"]]


def test_two_row_header_merged_span_roles():
    tables = _tables_in("ヘッダー種別")
    merged_table = next(t for t in tables if t.raw_header_roles == ["name", "name"])
    assert merged_table.raw_header_rows[1] == [None, "売上", "原価", "売上", "原価"]


# --- B2: 同一シート内の複数テーブル検出 -------------------------------------


def test_side_by_side_tables_detected_as_separate():
    tables = _tables_in("複数テーブル配置")
    cols_seen = {tuple(t.df.columns) for t in tables}
    assert ("支店", "売上") in cols_seen
    assert ("地域", "人口") in cols_seen


def test_vertically_stacked_table_detected():
    tables = _tables_in("複数テーブル配置")
    assert any(tuple(t.df.columns) == ("部門", "人数") for t in tables)


# --- B3: 表以外データの除外／末尾注釈の分離 ----------------------------------


def test_leading_note_attached_as_trailing_note_of_preceding_table():
    tables = _tables_in("複数テーブル配置")
    side_by_side = [t for t in tables if tuple(t.df.columns) in (("支店", "売上"), ("地域", "人口"))]
    assert all("※本データは概算値です" in t.notes for t in side_by_side)


def test_trailing_note_attached_to_table():
    tables = _tables_in("複数テーブル配置")
    dept_table = next(t for t in tables if tuple(t.df.columns) == ("部門", "人数"))
    assert "出典：自社調査" in dept_table.notes


# --- B4: 低品質領域の除外（および誤discardの解消） ---------------------------


def test_short_all_text_table_is_recognized():
    # 数値を一切含まない幅広の短いテキストブロック（項目A〜E）は、
    # 以前は形状だけで discard されていたが、短い文字列の正当な表は
    # 除外されるべきではない（回帰確認）
    tables = _tables_in("複数テーブル配置")
    matched = next((t for t in tables if "項目A" in list(t.df.columns)), None)
    assert matched is not None
    assert matched.df.shape == (2, 5)


def test_long_prose_block_still_excluded():
    # 数値をほぼ含まず長文セルが多いプロースブロックは、非表形式データとして
    # 引き続き除外される（_classify_table_quality の長文チェック）
    tables = _tables_in("複数テーブル配置")
    assert not any("説明1" in list(t.df.columns) for t in tables)
    assert len(tables) == 4


# --- B5: 重複カラム名の一意化、無名列判定 -----------------------------------


def test_duplicate_column_names_deduplicated():
    t = load_fixture_tables("step3_invalid_columns_test.xlsx")[0]
    cols = list(t.df.columns)
    assert "支店" in cols
    assert "支店_1" in cols


def test_placeholder_column_name_detection():
    assert _is_placeholder_col_name("Unnamed: 3")
    assert _is_placeholder_col_name("Unnamed:3")
    assert _is_placeholder_col_name("列")
    assert _is_placeholder_col_name("列3")
    assert _is_placeholder_col_name("")
    assert not _is_placeholder_col_name("支店")


# --- B6: 1列ギャップ横並び・1行テーブル・幅広テーブル中の全文字列行混在 -----


def _tables_in_new_layout():
    return _tables_in("新規対応レイアウト")


def test_single_column_gap_side_by_side_tables_detected_as_separate():
    # gap_threshold を2から1に緩和したことで、1列だけ空けた横並びの
    # 2テーブルも別々のテーブルとして検出される
    tables = _tables_in_new_layout()
    cols_seen = {tuple(t.df.columns) for t in tables}
    assert ("支店", "売上") in cols_seen
    assert ("地域", "人口") in cols_seen
    # 誤って1つのテーブルに結合されていないこと
    combined = next((t for t in tables if "売上" in t.df.columns and "人口" in t.df.columns), None)
    assert combined is None


def test_single_row_table_is_recognized():
    # dense_rows の最低要件を2から1に緩和したことで、データ行が1行だけの
    # テーブルも discard されずに検出される
    tables = _tables_in_new_layout()
    matched = next((t for t in tables if tuple(t.df.columns) == ("商品", "在庫")), None)
    assert matched is not None
    assert matched.df.shape == (1, 2)
    assert matched.df.iloc[0]["商品"] == "家電"


def test_wide_table_not_fragmented_by_all_text_row():
    # 幅広（5列）テーブルの中に全列が文字列の行（名古屋の行）が混在しても、
    # 表が分断されず1つのテーブルとして検出され、その行もデータとして残る
    tables = _tables_in_new_layout()
    matched = next(
        (t for t in tables if tuple(t.df.columns) == ("支店", "値1", "値2", "値3", "値4")),
        None,
    )
    assert matched is not None
    assert matched.df.shape == (5, 5)
    assert "名古屋" in matched.df["支店"].tolist()
    row = matched.df[matched.df["支店"] == "名古屋"].iloc[0]
    assert row["値1"] == "テキスト1"
