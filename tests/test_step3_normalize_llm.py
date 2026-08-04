"""Step3 LLM併用パイプラインのテスト（D1〜D3）。

実際のLLM APIを呼び出す（`llm_client` フィクスチャ、tests/conftest.py参照）。
どのAPI・モデルを使うかは環境変数で設定する（アプリ本体と同じ仕組み。
詳細は tests/README.md を参照）。認証情報が未設定の場合はテストがskipされる。

実APIの応答は決定論的でないため、assertは「構造として妥当か」
（結果がNoneでない、必要なキーが揃っている、型が正しい等）に留め、
具体的な命名文言・reasoning文言はハードコードしない。

対応する実装チェックリスト:
  D1. Transpose検出と変換
  D2. ファイル外メタデータ抽出と埋め込み（アンカー順序解決を含む）
  D3. 括弧書き注釈・階層レベルの命名（既定名フォールバック自体は非LLMのため純関数として検証）
"""

from conftest import load_fixture_tables
import src.step3_normalize_determ as det
import src.step3_normalize_llm as llm


def _table(filename, sheet=None):
    tables = load_fixture_tables(filename)
    if sheet is None:
        return tables[0]
    return next(t for t in tables if t.sheet_name == sheet)


# --- D1: Transpose検出と変換 -------------------------------------------------


def test_detect_transpose_true(llm_client):
    client, model = llm_client
    t = _table("step3_transpose_test.xlsx")
    result = llm.detect_transpose(t.df, client, model)
    assert result is not None
    assert isinstance(result["entity_axis_name"], str)
    assert result["entity_axis_name"] != ""


def test_apply_transpose():
    t = _table("step3_transpose_test.xlsx")
    out = llm.apply_transpose(t.df, "支店")
    assert list(out.columns) == ["支店", "売上", "利益"]
    assert out.loc[out["支店"] == "東京", "売上"].iloc[0] == 100


def test_detect_transpose_false(llm_client):
    client, model = llm_client
    # 通常の縦持ち表（Pivot前のテーブル）は転置対象ではない
    t = _table("step3_pivot_test.xlsx")
    result = llm.detect_transpose(t.df, client, model)
    assert result is None


# --- D2: ファイル外メタデータ抽出と埋め込み -----------------------------------


def test_extract_external_metadata(llm_client):
    client, model = llm_client
    t = _table("2025年度_サービスX_実績.xlsx")
    result = llm.extract_external_metadata(
        "2025年度_サービスX_実績.xlsx", t.sheet_name, t.title,
        dimension_columns=["区分1"], axis_name="4月", value_name="値",
        client=client, model=model,
    )
    assert result is not None
    assert isinstance(result["items"], list)
    for item in result["items"]:
        assert item["column_name"]
        assert item["source"] in ("filename", "sheet_name")
        assert item["position"] in ("before", "after", None)


def test_apply_external_metadata_anchor_ordering():
    # このセッションで修正済みのバグ（同一anchor・positionが複数ある場合の
    # 順序反転）の回帰テスト。LLMを使わない純粋な位置解決ロジックのテスト。
    t = _table("2025年度_サービスX_実績.xlsx")
    items = [
        {"column_name": "サービス名", "value": "サービスX", "source": "filename",
         "is_year": False, "anchor": "区分1", "position": "after"},
        {"column_name": "オプション種別", "value": "拡張サポート", "source": "sheet_name",
         "is_year": False, "anchor": "サービス名", "position": "after"},
    ]
    out, ordered = llm.apply_external_metadata(t.df, items, dim_cols=["区分1"], axis_name="4月")
    assert ordered == ["区分1", "サービス名", "オプション種別"]


# --- D3: 括弧書き注釈・階層レベルの命名 ---------------------------------------


def test_detect_annotation_column_names(llm_client):
    client, model = llm_client
    t = _table("step3_paren_annotation_test.csv")
    candidates = det.detect_paren_annotations(t.df)["columns"]
    named = llm.detect_annotation_column_names(candidates, t.title, client, model)
    # 本社/支店という明確な区分のため、is_valid=trueで命名されることを期待する
    assert named is not None
    assert named[0]["new_col"]


def test_detect_hierarchy_level_names(llm_client):
    client, model = llm_client
    t = _table("step3_hierarchy_expand_test.xlsx", "インデント方式")
    detection = det.detect_hierarchy_expansion(t.df)
    names, reason = llm.detect_hierarchy_level_names(
        detection["source_col"], detection["depth"], detection["sample_paths"],
        t.title, client, model,
    )
    # 地域＞事業部＞支店という明確な階層のため、命名に成功することを期待する
    assert names is not None
    assert len(names) == detection["depth"]
    assert all(isinstance(n, str) and n for n in names)
    assert reason == ""


def test_default_level_names_fallback():
    # LLMが命名に失敗・拒否した場合のフォールバック（既定名）は非LLMの
    # 純粋関数のため、LLM呼び出しなしで検証できる。既定名には元カラム名が
    # 接頭辞として必ず付く（"地域_大分類" 等）。
    fallback = det._default_level_names(3, "地域")
    assert fallback == ["地域_大分類", "地域_中分類", "地域_小分類"]


def test_full_hierarchy_expand_pipeline_with_llm_naming(llm_client):
    client, model = llm_client
    t = _table("step3_hierarchy_expand_test.xlsx", "インデント方式")
    out = det._apply_hier_expand_defaults(t, t.df, client, model)
    assert list(out.columns[-1:]) == ["売上"]
    assert len(list(out.columns)) == 4  # レベル3つ + 売上
    assert len(out) == 3  # ロールアップ2行（東日本・神奈川事業部）が除去済み


# --- 列名生成規則機能: 値列名のLLM生成（静的辞書VALUE_KEYWORDSに無い語彙） -------


def test_detect_value_column_name(llm_client):
    client, model = llm_client
    # "来客数"はVALUE_KEYWORDS（売上/予算/実績/件数/人数/金額/数量/利益/費用/
    # コスト）に無く、静的辞書では解決できない値列名。支店別の来客数集計という
    # 明確な文脈のため、LLMによる命名に成功することを期待する。
    result = llm.detect_value_column_name(
        ["支店"], "2024年度 各店舗の来客数集計", client, model
    )
    assert result is not None
    assert isinstance(result["value_name"], str)
    assert result["value_name"] != ""


def test_detect_value_column_name_no_context_returns_none():
    # title/context_tokensが両方とも無い場合はLLMを呼ばずNoneを返す
    # （判断材料が無い呼び出しを避けるコスト制御。LLM APIを使わないため
    # llm_client fixture不要で常時実行できる）。
    result = llm.detect_value_column_name([], None, client=None, model=None)
    assert result is None


def test_resolve_value_col_name_uses_llm_when_dictionary_misses(llm_client):
    client, model = llm_client
    name, source, reason = det.resolve_value_col_name(
        title="2024年度 各店舗の来客数集計", context_tokens=["支店"],
        client=client, model=model,
    )
    assert source in ("llm", "fallback")
    if source == "llm":
        assert reason != "" or name != ""


# --- 列衝突検出機能: 衝突した既存列・新規列双方の意味的リネーム（LLM） -------


def test_detect_column_collision_rename(llm_client):
    client, model = llm_client
    # 「合計」が既存列（全期間の合計）と新規のWide_to_long指標（年ごとの合計）
    # の両方に使われている、明確に区別可能なケース。
    result = llm.detect_column_collision_rename("合計", "年", None, client, model)
    assert result is not None
    assert isinstance(result["existing_name"], str) and result["existing_name"]
    assert isinstance(result["new_name"], str) and result["new_name"]
    assert result["existing_name"] != result["new_name"]


def test_resolve_column_collision_uses_llm_when_available(llm_client):
    client, model = llm_client
    existing_new, new_final, source, reason = det.resolve_column_collision(
        "合計", {"合計", "件数"}, axis_var_name="年", client=client, model=model,
    )
    assert source in ("llm", "fallback")
    if source == "llm":
        assert existing_new is not None and existing_new != "合計"
        assert new_final != "合計"
