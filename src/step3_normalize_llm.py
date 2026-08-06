"""
ステップ3 テーブル正規化モジュール（LLM使用処理）。

処理概要: Transpose（行列逆転）の検出と変換、多段ヘッダーの検出と解決機能（軸展開。
          多段ヘッダーが独立した複数カテゴリ軸の交差かどうかの判定・軸命名）、
          Wide_to_long Tier3（区切り文字のない複合列名から決定論的に発見した
          軸候補が本当に均質な1つのカテゴリ軸かどうかの確認・命名）、
          ファイル外メタデータからの派生カラム生成機能（ファイル名・シート名に
          しか現れない付帯情報の抽出）など、意味判定にLLMが必要なテーブル
          整形処理を扱う。
          Step3 と Step4（src/step4_analyze.py）は処理の性質が異なり
          （Step3 の出力が Step4 の入力になる関係）依存を持たせたくない
          ため、LLM クライアント生成・API 呼び出しは step4_analyze.py の
          ものを共有せず、このファイルに独立実装する。
          決定論的処理（正規表現・語彙辞書のみで完結する処理）は
          src/step3_normalize_determ.py を参照。
入力    : DetectedTable.df（step2_detect が構築した生 DataFrame）
出力    : 変換済み DataFrame、transpose_info（{entity_axis_name, reasoning}）、
          多段ヘッダーの検出と解決機能（軸展開）情報（{axis_names, value_name, reasoning}）、
          カテゴリ軸確認情報（{axis_name, reasoning}）
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Transpose（行列逆転）の検出と変換
# ---------------------------------------------------------------------------
#
# 「エンティティが列、属性が行」のように意味的に行列が逆転した表を検出し、
# 正しい向き（エンティティ＝行、属性＝列）に変換する。
#
# 列ヘッダーが固有名詞（地名・支店名等）かどうか、行ラベルが指標名かどうかは
# 正規表現・語彙辞書では汎用的に判定できないため、この機能のみ LLM を用いる。

_TRANSPOSE_SYSTEM_PROMPT = """あなたは表データ構造の分析専門家です。
与えられた表の行と列が意味的に入れ替わっている（Transpose）かどうかを判定し、
指定された JSON 形式のみで回答してください（説明文は不要）。"""

_TRANSPOSE_USER_PROMPT = """以下の表について、行と列が意味的に逆転していないか判定してください。

{table_text}

【判定基準】
- 「行列が逆転している」とは、本来は個体・エンティティ（例: 支店名、都市名、商品名、日付など、
  複数の観測対象を識別するもの）であるべき値が列ヘッダーに並び、本来は属性・指標名
  （例: 売上、利益、在庫、価格など、観測対象が共通して持つ性質の名称）であるべき値が
  1列目（ラベル列）に並んでいる状態を指す。
- 1列目（ラベル列）の値が指標・属性名のリストに見え、かつ2列目以降の列名がエンティティ
  （個体識別子）のリストに見える場合のみ is_transposed=true とする。
- 通常の縦持ち/横持ち表（エンティティが行、属性や時系列が列に並ぶ一般的な表）は対象外。
- 少しでも判断に迷う場合は is_transposed=false としてよい。

JSON形式で回答してください:
{{"is_transposed": true または false, "entity_axis_name": "エンティティ軸の新しい列名（例: 支店）。is_transposed=falseの場合はnull", "reasoning": "判断理由（日本語、1〜2文）"}}"""


def make_transpose_client() -> Tuple[Any, str]:
    """Transpose 検出用の LLM クライアントを生成する。

    step4_analyze.py の _make_client() と同等のロジックだが、Step3 は
    Step4 のプライベート関数に依存させたくないため独立実装している。

    OPENAI_API_TYPE が明示的に設定されていればそれに従う。未設定の場合は
    .env に実際に設定されている方（AZURE_OPENAI_API_KEY があれば Azure、
    なければ OpenAI）を自動選択する。
    """
    api_type = os.getenv("OPENAI_API_TYPE", "").strip().lower()
    if not api_type:
        api_type = "azure" if os.getenv("AZURE_OPENAI_API_KEY") else "openai"

    if api_type == "azure":
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )
        model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
    else:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_MODEL", "gpt-5.4")

    return client, model


def _call_transpose_api(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    max_completion_tokens: int = 500,
    timeout: float = 60.0,
) -> str:
    """429 / レートリミットエラー時にリトライしながら Transpose 判定 API を呼び出す。"""
    last_exc: Optional[Exception] = None
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_completion_tokens=max_completion_tokens,
                timeout=timeout,
            )
            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise ValueError(
                    "Transpose 判定の GPT レスポンスがトークン上限に達して途中で切れました。"
                )
            return choice.message.content or ""
        except Exception as exc:
            err = str(exc)
            if (
                "429" in err
                or "too_many_requests" in err.lower()
                or "rate_limit" in err.lower()
            ):
                last_exc = exc
                if attempt < 3:
                    time.sleep(15 * (2**attempt))  # 15秒、30秒、60秒
                    continue
            raise
    raise RuntimeError(f"Transpose 判定 API 呼び出しが最大リトライ回数を超えました: {last_exc}")


_TRANSPOSE_MAX_SAMPLE_ROWS = 30  # プロンプトに載せる最大サンプル行数（トークンコスト抑制用）


def _format_table_for_transpose_check(
    df: Any, max_sample_rows: int = _TRANSPOSE_MAX_SAMPLE_ROWS
) -> str:
    """Transpose 判定プロンプト用に列名とサンプル行を軽量テキスト化する。

    行数が多いテーブルでも max_sample_rows でトランケートしてトークンコストを
    抑える（LLM 呼び出し自体はスキップしない）。
    """
    columns = [str(c) for c in df.columns]
    sample = df.head(max_sample_rows)
    total_rows = len(df)
    lines = [", ".join(str(v) for v in row) for row in sample.itertuples(index=False)]
    return (
        f"列名: {columns}\n"
        f"サンプル行（先頭{len(sample)}行 / 全{total_rows}行中）:\n" + "\n".join(lines)
    )


def detect_transpose(df: Any, client: Any, model: str) -> Optional[Dict[str, Any]]:
    """表の行列が意味的に逆転しているかを LLM で判定する。

    列数が1列以下の場合は転置が意味をなさないため LLM を呼ばず None を返す
    （意味的なヒューリスティックではなく形状上自明な退化ケースの除外のみ）。
    それ以外は行数に関わらず必ず LLM を呼ぶ（サンプル行数のみ上限でトランケートする）。

    ネットワークエラー・JSON パース失敗等は握りつぶして None を返し、
    1テーブルの失敗が検出処理全体を落とさないようにする。

    Returns:
      検出された場合: {"entity_axis_name": str, "reasoning": str}
      検出されなかった場合 / 判定不能な場合: None
    """
    if df is None or df.empty or len(df.columns) < 2:
        return None

    table_text = _format_table_for_transpose_check(df)
    messages = [
        {"role": "system", "content": _TRANSPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": _TRANSPOSE_USER_PROMPT.format(table_text=table_text)},
    ]
    try:
        content = _call_transpose_api(client, model, messages)
        raw = json.loads(content)
    except Exception:
        return None

    if not isinstance(raw, dict) or not raw.get("is_transposed"):
        return None

    entity_axis_name = str(raw.get("entity_axis_name") or "").strip()
    if not entity_axis_name:
        entity_axis_name = "項目"

    return {
        "entity_axis_name": entity_axis_name,
        "reasoning": str(raw.get("reasoning") or ""),
    }


# ---------------------------------------------------------------------------
# 多段ヘッダーの検出と解決機能（軸展開）
# ---------------------------------------------------------------------------
#
# src/step3_normalize_determ.py の detect_multi_axis_header が構造的に
# 「単一値でも時系列でもない軸候補行」を切り分けた後、それが本当に独立した
# 複数のカテゴリ軸（例: 科目, 支店）を交差させたものかどうか、また各軸に
# どんな列名を付けるべきかは、値の意味を理解する必要があるため LLM を用いる。

_MULTI_AXIS_SYSTEM_PROMPT = """あなたは表データ構造の分析専門家です。
多段（複数行）の結合セルヘッダーが、複数の独立したカテゴリ軸（次元）を
交差させたクロス集計形式かどうかを判定し、指定された JSON 形式のみで
回答してください（説明文は不要）。"""

_MULTI_AXIS_USER_PROMPT = """以下は、表の多段ヘッダーのうち、単一値でも時系列でもないと
判定された「軸候補」の各行の値一覧です（表の上段から下段の順）。

{axis_lines}
{title_line}

【判定基準】
- 各行が、互いに独立したカテゴリ軸（例: 科目、支店、商品カテゴリなど）を
  表しており、これらを交差させて元の値列が構成されている場合のみ
  is_valid=true としてください。
- 単に表記ゆれや不規則な値の羅列で、明確な軸として意味づけできない場合は
  is_valid=false としてください。
- axis_names は入力された行の順序と対応する配列にしてください（行数と同じ長さ）。
- 少しでも判断に迷う場合は is_valid=false としてよい。

【indicator_axis_index について】
- 軸候補の中に、「売上」「原価」「利益」のように**異なる指標・メトリクスの名称**を
  表す行が含まれる場合、その行は他の軸（支店・年度など）と違い、縦持ちの1つの
  値列に統合するより、指標ごとに別々の列として残した方が扱いやすいことが多いです。
  該当する行がある場合はその行番号（0始まり、axis_namesの配列位置と対応）を
  indicator_axis_index に設定してください。
- 該当する行がない場合（すべての軸が支店・年度・性別など、同じ1つの値を
  分類するだけの軸である場合）は indicator_axis_index=null としてください。
- 少しでも判断に迷う場合は null としてよい。

JSON形式で回答してください:
{{"is_valid": true または false, "axis_names": ["各行に対応する軸名（日本語、例: 科目、支店）", "..."], "value_name": "値列の名称（日本語、例: 金額、数量）。文脈から判断できない場合は「値」", "indicator_axis_index": 指標軸の行番号（0始まり）または null, "reasoning": "判断理由（日本語、1〜2文）"}}"""


def detect_dimension_axes(
    axis_candidates: List[List[str]],
    title: Optional[str],
    client: Any,
    model: str,
) -> Optional[Dict[str, Any]]:
    """多段ヘッダーの軸候補行が、真に独立したカテゴリ軸の交差かどうかを LLM で判定する。

    axis_candidates: 各軸候補行の重複排除済み値リスト（表の上段→下段の順）。
    空リストの場合は LLM を呼ばず None を返す。

    ネットワークエラー・JSON パース失敗・行数不一致等は握りつぶして None を返し、
    1テーブルの失敗が検出処理全体を落とさないようにする。

    Returns:
      有効と判定された場合: {"axis_names": [...], "value_name": str,
        "indicator_axis_index": Optional[int]（axis_candidatesの何番目が指標軸か。
        該当なしなら None。Wide_to_long同様、指標軸は縦持ちにせず列として残す
        ための情報として apply_multi_axis_header が使う）, "reasoning": str}
      無効 / 判定不能な場合: None
    """
    if not axis_candidates:
        return None

    axis_lines = "\n".join(
        f"行{i + 1}: {values}" for i, values in enumerate(axis_candidates)
    )
    title_line = f"\n表タイトル: {title}" if title else ""
    messages = [
        {"role": "system", "content": _MULTI_AXIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _MULTI_AXIS_USER_PROMPT.format(
                axis_lines=axis_lines, title_line=title_line
            ),
        },
    ]
    try:
        content = _call_transpose_api(client, model, messages)
        raw = json.loads(content)
    except Exception:
        return None

    if not isinstance(raw, dict) or not raw.get("is_valid"):
        return None

    axis_names = raw.get("axis_names")
    if not isinstance(axis_names, list) or len(axis_names) != len(axis_candidates):
        return None
    axis_names = [str(n).strip() or f"軸{i + 1}" for i, n in enumerate(axis_names)]

    value_name = str(raw.get("value_name") or "").strip() or "値"

    indicator_axis_index: Optional[int] = None
    raw_idx = raw.get("indicator_axis_index")
    if isinstance(raw_idx, int) and not isinstance(raw_idx, bool) and 0 <= raw_idx < len(axis_candidates):
        indicator_axis_index = raw_idx

    return {
        "axis_names": axis_names,
        "value_name": value_name,
        "indicator_axis_index": indicator_axis_index,
        "reasoning": str(raw.get("reasoning") or ""),
    }


# ---------------------------------------------------------------------------
# カテゴリ軸の確認（Wide_to_long Tier3: 区切り文字のない複合列名向け）
# ---------------------------------------------------------------------------
#
# src/step3_normalize_determ.py の _find_concatenated_axis_candidates が、
# 区切り文字のない複合列名（例: "東京支店売上"）から頻度ベース（語彙を使わない
# 構造シグナルのみ）で「軸トークン候補」を発見済みの場合に呼ばれる。
# その候補群が本当に均質な1つのカテゴリ軸（支店・性別・年代など）として
# 意味を持つかどうかは値の意味理解が必要なため、ここのみ LLM を用いる。
# detect_dimension_axes と異なり、LLM は分割点そのものを発見するのではなく、
# 既に決定論的に発見された候補を検証・命名するだけに役割を限定する。

_CATEGORY_AXIS_SYSTEM_PROMPT = """あなたは表データ構造の分析専門家です。
列名の分解によって発見された「軸候補」の値一覧が、意味のある1つの均質な
カテゴリ軸として成立するかどうかを判定し、指定された JSON 形式のみで
回答してください（説明文は不要）。"""

_CATEGORY_AXIS_USER_PROMPT = """以下は、横持ち表の列名を分解した結果、発見された
「軸候補」の値一覧です。

軸候補: {candidate_values}
{other_axis_line}{title_line}

【判定基準】
- 軸候補が、互いに独立した1つの均質なカテゴリ軸（例: 支店、性別、年代、
  商品カテゴリなど）として意味を持つ場合のみ is_valid=true としてください。
- 単なる偶然の部分文字列一致（無関係な語が同じ接頭辞・接尾辞を共有している
  だけ）、あるいは列名の衝突回避のための連番等の場合は is_valid=false と
  してください。
- 少しでも判断に迷う場合は is_valid=false としてよい。

JSON形式で回答してください:
{{"is_valid": true または false, "axis_name": "軸の名称（日本語、例: 支店）。is_valid=falseの場合はnull", "reasoning": "判断理由（日本語、1〜2文）"}}"""


def detect_category_axis(
    candidate_values: List[str],
    other_axis_hint: Optional[List[str]],
    title: Optional[str],
    client: Any,
    model: str,
) -> Optional[Dict[str, Any]]:
    """決定論的に発見済みの軸トークン候補が、本当に均質な1つのカテゴリ軸として
    意味を持つかを LLM に確認・命名させる（構造の発見自体はさせない）。

    candidate_values: 発見済みの軸トークン候補（重複排除済み）。空リストの場合は
    LLM を呼ばず None を返す。
    other_axis_hint: 参考情報として渡す、組み合わさる指標候補（あれば）。

    ネットワークエラー・JSON パース失敗等は握りつぶして None を返し、
    1テーブルの失敗が検出処理全体を落とさないようにする。

    Returns:
      有効と判定された場合: {"axis_name": str, "reasoning": str}
      無効 / 判定不能な場合: None
    """
    if not candidate_values:
        return None

    other_axis_line = (
        f"（参考）これらと組み合わさる指標候補: {other_axis_hint}\n"
        if other_axis_hint
        else ""
    )
    title_line = f"表タイトル: {title}" if title else ""
    messages = [
        {"role": "system", "content": _CATEGORY_AXIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _CATEGORY_AXIS_USER_PROMPT.format(
                candidate_values=candidate_values,
                other_axis_line=other_axis_line,
                title_line=title_line,
            ),
        },
    ]
    try:
        content = _call_transpose_api(client, model, messages)
        raw = json.loads(content)
    except Exception:
        return None

    if not isinstance(raw, dict) or not raw.get("is_valid"):
        return None

    axis_name = str(raw.get("axis_name") or "").strip() or "区分"

    return {
        "axis_name": axis_name,
        "reasoning": str(raw.get("reasoning") or ""),
    }


# ---------------------------------------------------------------------------
# 値列名の生成（列名生成規則機能: 静的辞書で決まらない値列名をLLMで補う）
# ---------------------------------------------------------------------------
#
# src/step3_normalize_determ.py の resolve_value_col_name が、表タイトルと
# VALUE_KEYWORDS（静的辞書）の一致で値列名を決められなかった場合にのみ
# 呼ばれる。detect_category_axis と同様、構造の発見自体はさせず、
# 与えられた文脈（表タイトル・区分列名）から値列の意味を命名させるだけに
# 役割を限定する。

_VALUE_COLUMN_NAME_SYSTEM_PROMPT = """あなたは表データ構造の分析専門家です。
横持ち表を縦持ちに変換する際に新規生成される「値列」（集計された数値の列）に
ふさわしい名称を判定し、指定された JSON 形式のみで回答してください
（説明文は不要）。"""

_VALUE_COLUMN_NAME_USER_PROMPT = """以下は、横持ち表を縦持ちに変換する際に
新規生成される値列（数値）の文脈情報です。

{title_line}
（参考）表に含まれる区分・ラベル列名: {context_tokens}

【判定基準】
- 表タイトルや区分列名から、値列が何を表す数値か明確に読み取れる場合のみ
  value_name に具体的な名称（例: 売上、原価、来店者数）を設定してください。
- 何を集計した数値か判断できない場合は value_name を null にしてください。
- 少しでも判断に迷う場合は null としてよい。

JSON形式で回答してください:
{{"value_name": "値列の名称（日本語）。判断できない場合はnull", "reasoning": "判断理由（日本語、1〜2文）"}}"""


def detect_value_column_name(
    context_tokens: List[str],
    title: Optional[str],
    client: Any,
    model: str,
) -> Optional[Dict[str, Any]]:
    """Stack/Wide_to_longで新規生成される値列の列名をLLMに判定・命名させる。

    context_tokens: 参考情報として渡す、表に含まれる区分・ラベル列名。
    title が無く context_tokens も空の場合は LLM を呼ばず None を返す
    （判断材料が無い呼び出しはコスト的に無意味なため）。

    ネットワークエラー・JSON パース失敗等は握りつぶして None を返し、
    1テーブルの失敗が検出処理全体を落とさないようにする。

    Returns:
      命名できた場合: {"value_name": str, "reasoning": str}
      判定不能な場合: None
    """
    if not title and not context_tokens:
        return None

    title_line = f"表タイトル: {title}" if title else ""
    messages = [
        {"role": "system", "content": _VALUE_COLUMN_NAME_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _VALUE_COLUMN_NAME_USER_PROMPT.format(
                title_line=title_line, context_tokens=context_tokens,
            ),
        },
    ]
    try:
        content = _call_transpose_api(client, model, messages)
        raw = json.loads(content)
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None
    value_name = str(raw.get("value_name") or "").strip()
    if not value_name:
        return None

    return {
        "value_name": value_name,
        "reasoning": str(raw.get("reasoning") or ""),
    }


# ---------------------------------------------------------------------------
# 列衝突検出機能（Stack/Wide_to_longで新規追加される列名が既存列と衝突する場合）
# ---------------------------------------------------------------------------
#
# src/step3_normalize_determ.py の resolve_column_collision が、Wide_to_long/
# クロス集計で新規生成する列名（指標名・値列名）が既存のラベル列と完全一致
# する場合にのみ呼ぶ。単純に "_1" と連番を振るだけでは「どちらが元の値で
# どちらが新しく縦持ち変換で分解された値か」が列名から読み取れなくなるため、
# 双方に意味の異なる名前を付け直せるかどうかの判断にのみ LLM を用いる。

_COLUMN_COLLISION_SYSTEM_PROMPT = """あなたは表データ構造の分析専門家です。
表の変換によって新しく追加される列の名前が、その表に既に存在する列の名前と
完全に一致してしまうケースについて、双方を区別できる分かりやすい列名を
提案し、指定された JSON 形式のみで回答してください（説明文は不要）。"""

_COLUMN_COLLISION_USER_PROMPT = """以下の表で、列名の衝突が発生しています。

{title_line}
衝突している列名: {colliding_name}
- 既存列: 変換前からこの表に存在していた列（集計期間・集計範囲などの
  前提はタイトルや表の文脈から推測してください）
- 新規列: {new_col_context}

【判定基準】
- 既存列・新規列それぞれが実際に何を表しているかを文脈から推測し、
  両者を区別できる名前を1つずつ提案してください
  （例: 既存が全期間の合計、新規が軸ごとの合計なら「全期間合計」「{axis_hint}合計」等）。
- 意味の違いを自信を持って判断できない場合は、無理に具体的な名前を
  ひねり出さず、is_valid=false としてください。

JSON形式で回答してください:
{{"is_valid": true または false, "existing_name": "既存列の新しい名前（日本語）。is_valid=falseの場合はnull", "new_name": "新規列の新しい名前（日本語）。is_valid=falseの場合はnull", "reasoning": "判断理由（日本語、1〜2文）"}}"""


def detect_column_collision_rename(
    colliding_name: str,
    axis_var_name: Optional[str],
    title: Optional[str],
    client: Any,
    model: str,
    new_col_context_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """既存列と新規生成列が同名で衝突する場合に、双方が区別できる意味の異なる
    列名をLLMに提案させる。axis_var_name は新規列が属する軸名（例: "年"）で、
    新規側の命名根拠として渡す。

    ネットワークエラー・JSON パース失敗・LLMが判断不能とした場合は None を
    返し、呼び出し側は機械的な連番付与にフォールバックする。

    Returns:
      提案できた場合: {"existing_name": str, "new_name": str, "reasoning": str}
      判定不能な場合: None
    """
    axis_hint = axis_var_name or "区分ごとの"
    new_col_context = new_col_context_override or (
        f"「{axis_var_name}」ごとに縦持ち変換で分解された値"
        if axis_var_name
        else "縦持ち変換で新たに分解された値"
    )
    title_line = f"表タイトル: {title}" if title else ""
    messages = [
        {"role": "system", "content": _COLUMN_COLLISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _COLUMN_COLLISION_USER_PROMPT.format(
                title_line=title_line, colliding_name=colliding_name,
                new_col_context=new_col_context, axis_hint=axis_hint,
            ),
        },
    ]
    try:
        content = _call_transpose_api(client, model, messages)
        raw = json.loads(content)
    except Exception:
        return None

    if not isinstance(raw, dict) or not raw.get("is_valid"):
        return None
    existing_name = str(raw.get("existing_name") or "").strip()
    new_name = str(raw.get("new_name") or "").strip()
    if not existing_name or not new_name or existing_name == new_name:
        return None

    return {
        "existing_name": existing_name,
        "new_name": new_name,
        "reasoning": str(raw.get("reasoning") or ""),
    }


# ---------------------------------------------------------------------------
# ファイル外メタデータからの派生カラム生成機能
# ---------------------------------------------------------------------------
#
# ファイル名・シート名には、サービス名・オプション種別・指標名・年度など、
# 表データ本体（列名・値）には現れない付帯情報が含まれることがある。
# どのトークンがそうした意味のあるメタデータで、どれが「販売月報」「一覧」の
# ような単なる文書種別を表す定型語かは意味理解が必要なため LLM を用いる。
# 抽出結果を定数列としてデータに埋め込む処理は決定論的なため、
# step3_normalize_determ.apply_external_metadata が担う。

_EXTERNAL_META_SYSTEM_PROMPT = """あなたは表データ構造の分析専門家です。
ファイル名・シート名に含まれる、表データ本体（列名・値）には現れない付帯情報
（メタデータ）を抽出し、それぞれを表のどの位置に置くべきかを判定して、
指定された JSON 形式のみで回答してください（説明文は不要）。"""

_EXTERNAL_META_USER_PROMPT = """以下のファイル名・シート名から、表データ本体
（列名・値）には現れていない付帯情報を抽出し、それぞれを表のどの位置に
挿入すべきかを判定してください。

ファイル名: {filename}
シート名: {sheet_name}
{title_line}既存の区分列（左→右の順）: {dimension_columns}
集計軸列（区分列より右側、値列より左側に来る列）: {axis_name}
値列（集計された数値そのもの。最も右側に来る）: {value_name}

【抽出対象の例（特定の語彙ではなく、一般的なパターンとして捉えてください。
具体的な固有名詞例は挙げません。ファイル名・シート名の構造・位置関係と
文脈から意味的に判断してください）】
- サービス名・製品名: ファイル名や表タイトルに含まれる、対象サービス／
  製品を指す固有名詞
- オプション種別・機能区分: サービスの中でも特定の機能・オプションを
  指す語（シート名に含まれることが多い）
- 指標名: そのシート／表全体が集計対象とする数値の種類・単位を表す語。
  シート名が「◯◯別（△△）」のような形式の場合、末尾の括弧内 "△△" が
  該当することが多い
- 年度: 表全体を通じて一律に成り立つ単一の西暦4桁の数字（例:「2024年度」）。
  「2023〜2026年度」のような複数年にまたがる範囲表記は対象外
  （下記「抽出してはいけないもの」参照）

【抽出してはいけないもの】
- 既存の列名に既に含まれる情報
- 「販売月報」「エリア別」「一覧」のような、文書の種類・体裁を表すだけの汎用語
- ファイル管理上の記号（バージョン番号、年度以外の連番など）で表の値として
  意味を持たないもの
- 「2023〜2026年度」「FY22-24」のような年・期間の範囲表記。範囲は表の全行に
  一律で成り立つ固定値ではなく、集計軸列（{axis_name}）が時系列（年・年度等）
  を表す場合はその軸が扱う期間そのものを言い換えているだけであることが多い
  （集計軸へ展開後の値としてすでに表本体に現れる情報のため、範囲の両端を
  別々の固定値として抽出しないこと）
- 少しでも判断に迷うものは抽出しない（抽出しすぎより、何も抽出しない方が安全）

【column_name について】
- 日本語で簡潔な列名にしてください（例: "サービス名", "オプション種別", "年度"）
- 抽出した値が「何を数えた／集計した数値か」を表す指標名の場合は、
  "区分"のような曖昧な語を避け "指標名" としてください（例: シート名が
  「◯◯別（△△）」の形式で、末尾の括弧内 "△△" が集計対象の数値の種類を
  表している場合、column_name="指標名", value="△△" とする）
- 既存の区分列名・他の抽出項目の column_name と重複しない名前にしてください

【並び順（anchor / position）の決め方 ── 重要 ──】
表の区分列は「粒度の階層」で左から右へ並べる。他を包含する広い区分ほど左、
包含される狭い区分ほど右に来る。集計軸列は全区分列より右、値列は最も右。
抽出した各項目についても、既存の区分列・集計軸列・他の抽出項目との
「包含関係」だけを根拠に位置を決め、anchor（隣接させる列名）と
position（"before"=その列の左 / "after"=その列の右）を指定すること:
{primary_axis_line}- 項目 X が既存の区分列（または他の抽出項目）Y を概念的に包含する
  （X が Y の上位区分にあたる）場合 → anchor=Y, position="before"
  （ただし上記の先頭列固定ルールが優先される。先頭列に対しては
  position="before" を使わないこと）
- 逆に X が Y に包含される（X が Y の下位区分にあたる）場合 →
  anchor=Y, position="after"
- 集計対象の数値そのものの種類・単位を表す項目（指標名に該当するもの）は、
  既存区分列・他の抽出項目を合わせた中で最も細かい（最も右にくる）区分の
  直後、つまり値に最も近い位置に置く → anchor=その最も細かい区分の列名
  （既存区分列でも他の抽出項目の column_name でもよい）, position="after"
- 集計軸列（{axis_name}）を時間的・階層的に包含する上位区分にあたる項目
  （例: 集計軸が月なら、その年を表す項目）は、軸の直前に置く →
  anchor="{axis_name}", position="before"
- anchor には既存の区分列名・集計軸列名（"{axis_name}"）・他の抽出項目の
  column_name のいずれかを指定できる。包含関係が判断できない場合は
  anchor=null, position=null とする（この場合は集計軸の直前に置かれる）。
- 具体的な語彙ではなく、列同士の意味的な包含関係だけで判断すること。

JSON形式で回答してください:
{{"items": [{{"column_name": "列名", "value": "抽出した値", "source": "filename" または "sheet_name", "is_year": true または false, "anchor": "隣接させる列名または null", "position": "before または after または null"}}, ...], "reasoning": "判断理由（日本語、1〜2文）"}}
抽出対象が無い場合は {{"items": [], "reasoning": "..."}} としてください。"""


def extract_external_metadata(
    filename: Optional[str],
    sheet_name: Optional[str],
    title: Optional[str],
    dimension_columns: List[str],
    axis_name: str,
    value_name: str,
    client: Any,
    model: str,
) -> Optional[Dict[str, Any]]:
    """ファイル名・シート名から、表データ本体には現れない付帯メタデータを LLM で抽出し、
    各項目の挿入位置（anchor/position）も判定させる。

    filename・sheet_name のいずれも無い場合は LLM を呼ばず None を返す。
    dimension_columns: 縦持ち変換後も残る既存の区分列（横持ち時の左→右の順）。
    axis_name / value_name: 縦持ち変換後の集計軸列名・値列名（区分列より右に来る）。
    位置判定のアンカー候補として、既存列との重複防止の参照としても使う。

    ネットワークエラー・JSON パース失敗等は握りつぶして None を返し、
    1テーブルの失敗が検出処理全体を落とさないようにする。

    Returns:
      抽出された場合: {"items": [{"column_name", "value", "source", "is_year",
        "anchor", "position"}, ...], "reasoning": str}（items が空リストの
        場合も含む。既存列と重複する column_name や不正な形式の項目は除去済み。
        anchor/position は解決できない場合 None）
      判定不能な場合: None
    """
    if not filename and not sheet_name:
        return None

    title_line = f"表タイトル: {title}\n" if title else ""
    primary_axis_line = ""
    if dimension_columns:
        primary_axis_line = (
            f"- 既存区分列の先頭列「{dimension_columns[0]}」は、この表のデフォルトの主軸\n"
            "  として常に一番左を維持する。どの抽出項目も、この列を anchor にして\n"
            "  position=\"before\" を指定してはならない（この列より上位の概念に見える\n"
            "  項目であっても、必ず position=\"after\" として先頭列の直後に置くこと）。\n"
        )
    messages = [
        {"role": "system", "content": _EXTERNAL_META_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _EXTERNAL_META_USER_PROMPT.format(
                filename=filename or "（なし）",
                sheet_name=sheet_name or "（なし）",
                title_line=title_line,
                dimension_columns=[str(c) for c in dimension_columns],
                axis_name=axis_name,
                value_name=value_name,
                primary_axis_line=primary_axis_line,
            ),
        },
    ]
    try:
        content = _call_transpose_api(client, model, messages)
        raw = json.loads(content)
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None
    items = raw.get("items")
    if not isinstance(items, list):
        return None

    existing_lower = {str(c).strip().lower() for c in dimension_columns} | {
        axis_name.strip().lower(),
        value_name.strip().lower(),
    }
    cleaned: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        column_name = str(item.get("column_name") or "").strip()
        value = item.get("value")
        source = item.get("source")
        if not column_name or value is None or source not in ("filename", "sheet_name"):
            continue
        if column_name.lower() in existing_lower:
            continue
        anchor_raw = item.get("anchor")
        anchor = anchor_raw.strip() if isinstance(anchor_raw, str) and anchor_raw.strip() else None
        position = item.get("position") if item.get("position") in ("before", "after") else None
        if anchor is None or position is None:
            anchor, position = None, None
        cleaned.append(
            {
                "column_name": column_name,
                "value": str(value).strip(),
                "source": source,
                "is_year": bool(item.get("is_year")),
                "anchor": anchor,
                "position": position,
            }
        )

    return {"items": cleaned, "reasoning": str(raw.get("reasoning") or "")}


# ---------------------------------------------------------------------------
# Wide_to_long/クロス集計: 集計軸列・指標列の挿入位置判定
# ---------------------------------------------------------------------------
#
# stack_wide_to_long/stack_cross_table は、区分列（label_cols）＋集計軸列＋
# 指標列という構造で縦持ち表を組み立てる。区分列の中に、集計軸の各水準とは
# 無関係に表全体で意味を持つ値（列衝突検出機能で退避リネームされた
# 「全期間累計」のような列など）が混ざっている場合、それを集計軸・指標列の
# 手前に置くより後ろ（値に近い位置）に置く方が自然な場合がある。
# extract_external_metadata の anchor/position 判定と同じ考え方で、
# LLMに区分列のどこを境に集計軸・指標列を挿入すべきか判定させる。

_AXIS_POSITION_SYSTEM_PROMPT = """あなたは表データ構造の分析専門家です。
横持ち表を縦持ちに変換する際に新たに生成される集計軸列・指標列を、既存の
区分列のどの位置に挿入すべきか判定して、指定された JSON 形式のみで
回答してください（説明文は不要）。"""

_AXIS_POSITION_USER_PROMPT = """以下の表を縦持ちに変換します。新たに生成される
集計軸列（および指標列）を、既存の区分列のどの位置に挿入すべきか判定して
ください。

{title_line}既存の区分列（変換後も残る列。左→右の順）: {label_cols}
新たに生成される集計軸列: {axis_name}
新たに生成される指標列: {indicator_names}

集計軸列・指標列は必ず連続したまとまりとして挿入されます（分割して挿入する
ことはできません）。

区分列の中に、集計軸の各水準（{axis_name}の値）とは無関係に表全体で意味を
持つ値（例: 軸をまたいだ全期間の累計・総合計のような値）が含まれる場合、
その列は集計軸・指標列より右側（値に近い位置）に置く方が自然です。それ以外の、
行を一意に識別するための区分列（例: 支店名・商品名）は、集計軸より左側に
残してください。

区分列のうち、どの列を境に集計軸・指標列を挿入すべきか、anchor（隣接させる
区分列名）と position（"before"=その列の左に挿入／"after"=その列の右に挿入）
で答えてください。区分列の先頭列は常に一番左を維持し、先頭列を anchor に
position="before" を指定してはいけません。
判断に迷う場合、または区分列がすべて行の識別用（軸と無関係な値が無い）場合は
anchor=null, position=null としてください（この場合は全区分列の直後に
集計軸・指標列が挿入されます＝現状のデフォルト）。

JSON形式で回答してください:
{{"anchor": "区分列名またはnull", "position": "before または after または null", "reasoning": "判断理由（日本語、1〜2文）"}}"""


def detect_axis_group_position(
    label_cols: List[str],
    axis_name: str,
    indicator_names: List[str],
    title: Optional[str],
    client: Any,
    model: str,
) -> Optional[Dict[str, Any]]:
    """Wide_to_long/クロス集計が生成する集計軸列(+指標列)を、区分列のどの
    位置に挿入すべきかを LLM で判定する。

    label_cols が2列未満（判断の余地がない）、または client 未指定の場合は
    LLM を呼ばず None を返す（呼び出し側は全区分列の後ろに軸列を挿入する
    デフォルト挙動にフォールバックする）。
    ネットワークエラー・JSON パース失敗等は握りつぶして None を返し、
    1テーブルの失敗が検出処理全体を落とさないようにする。

    Returns:
      判定できた場合: {"anchor": Optional[str], "position": Optional["before"|"after"],
        "reasoning": str}
      判定不能な場合: None
    """
    if client is None or len(label_cols) < 2:
        return None

    title_line = f"表タイトル: {title}\n" if title else ""
    messages = [
        {"role": "system", "content": _AXIS_POSITION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _AXIS_POSITION_USER_PROMPT.format(
                title_line=title_line,
                label_cols=[str(c) for c in label_cols],
                axis_name=axis_name,
                indicator_names=[str(c) for c in indicator_names],
            ),
        },
    ]
    try:
        content = _call_transpose_api(client, model, messages)
        raw = json.loads(content)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    anchor_raw = raw.get("anchor")
    anchor = anchor_raw.strip() if isinstance(anchor_raw, str) and anchor_raw.strip() else None
    position = raw.get("position") if raw.get("position") in ("before", "after") else None
    if anchor not in label_cols or position is None:
        anchor, position = None, None
    return {"anchor": anchor, "position": position, "reasoning": str(raw.get("reasoning") or "")}


# ---------------------------------------------------------------------------
# 括弧書き注釈の分離（命名・妥当性判定）
# ---------------------------------------------------------------------------
#
# src/step3_normalize_determ.py の detect_paren_annotations が「セル値末尾の
# 括弧書き注釈を、決定論的に（単位表記等を除外した上で）抽出済み」の候補列を
# 渡してくる。ここでは、抽出された注釈値の集合が「元の値を分類する独立した
# 属性」として意味を持つか、意味を持つ場合に何という列名を与えるべきかを
# 判断するため LLM を用いる（語彙辞書だけでは新規カラム名を決められないため）。

_PAREN_SPLIT_MAX_ANNOTATIONS = 20  # プロンプトに渡す注釈値の上限

_PAREN_ANNOTATION_SYSTEM_PROMPT = """あなたは表データ構造の分析専門家です。
セル値の末尾に括弧書きで埋め込まれた注釈が、独立した分類属性として別カラムに
分離する価値があるかどうかを判定し、指定された JSON 形式のみで回答してください
（説明文は不要）。"""

_PAREN_ANNOTATION_USER_PROMPT = """以下は、ある表の中で「値の末尾に括弧書き注釈が
付いている」と機械的に検出された列の一覧です。各列について、括弧を除いた値の
サンプルと、括弧内に現れた注釈値の一覧（重複排除）を示します。

{columns_block}
{title_line}
既存のカラム名（分離後もそのまま残る列）: {existing_columns}

【判定基準】
- 括弧内の値が、元の値を分類する独立した属性（例: 支店の"本社/支店"区分、
  担当者の所属部署など）として意味を持つ場合のみ is_valid=true としてください。
- 単なる補足説明・表記ゆれ・脚注・データ由来の注記など、独立した分類属性とは
  言えない場合は is_valid=false としてください。
- is_valid=true の場合、column_name には分離後の注釈列につける日本語の列名を
  設定してください（例: 元列が"支店"で注釈が"本社"等の場合は"支店種別"）。
- column_name は上記の既存のカラム名と同じ、または紛らわしい名前を避けて
  ください。既に同じ意味の列が存在する場合は、既存列と区別できる具体的な
  名前にするか、区別できないなら is_valid=false としてください。
- 少しでも判断に迷う場合は is_valid=false としてよい。

JSON形式で回答してください:
{{"results": [{{"index": 対象列のindex（0始まり）, "is_valid": true または false, "column_name": "分離後の注釈列名（日本語）または空文字", "reasoning": "判断理由（日本語、1〜2文）"}}, ...]}}"""


def detect_annotation_column_names(
    candidates: List[Dict[str, Any]],
    title: Optional[str],
    client: Any,
    model: str,
    existing_columns: Optional[List[str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """括弧書き注釈の分離候補列について、分離の妥当性と新カラム名を LLM で判定する。

    candidates: detect_paren_annotations が返した候補dictのリスト
      （{source_col, position, mapping, distinct_annotations, match_count,
       str_count, label_samples} を含む）。1テーブルあたり1回の呼び出しで
      全候補列をまとめて判定する。

    existing_columns: 分離時点で表に既に存在する列名一覧。LLMに渡し、既存列と
    同名・紛らわしい名前を避けさせる（それでも衝突した場合の機械的な
    退避は resolve_column_collision/resolve_generated_column_names が担う。
    ここでのプロンプト側の配慮は、その後段処理に頼らずとも最初から
    自然で重複の無い名前を選ばせるためのもの）。

    ネットワークエラー・JSON パース失敗・想定外の構造等は握りつぶして None を
    返し、1テーブルの失敗が検出処理全体を落とさないようにする。

    Returns:
      is_valid と判定された列が1つ以上あれば、それらの候補dictに
      "new_col"（LLMが決めた列名）と "reasoning" を加えたリスト。
      1つも妥当と判定されなければ None。
    """
    if not candidates:
        return None

    blocks = []
    for i, c in enumerate(candidates):
        samples = "、".join(c.get("label_samples", [])[:5])
        annotations = "、".join(c.get("distinct_annotations", [])[:_PAREN_SPLIT_MAX_ANNOTATIONS])
        blocks.append(
            f"列{i}: 元カラム名「{c['source_col']}」\n"
            f"  括弧を除いた値のサンプル: {samples}\n"
            f"  括弧内の注釈値一覧: {annotations}"
        )
    columns_block = "\n".join(blocks)
    title_line = f"\n表タイトル: {title}" if title else ""

    messages = [
        {"role": "system", "content": _PAREN_ANNOTATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _PAREN_ANNOTATION_USER_PROMPT.format(
                columns_block=columns_block, title_line=title_line,
                existing_columns=[str(c) for c in (existing_columns or [])],
            ),
        },
    ]
    try:
        content = _call_transpose_api(client, model, messages)
        raw = json.loads(content)
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None
    results = raw.get("results")
    if not isinstance(results, list):
        return None

    named: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict) or not item.get("is_valid"):
            continue
        idx = item.get("index")
        if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < len(candidates)):
            continue
        new_col = str(item.get("column_name") or "").strip()
        if not new_col:
            continue
        entry = dict(candidates[idx])
        entry["new_col"] = new_col
        entry["reasoning"] = str(item.get("reasoning") or "")
        named.append(entry)

    return named or None


# ---------------------------------------------------------------------------
# 階層圧縮カラムの展開（レベル命名）
# ---------------------------------------------------------------------------
#
# src/step3_normalize_determ.py の detect_hierarchy_expansion が「インデント
# またはセル内区切り文字で1カラムに圧縮された階層構造を、決定論的に検出・
# 深さ確定済み」の候補列を渡してくる。ここでは、各階層レベルに何という
# カラム名を与えるべきかを判断するため LLM を用いる（語彙辞書だけでは新規
# カラム名を決められないため）。妥当な既定名（大分類/中分類/小分類）が
# 別途用意されているため、失敗・否定時は呼び出し側でフォールバックする。

_HIER_LEVEL_NAMING_SYSTEM_PROMPT = """あなたは表データ構造の分析専門家です。
1カラムに圧縮された階層カテゴリの各階層レベルに、内容にふさわしい日本語の
カラム名を付け、指定された JSON 形式のみで回答してください（説明文は不要）。"""

_HIER_LEVEL_NAMING_USER_PROMPT = """以下は、ある表で「1カラムに階層カテゴリが
圧縮されている」と機械的に検出された列の情報です。

元カラム名: {source_col}
階層の深さ: {depth}
階層パスのサンプル（浅い階層 > 深い階層の順）:
{sample_paths}
{title_line}
既存のカラム名（展開後もそのまま残る列。元カラム自体は展開により置き換わる
ため含まない）: {existing_columns}

【判定基準】
- 各階層レベルの内容から、そのレベルが何を表す分類かを判断できる場合のみ
  is_valid=true とし、各レベルに具体的で自然な日本語のカラム名を付けて
  ください（例: 地域、支社、営業所）。
- level_names は浅い階層から深い階層の順で、必ず {depth} 個にしてください。
- 内容から意味づけが困難な場合（特定分野の専門知識やマスタ辞書がないと
  分類の意味が判断できない場合を含む）は is_valid=false としてください。
- level_names は上記の既存のカラム名と同じ、または紛らわしい名前を避けて
  ください。既に同じ意味の列が存在する場合は、既存列と区別できる具体的な
  名前にしてください。
- 少しでも判断に迷う場合は is_valid=false としてよい。

JSON形式で回答してください:
{{"is_valid": true または false, "level_names": ["浅い階層から深い階層順のカラム名（日本語）", "..."], "reasoning": "判断理由（日本語、1〜2文）"}}"""


def detect_hierarchy_level_names(
    source_col: str,
    depth: int,
    sample_paths: List[str],
    title: Optional[str],
    client: Any,
    model: str,
    existing_columns: Optional[List[str]] = None,
) -> Tuple[Optional[List[str]], str]:
    """階層圧縮カラムの展開候補について、各階層レベルのカラム名を LLM で判定する。

    sample_paths: "東日本 > 神奈川事業部 > 横浜支店" のような、浅い階層から
    深い階層順の文字列のリスト（detect_hierarchy_expansion が生成したサンプル）。

    existing_columns: 展開時点で表に既に存在する列名一覧（展開により消える
    source_col 自身は呼び出し側で除いて渡すこと）。LLMに渡し、既存列と
    同名・紛らわしい名前を避けさせる（それでも衝突した場合の機械的な退避は
    resolve_column_collision/resolve_generated_column_names が担う）。

    ネットワークエラー・JSON パース失敗・想定外の構造・長さ不一致等は
    握りつぶして (None, 理由) を返す。呼び出し側（_apply_hier_expand_defaults）は
    None の場合、既定名（大分類/中分類/小分類）にフォールバックする。

    Returns:
      (level_names, reason)
      level_names — is_valid かつ depth と同じ長さの level_names が得られた
        場合はそのリスト、それ以外は None。
      reason — level_names が None のとき、なぜ命名できなかったかを示す
        日本語の文字列（LLM が is_valid=false とした場合はその reasoning、
        それ以外は失敗種別）。UI側で「なぜ既定名になったか」を表示するために
        使う。命名に成功した場合は空文字。
    """
    if depth < 1 or not sample_paths:
        return None, "階層情報が不足しているため命名を実行しませんでした"

    paths_block = "\n".join(f"  {p}" for p in sample_paths)
    title_line = f"\n表タイトル: {title}" if title else ""
    messages = [
        {"role": "system", "content": _HIER_LEVEL_NAMING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _HIER_LEVEL_NAMING_USER_PROMPT.format(
                source_col=source_col,
                depth=depth,
                sample_paths=paths_block,
                title_line=title_line,
                existing_columns=[str(c) for c in (existing_columns or [])],
            ),
        },
    ]
    try:
        content = _call_transpose_api(client, model, messages)
        raw = json.loads(content)
    except Exception:
        return None, "LLM の呼び出しまたは応答の解析に失敗しました"

    if not isinstance(raw, dict):
        return None, "LLM の応答が想定した形式ではありませんでした"
    if not raw.get("is_valid"):
        reasoning = str(raw.get("reasoning") or "").strip()
        return None, reasoning or "LLM が各階層の意味を特定できないと判断しました"

    level_names = raw.get("level_names")
    if not isinstance(level_names, list) or len(level_names) != depth:
        return None, "LLM の応答が想定した形式ではありませんでした"

    names = [str(n).strip() for n in level_names]
    if any(not n for n in names):
        return None, "LLM の応答が想定した形式ではありませんでした"
    return names, ""


def apply_transpose(df: Any, entity_axis_name: str) -> Any:
    """先頭列をラベル列とみなして表を転置し、新しいラベル列名を entity_axis_name にする。

    例: 列=[指標,東京,大阪,名古屋] 行=[売上,利益,在庫] の表を、
        列=[支店,売上,利益,在庫] 行=[東京,大阪,名古屋] に変換する。
    """
    label_col = df.columns[0]
    out = df.set_index(label_col).T.reset_index()
    out = out.rename(columns={"index": entity_axis_name})
    out.columns.name = None
    return out
