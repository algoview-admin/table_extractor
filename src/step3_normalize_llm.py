"""
ステップ3 テーブル正規化モジュール（LLM使用処理）。

処理概要: Transpose（行列逆転）の検出と変換、多段ヘッダーの検出と解決機能（軸展開。
          多段ヘッダーが独立した複数カテゴリ軸の交差かどうかの判定・軸命名）、
          Wide_to_long Tier3（区切り文字のない複合列名から決定論的に発見した
          軸候補が本当に均質な1つのカテゴリ軸かどうかの確認・命名）など、
          意味判定にLLMが必要なテーブル整形処理を扱う。
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
    """
    api_type = os.getenv("OPENAI_API_TYPE", "openai").strip().lower()

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
