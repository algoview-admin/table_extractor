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


# ---------------------------------------------------------------------------
# ファイル外メタデータからの派生カラム生成機能
# ---------------------------------------------------------------------------
#
# ファイル名・シート名には、サービス名・オプション種別・指標名・年度など、
# 表データ本体（列名・値）には現れない付帯情報が含まれることがある。
# どのトークンがそうした意味のあるメタデータで、どれが「販売月報」「一覧」の
# ような単なる文書種別を表す定型語かは意味理解が必要なため LLM を用いる。
# 抽出結果を定数列としてデータに埋め込む処理自体（apply_external_metadata）は
# 決定論的な薄い処理のため、detect/apply の対を Transpose 等と同じくこの
# ファイルにまとめて置く。

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
- 年度: 西暦4桁の数字

【抽出してはいけないもの】
- 既存の列名に既に含まれる情報
- 「販売月報」「エリア別」「一覧」のような、文書の種類・体裁を表すだけの汎用語
- ファイル管理上の記号（バージョン番号、年度以外の連番など）で表の値として
  意味を持たないもの
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
- 項目 X が既存の区分列（または他の抽出項目）Y を概念的に包含する
  （X が Y の上位区分にあたる）場合 → anchor=Y, position="before"
- 逆に X が Y に包含される（X が Y の下位区分にあたる）場合 →
  anchor=Y, position="after"
- 集計対象の数値そのものの種類・単位を表す項目（指標名に該当するもの）は、
  既存区分列のうち最も細かい（最も右にある）区分の直後、つまり値に最も
  近い位置に置く → anchor=既存区分列の最後の列名, position="after"
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


def apply_external_metadata(
    df: Any, items: List[Dict[str, Any]], dim_cols: List[str], axis_name: str
) -> Tuple[Any, List[str]]:
    """extract_external_metadata が抽出したメタデータ列を、各項目の anchor/position
    （既存区分列・集計軸・他項目との包含関係）に基づいて区分列の並びに差し込む
    （決定論処理）。

    dim_cols: 縦持ち変換後もそのまま残る既存の区分列（横持ち時の左→右の順）。
    axis_name: 集計軸列名（時系列等が畳まれる列。区分列より右に来る）。
    anchor がその時点で未配置の他メタ項目を指している場合は複数パスで解決する
    （チェーン解決）。anchor が無効・null・循環参照等でどうしても解決できない
    場合は集計軸の直前（＝区分列の末尾）に置く（安全側のフォールバック）。

    Returns:
      (派生列を挿入した df, 新しい区分列の並び順のリスト)
    """
    if not items:
        return df, list(dim_cols)

    # 1. 列名の重複回避（既存区分列・集計軸・相互）。anchor が他項目の
    # （採番前の）column_name を指している場合に実際の名前へ引き直せるよう
    # 対応表を作る。
    used = set(dim_cols) | {axis_name}
    name_map: Dict[str, str] = {}
    resolved: List[Dict[str, Any]] = []
    for item in items:
        base = item["column_name"]
        name = base
        n = 1
        while name in used:
            n += 1
            name = f"{base}_{n}"
        used.add(name)
        name_map[base] = name
        resolved.append({**item, "column_name": name})

    # 2. anchor / position を解決し、区分列の並び順を決める（チェーン解決）。
    ordered = list(dim_cols)
    pending = list(resolved)
    for _ in range(len(pending) + 1):
        if not pending:
            break
        still_pending: List[Dict[str, Any]] = []
        progressed = False
        for item in pending:
            raw_anchor = item.get("anchor")
            anchor = name_map.get(raw_anchor, raw_anchor) if raw_anchor else None
            position = item.get("position")
            if not anchor or anchor == axis_name or position not in ("before", "after"):
                ordered.append(item["column_name"])
                progressed = True
                continue
            if anchor in ordered:
                idx = ordered.index(anchor)
                ordered.insert(idx if position == "before" else idx + 1, item["column_name"])
                progressed = True
                continue
            # anchor が他の未配置メタ項目を指している → 次パスへ持ち越す
            still_pending.append(item)
        pending = still_pending
        if not progressed:
            break
    # 循環参照等でどうしても解決できなかった残りは末尾（集計軸の直前）に追加
    for item in pending:
        ordered.append(item["column_name"])

    # 3. 実データへ反映。新規列を追加してから、区分列を ordered の並びに
    # 並べ替える（区分列以外の列は元の順序を保つ）。
    out = df.copy()
    for item in resolved:
        out[item["column_name"]] = item["value"]
    other_cols = [c for c in df.columns if c not in dim_cols]
    out = out[ordered + other_cols]

    return out, ordered


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
