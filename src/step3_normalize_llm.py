"""
ステップ3 テーブル正規化モジュール（LLM使用処理）。

処理概要: Transpose（行列逆転）の検出と変換など、意味判定にLLMが必要な
          テーブル整形処理を扱う。
          Step3 と Step4（src/step4_analyze.py）は処理の性質が異なり
          （Step3 の出力が Step4 の入力になる関係）依存を持たせたくない
          ため、LLM クライアント生成・API 呼び出しは step4_analyze.py の
          ものを共有せず、このファイルに独立実装する。
          決定論的処理（正規表現・語彙辞書のみで完結する処理）は
          src/step3_normalize_determ.py を参照。
入力    : DetectedTable.df（step2_detect が構築した生 DataFrame）
出力    : 変換済み DataFrame、transpose_info（{entity_axis_name, reasoning}）
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
