import json
import os
from typing import Any, Dict, List, Tuple

from .models import (
    AIAnalysisResult,
    DetectedTable,
    IntegrationRecommendation,
    MasterTableInfo,
    SheetClassification,
    TableAnalysisResult,
)
from .aggregation_detector import detect_sum_relations, format_relation_facts


def _make_client() -> Tuple[Any, str]:
    """Return (client, model_or_deployment) based on OPENAI_API_TYPE env var.

    Set OPENAI_API_TYPE=azure to use Azure OpenAI; omit or set to "openai" for
    the standard OpenAI API.  All credentials are read from environment variables
    so no secrets need to be passed through function arguments.
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

SYSTEM_PROMPT = """あなたはExcelデータ構造の分析専門家です。
Excelファイルから自動検出されたテーブル情報を精密に分析し、
データの意味・階層関係・最小粒度・マスタデータ・統合可能性を判断します。

必ず以下のJSON形式のみで回答してください（説明文は不要）:
{
  "sheet_classifications": [
    {
      "sheet_name": "シート名",
      "is_data_sheet": true,
      "description": "シートの内容説明"
    }
  ],
  "table_analyses": [
    {
      "table_id": "テーブルID（入力と完全一致）",
      "display_name": "わかりやすい日本語表示名",
      "description": "テーブルの内容・目的の説明",
      "granularity_level": "detail または summary または master または reference または unknown",
      "is_master_table": false,
      "parent_table_ids": [],
      "child_table_ids": [],
      "similar_table_ids": [],
      "is_minimum_granularity_candidate": false,
      "recommended_for_extraction": true,
      "has_external_info": false,
      "external_info_description": null,
      "reasoning": "この判断をした理由（日本語）"
    }
  ],
  "integration_recommendations": [
    {
      "recommendation_id": "IR_1",
      "group_name": "統合後のテーブル表示名",
      "description": "統合の内容説明",
      "table_ids": ["統合対象テーブルIDのリスト"],
      "new_column_names": ["識別軸1の列名", "識別軸2の列名（1軸のみなら要素1つ）"],
      "new_column_values": {"テーブルID": ["軸1の値", "軸2の値（1軸なら要素1つ）"]},
      "axis_parent_table_ids": ["軸1の親テーブルID（なければnull）", "軸2の親テーブルID（なければnull）"],
      "axis_parent_label_columns": ["軸1の上位区分列名", "軸2の上位区分列名"],
      "reasoning": "統合を推奨する理由（日本語）"
    }
  ],
  "master_tables": [
    {
      "table_id": "マスタとなるテーブルID",
      "key_column": "キーとなる列名",
      "referenced_by": ["参照元テーブルIDのリスト"],
      "description": "このマスタテーブルの説明"
    }
  ],
  "summary": "ファイル全体の分析サマリー（2〜3文、日本語）"
}

【granularity_levelの定義】
- detail: 全ての集計軸において最下位レベルのデータ。これ以上分解されない最小単位
- summary: 1つ以上の集計軸で上位集計されているデータ（組織合計・サービス合計・時間合計など）
- master: コード・名称・属性等のマスタテーブル
- reference: 参照用補助データ（説明・注記等）
- unknown: 判断不能

【多次元集計軸の識別（最重要）】
Excelデータは複数の独立した集計軸を同時に持つことがある。全ての軸を列挙して判定すること。

▼ 代表的な集計軸の例:
  - 組織軸: 本社 > 事業部 > 支社/支店 > 営業所
  - サービス/商品軸: サービス合計/全商品計 > 個別サービス/個別商品
  - 時間軸: 年次 > 四半期 > 月次 > 日次

▼ 最小粒度の判定ルール（最重要）:
  is_minimum_granularity_candidate=true の条件:「全ての集計軸において最下位レベルであること」
  判定方法: 「事前検証済み：数値合計関係」セクションで、そのテーブルが誰かの「右辺（子）」に
  含まれている場合、別の親テーブル関係で確認する。「左辺（親）」として出現しなければ最小粒度。
  出現すれば中間層。

【表間の階層関係判定（3観点）】
以下の3観点を順に適用して表間の集計元→集計結果の関係を判定してください。
観点1・2はAIが推論し、観点3はシステムが事前に計算した結果を利用します。
3観点すべてが整合する場合に親子関係を確定し、矛盾する場合は観点3（数値一致）を優先します。

▼ 観点1：名称関連性
- シート名・セクションタイトルに集計を示す語（「合計」「総計」「小計」「計」「全体」「全社」「部門計」等）がある表は上位集計の候補
- 別の表タイトルの上位概念・集約を示すシート名や接頭辞の差（例：「月次」→「四半期」→「年次」、「詳細」→「サマリー」）

▼ 観点2：構造整合性
- 行見出し・列見出し（期間軸・指標等）の構造が対応している表は同一系列の候補
- 上位表の行が下位表の「計」「小計」「合計」行に対応する場合、上下関係として採用する
- 下位表は上位表より見出し階層が深い（追加の分類軸・区分が存在する）
- 下位表にのみ存在する追加軸を集約すると上位表の構造になる場合、集計元→集計結果とみなす
- 見出し比較では形式的な差異だけで関係を否定しないこと（正規化指針を参照）

  ＜正規化指針：構造差を吸収し見落としを防ぐ＞
  - 空白セルの見出し継承：上行・左列から値を引き継ぐ（マージセル展開後として解釈）
  - 省略された上位・中間階層の補完：上位カテゴリが省略されていても同一系列と判断できる場合がある
  - 計・小計行の有無：下位表の「計」行が上位表の対応行に集約されている構造は親子関係の根拠になる
  - 見出し階層の深さの違い：下位表が多段階見出しを持ち上位表が1段階に集約される場合も親子関係
  - 追加分類軸の集約：下位表にのみ存在する分類軸を集約した場合に上位表が再現されるなら集計元

▼ 観点3：数値一致性（事前検証済み）
「事前検証済み：数値合計関係」セクションに記載された関係はシステムが数値計算で確認済みです。
AIによる推測不要であり、観点1・2の推論より優先して採用してください。
- 「左辺（親）」のテーブル → granularity_level=summary, is_minimum_granularity_candidate=false
- 「右辺（子）」のみのテーブル → 全集計軸で最下位なら is_minimum_granularity_candidate=true
- 「右辺（子）」かつ別の関係で「左辺（親）」のテーブル → 中間層、is_minimum_granularity_candidate=false
- 上位テーブルの child_table_ids=[下位テーブルIDリスト]、下位テーブルの parent_table_ids=[上位テーブルID]

【similar_table_idsとintegration_recommendationsのルール（最重要）】
- similar_table_ids には「同一階層レベル」のテーブルのみを含める
  ✅ 正例: 同じ階層単位（例: 同レベルの複数拠点・複数月・複数商品）→ 互いにsimilar
  ❌ 誤例: 上位集計テーブルと下位明細テーブルを同一similar_table_idsに含める（階層が異なる）
- integration_recommendations は同一階層レベルのテーブルのみを統合対象にする
  - summaryレベルのテーブルとdetailレベルのテーブルを混ぜて統合してはいけない
  - 上位集計テーブル（summary）は統合推奨から除外する
- similar_table_idsが2件以上ある場合のみ統合推奨を出す

【new_column_names / new_column_values の多軸ルール（最重要）】
テーブルが「支店×サービス」「支店×商品」など複数の独立した分類軸で識別される場合、
絶対に1つの列に結合してはいけない。軸ごとに別々の列を使うこと。

- new_column_names: 識別軸ごとに1つの列名を並べたリスト
  - 1軸（例: 月次）  → ["月"]
  - 2軸（例: 支店×サービス） → ["支店", "サービス区分"]
- new_column_values: 各テーブルIDに対し、new_column_namesと同じ長さのリストで値を指定
  - 1軸例: {"T2": ["1月"], "T3": ["2月"]}
  - 2軸例: {"東京北T2": ["東京北支店", "サービスA"], "東京北T3": ["東京北支店", "サービスB"]}

❌ 禁止例: new_column_names=["支店_サービス区分"], values={"T2": ["東京北支店_サービスA"]}
✅ 正例:   new_column_names=["支店", "サービス区分"], values={"T2": ["東京北支店", "サービスA"]}

【階層対応マスタの自動生成（axis_parent_table_ids / axis_parent_label_columns）】
統合の各識別軸（new_column_names の各要素）に対して、上位集計テーブルが存在する場合は
axis_parent_table_ids / axis_parent_label_columns にその情報をセットする。

  - axis_parent_table_ids: new_column_namesと同じ長さのリスト。各軸の親テーブルID（なければnull）
  - axis_parent_label_columns: 同じ長さのリスト。各軸の上位区分を示す列名（なければnull）

設定するとシステムが各軸ごとに「子区分値 → 親区分値」のマスタを自動生成する。

例（支店 × サービス区分 の2軸統合の場合）:
  new_column_names: ["支店", "サービス区分"]
  axis_parent_table_ids: ["東京事業部_T2", "東京事業部_T1"]
    → 軸0(支店): 東京事業部_T2（事業部シートにある同構造テーブル）が親
    → 軸1(サービス区分): 東京事業部_T1（全サービス合計テーブル）が親
  axis_parent_label_columns: ["事業部", "サービス集計"]
  これにより2つのマスタが生成される:
    マスタ1: 支店 × 事業部（支店→事業部の対応）
    マスタ2: サービス区分 × サービス集計（個別サービス→合計区分の対応）

親テーブル選定のルール:
  - シート軸（支店など）の親: そのシート軸の上位組織を示すシート上の任意テーブル
    → マスタの親値にはそのシートのシート名を使用
  - タイトル軸（サービスなど）の親: セクションタイトルレベルの集計テーブル（合計行）
    → マスタの親値にはそのテーブルのセクションタイトルを使用
親が存在しない軸はnullをセットすること。

【その他の分析基準】
- is_minimum_granularity_candidate: 全ての集計軸（組織・サービス/商品・時間など）で最下位レベルのテーブルのみtrue。1つでも上位集計軸があればfalse
- has_external_info: 上位テーブルにしか存在しない情報列がある場合true
- recommended_for_extraction: 説明書きシート・意味のない小テーブル・summaryテーブル（詳細で代替可能な場合）はfalseも可"""

USER_PROMPT = """以下のExcelファイルから検出されたテーブルを分析してください。

=== シート一覧 ===
{sheet_list}

=== 検出テーブル詳細 ===
{table_details}

{relation_facts}

上記を踏まえ、指定のJSON形式で分析結果を返してください。"""


def _format_table_detail(
    t: DetectedTable,
    max_sample: int = 5,
    max_tail: int = 3,
) -> str:
    s = t.to_summary_dict(max_sample_rows=max_sample)
    cols_str = "、".join(s["columns"])
    if s.get("columns_truncated"):
        cols_str += f"  ほか{t.col_count - 20}列"

    sample_lines = ""
    for i, row in enumerate(s["sample_data"], 1):
        items = "、".join(f"{k}={v}" for k, v in list(row.items())[:10])
        sample_lines += f"  先頭行{i}: {items}\n"

    # Include tail rows so the AI can compare totals across tables
    tail_lines = ""
    if t.df is not None and len(t.df) > max_sample:
        tail = t.df.tail(max_tail).copy()
        for i, (_, row) in enumerate(tail.iterrows(), 1):
            items = "、".join(
                f"{k}={v}" for k, v in list(row.items())[:10] if v is not None and str(v).strip()
            )
            tail_lines += f"  末尾行{i}: {items}\n"

    title_line = f"  セクションタイトル: {s['title']}\n" if s.get("title") else ""

    # Flag rows that appear to be total/subtotal rows so the AI can use them for
    # structural consistency checks (判定B) when comparing tables.
    total_line = ""
    if t.df is not None and not t.df.empty:
        _TOTAL_MARKERS = ("計", "合計", "総計", "小計", "Total", "total", "SUM", "sum")
        total_labels = []
        for _, row in t.df.iterrows():
            row_head = " ".join(str(v) for v in list(row)[:3] if v is not None and str(v).strip())
            if any(m in row_head for m in _TOTAL_MARKERS):
                total_labels.append(row_head[:20])
        if total_labels:
            total_line = f"  計・合計行の見出し例: {', '.join(dict.fromkeys(total_labels))}\n"

    return (
        f"[{s['table_id']}]\n"
        f"  シート: {s['sheet_name']}  位置: {s.get('position', '不明')}\n"
        f"{title_line}"
        f"  サイズ: {s['row_count']}行 × {s['col_count']}列\n"
        f"  列名: {cols_str}\n"
        f"{total_line}"
        f"{sample_lines}"
        f"{tail_lines}"
    )


def analyze_tables(
    tables: List[DetectedTable],
    sheet_names: List[str],
) -> AIAnalysisResult:
    """Analyze table relationships using OpenAI or Azure OpenAI (via OPENAI_API_TYPE env var)."""
    client, model = _make_client()

    # Scale down per-table detail verbosity for large table sets so the prompt
    # stays manageable.  The AI still gets enough structure to classify tables.
    n = len(tables)
    max_sample = 3 if n > 25 else (4 if n > 15 else 5)
    max_tail   = 2 if n > 25 else 3

    sheet_list = "\n".join(f"- {name}" for name in sheet_names)
    table_details = "\n\n".join(
        _format_table_detail(t, max_sample=max_sample, max_tail=max_tail) for t in tables
    )

    relations = detect_sum_relations(tables)
    relation_facts = format_relation_facts(relations, tables)

    user_msg = USER_PROMPT.format(
        sheet_list=sheet_list,
        table_details=table_details,
        relation_facts=relation_facts,
    )

    # Scale completion token budget: larger files may produce larger JSON responses.
    completion_tokens = min(16384, max(8192, n * 300))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_completion_tokens=completion_tokens,
        timeout=360.0,  # 6-minute hard limit; prevents indefinite hangs
    )

    choice = response.choices[0]
    content = choice.message.content or ""

    if choice.finish_reason == "length":
        raise ValueError(
            f"GPT レスポンスがトークン上限に達して途中で切れました。"
            f"テーブル数を減らすか、より少ないシートを対象にしてください。"
            f"(finish_reason=length, 文字数={len(content)})"
        )

    try:
        raw: Dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"GPT のレスポンスが不正な JSON です: {e}\n"
            f"レスポンス先頭500文字: {content[:500]}"
        ) from e

    return _parse_response(raw, tables)


def _parse_response(raw: Dict[str, Any], tables: List[DetectedTable]) -> AIAnalysisResult:
    valid_ids = {t.table_id for t in tables}

    sheet_classifications = [
        SheetClassification(
            sheet_name=sc.get("sheet_name", ""),
            is_data_sheet=sc.get("is_data_sheet", True),
            description=sc.get("description", ""),
        )
        for sc in raw.get("sheet_classifications", [])
    ]

    table_analyses: List[TableAnalysisResult] = []
    for ta in raw.get("table_analyses", []):
        tid = ta.get("table_id", "")
        if tid not in valid_ids:
            continue
        table_analyses.append(
            TableAnalysisResult(
                table_id=tid,
                display_name=ta.get("display_name", tid),
                description=ta.get("description", ""),
                granularity_level=ta.get("granularity_level", "unknown"),
                is_master_table=ta.get("is_master_table", False),
                parent_table_ids=[x for x in ta.get("parent_table_ids", []) if x in valid_ids],
                child_table_ids=[x for x in ta.get("child_table_ids", []) if x in valid_ids],
                similar_table_ids=[x for x in ta.get("similar_table_ids", []) if x in valid_ids],
                is_minimum_granularity_candidate=ta.get("is_minimum_granularity_candidate", False),
                recommended_for_extraction=ta.get("recommended_for_extraction", False),
                has_external_info=ta.get("has_external_info", False),
                external_info_description=ta.get("external_info_description"),
                reasoning=ta.get("reasoning", ""),
            )
        )

    # Tables that are at the finest granularity across every aggregation axis.
    # Integration / master suggestions are restricted to these so that only the
    # minimum-granularity data flows through the rest of the pipeline.
    min_gran_ids = {
        ta.table_id for ta in table_analyses if ta.is_minimum_granularity_candidate
    }

    integration_recs: List[IntegrationRecommendation] = []
    seen_relations: set = set()  # Track (parent_id, frozenset(child_ids)) to deduplicate
    for ir in raw.get("integration_recommendations", []):
        valid_tids = [x for x in ir.get("table_ids", []) if x in valid_ids]
        if len(valid_tids) < 2:
            continue

        # Restrict to minimum-granularity integrations (when a hierarchy was detected).
        # If no minimum-granularity candidate exists (flat file), keep all.
        if min_gran_ids and not all(tid in min_gran_ids for tid in valid_tids):
            continue

        parent_tid = ir.get("parent_table_id")
        if parent_tid and parent_tid not in valid_ids:
            parent_tid = None

        # Deduplication: same parent + same child set = redundant
        relation_key = (parent_tid, frozenset(valid_tids))
        if relation_key in seen_relations:
            continue
        seen_relations.add(relation_key)

        # Support both old (new_column_name str) and new (new_column_names list) formats
        raw_names = ir.get("new_column_names", None)
        if raw_names and isinstance(raw_names, list) and raw_names:
            col_names = [str(n) for n in raw_names]
        else:
            col_names = [ir.get("new_column_name", "区分")]

        raw_vals = ir.get("new_column_values", {})
        multi_vals: Dict[str, List[str]] = {}
        for k, v in raw_vals.items():
            if k not in valid_ids:
                continue
            if isinstance(v, list):
                multi_vals[k] = [str(x) for x in v]
            else:
                multi_vals[k] = [str(v)]

        # Per-axis parent info (new multi-axis format)
        raw_axis_parents = ir.get("axis_parent_table_ids", []) or []
        raw_axis_parent_cols = ir.get("axis_parent_label_columns", []) or []
        axis_parents: List[Optional[str]] = [
            (p if (p and p in valid_ids) else None) for p in raw_axis_parents
        ]
        axis_parent_cols: List[Optional[str]] = [
            (c if c else None) for c in raw_axis_parent_cols
        ]
        # Pad to match col_names length
        while len(axis_parents) < len(col_names):
            axis_parents.append(None)
        while len(axis_parent_cols) < len(col_names):
            axis_parent_cols.append(None)

        # Backward compat: if no axis_parent_table_ids but old parent_table_id exists, use it for axis 0
        if not any(axis_parents) and parent_tid:
            axis_parents[0] = parent_tid
            old_pcol = ir.get("parent_label_column") or None
            if old_pcol and not axis_parent_cols[0]:
                axis_parent_cols[0] = old_pcol

        # Derive backward-compat single-axis fields from axis lists
        compat_parent_tid = axis_parents[0] if axis_parents else parent_tid
        compat_parent_col = axis_parent_cols[0] if axis_parent_cols else (ir.get("parent_label_column") or None)

        integration_recs.append(
            IntegrationRecommendation(
                recommendation_id=ir.get("recommendation_id", "IR_?"),
                group_name=ir.get("group_name", "統合テーブル"),
                description=ir.get("description", ""),
                table_ids=valid_tids,
                new_column_name=col_names[0],
                new_column_values={k: v[0] if v else "" for k, v in multi_vals.items()},
                reasoning=ir.get("reasoning", ""),
                parent_table_id=compat_parent_tid,
                parent_label_column=compat_parent_col,
                new_column_names=col_names,
                new_column_multi_values=multi_vals,
                axis_parent_table_ids=axis_parents,
                axis_parent_label_columns=axis_parent_cols,
            )
        )

    master_tables: List[MasterTableInfo] = []
    for mt in raw.get("master_tables", []):
        tid = mt.get("table_id", "")
        if tid not in valid_ids:
            continue
        master_tables.append(
            MasterTableInfo(
                table_id=tid,
                key_column=mt.get("key_column", ""),
                referenced_by=[x for x in mt.get("referenced_by", []) if x in valid_ids],
                description=mt.get("description", ""),
            )
        )

    return AIAnalysisResult(
        sheet_classifications=sheet_classifications,
        table_analyses=table_analyses,
        integration_recommendations=integration_recs,
        master_tables=master_tables,
        summary=raw.get("summary", ""),
        raw_response=raw,
    )
