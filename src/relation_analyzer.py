import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    AIAnalysisResult,
    DetectedTable,
    IntegrationRecommendation,
    MasterTableInfo,
    SheetClassification,
    TableAnalysisResult,
)
from .aggregation_detector import (
    detect_sheet_levels,
    detect_sum_relations,
    format_relation_facts,
    format_sheet_level_hints,
)


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
      "display_name": "シート名を前置した表示名（形式: 「シート名 テーブル内容」必須。例: 「横浜支店 サービスA合計」）",
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
  - サービス/商品軸: サービス合計 > 中間集計 > 個別サービス（複数レベル可）
  - 時間軸: 年次 > 四半期 > 月次 > 日次

▼ テーブル外注記（「テーブル外注記:」として入力される）の活用（最優先）:
  テーブルに付随する注記にはそのテーブルの集計構造が直接記されていることが多い。
  注記の内容を最優先の根拠として granularity_level と child_table_ids を決定すること。
  - 「※ A＋B＋C の合計」「※ Xを除く」などの注記がある → そのテーブルは summary
  - 「合計＝A＋B＋C」形式 → そのテーブルは A, B, C を子テーブルとする中間集計層
  - 注記内の名称と「事前検証済み：数値合計関係」を照合して child_table_ids を特定すること

▼ 多段階（ネスト）集計の識別（最重要）:
  集計は複数レベルでネストされることがある。汎用的な構造例（固有名詞は使わない）:
    詳細X + 詳細Y → 中間集計Z → 上位集計W → 全体集計V
  判定は「事前検証済み：数値合計関係」の左辺/右辺の出現有無に従う:
  - 詳細X, 詳細Y: 右辺のみ（どの親の左辺にも登場しない）
    → granularity_level=detail, is_minimum_granularity_candidate=true
  - 中間集計Z: 左辺（詳細X,詳細Yの集計親）かつ右辺（上位集計Wの子）
    → granularity_level=summary, is_minimum_granularity_candidate=false
  - 上位集計W, 全体集計V: 左辺（集計親として登場）
    → granularity_level=summary, is_minimum_granularity_candidate=false

  【summary/detail 判定の大原則（誤分類防止）】
  summary の条件: 「このテーブル自体が、ファイル内の他テーブルを集計した結果である」こと。
  「上位組織のシートに同一構造のテーブルが存在する」だけでは summary にしない。
    正しい判定: 集計元（下位シートの詳細データ）→ detail
                集計結果（上位集計シートが下位シートを集計したもの）→ summary

  【判定の優先順位（番号順に厳守。上位ルールが適用されたら下位は無視）】

  ★ 優先度A（最高・絶対優先）:
    「シート間集計構造（推定）」の集計シート候補に含まれるシートのテーブル
    → granularity_level=summary かつ is_minimum_granularity_candidate=false
    ※ 「事前検証済み：数値合計関係」での左辺・右辺の出現有無に関わらず、このルールを最優先。
       右辺のみのテーブルであっても集計シート内なら必ず summary とする。

  ★ 優先度B（数値確定・疑いなし）:
    「事前検証済み：数値合計関係」で左辺（親）に登場するテーブル
    → granularity_level=summary

  ★ 優先度C（cross-sheet整合推定）:
    「事前検証済み：数値合計関係」に、このテーブルと論理的に同一（別シートの同一論理テーブル）
    が左辺（親）として登場しており、かつこのテーブルが「事前検証済み」で右辺のみと確認されていない場合
    → granularity_level=summary
    ※ 論理的同一の判定: 別シートで末尾（または中核部分）の名称が一致するテーブルが親として登場する場合。
       例: 「シートX_Z合計」が親として登場 → このシートの「シートY_Z合計」も summary

  ★ 優先度D（名称/構造推論）:
    観点1・2（名称・構造）から集計結果と明確に判断できる場合
    → granularity_level=summary

  ★ 優先度E（数値確定・detail）:
    「事前検証済み：数値合計関係」で右辺のみ（どの親の左辺にも登場しない）かつ非集計シートのテーブル
    → granularity_level=detail かつ is_minimum_granularity_candidate=true
    ※ テーブル名に「合計」「計」「集計」等を含んでいてもこのルールが適用される
       （右辺のみで確認済みであれば、最小粒度の集計名テーブルが存在することがある）

  ★ 優先度F（フォールバック）:
    上記A〜Eのいずれにも明確に該当しない場合のみ、観点1・2・3を総合的に判断する。
    → unknown は最終手段。名称・構造・注記から少しでも推定できる場合は summary または detail を選択。

▼ 最小粒度の判定ルール（最重要）:
  is_minimum_granularity_candidate=true の条件:
    「ファイル内の他テーブルを集計・合算した結果ではなく、最も詳細な観測/記録データである」こと。
  判定方法:
    - 優先度A（集計シート候補）のテーブル → NOT最小粒度（summary）
    - 「事前検証済み：数値合計関係」で左辺（親）に登場 → NOT最小粒度（summary）
    - 「事前検証済み：数値合計関係」で右辺のみ かつ 非集計シート → 最小粒度候補
  重要: 組織軸の上位シートに同一構造の集計テーブルが存在する場合でも、
    そのシートの元となる詳細データ（下位シート等の明細）は is_minimum_granularity_candidate=true。
    「集計結果か否か」で判定し、「組織の最下位か否か」で判定しない。

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
- 「右辺（子）」のみのテーブル（どの親の左辺にも登場しない）かつ集計シートでないシートのテーブル
  → is_minimum_granularity_candidate=true
  ※ 「シート間集計構造（推定）」の集計シート候補に含まれるシートのテーブル
    → is_minimum_granularity_candidate=false（右辺のみでも集計シートのテーブルは除外）
  ※ 「名称に合計・計・集計を含む」は is_minimum_granularity_candidate=false の根拠にしない
     relation_factsで左辺に登場せず、かつ集計シートに属さなければ is_minimum_granularity_candidate=true
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

▼ 多段階集計のマスタ生成（集計構造を後から再現するための階層マスタ）:
  「事前検証済み：数値合計関係」に多段階のネスト集計がある場合、
  各集計テーブル（summary）の「直接の子テーブルのうち is_minimum_granularity_candidate=true のもの」を
  即時親ごとにグループ化し、グループごとに integration_recommendation を生成すること。

  手順:
    1. 各 summary テーブルの直接の子 detail テーブルをグループ化（グループキー = その summary テーブルID）
    2. グループメンバーが2件以上の場合、integration_recommendation を生成:
       - table_ids: グループ内の detail テーブルIDリスト
       - new_column_names: ["区分"]（汎用列名。実際の意味に応じて適宜設定）
       - new_column_values: {各テーブルID: [そのテーブルの識別値]}
       - axis_parent_table_ids: [即時親の summary テーブルID]
       - axis_parent_label_columns: [null]（なければnull）
    3. 全ての summary テーブルに対してステップ1・2を繰り返す

  ※ 1件のみのグループは、同じ即時親の他グループと合算するか省略
  ※ これにより「詳細区分→直近上位集計」のマスタが集計レベルごとに生成され、
     全体の集計構造を後から再現可能になる

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

【マスタ生成の基本方針】
統合の識別軸ごとに、その軸の「下位区分 → 上位区分」の対応マスタを1件生成する。
識別軸が複数ある場合は、上位集計テーブルが存在する軸すべてに親テーブルを設定すること。

例（2軸統合の場合）:
  new_column_names: ["軸A", "軸B"]
  axis_parent_table_ids: ["軸Aの上位集計テーブルID", "軸Bの上位集計テーブルID"]
    → 軸A: 上位集計テーブルが親 → 生成マスタ1: 軸A下位区分 × 軸A上位区分 の対応表
    → 軸B: 上位集計テーブルが親 → 生成マスタ2: 軸B下位区分 × 軸B上位区分 の対応表
  axis_parent_label_columns: ["軸Aの上位区分列名", "軸Bの上位区分列名"]

親テーブルの選定ルール（汎用）:
  - シート名で識別される軸（例: シートごとに分かれたデータ単位）の親:
    そのシート群の上位集計シートにある同構造テーブル。
    マスタの親値にはその上位シート名を使用。
  - セクションタイトルで識別される軸（例: 同一シート内の複数テーブル）の親:
    そのセクションの上位集計テーブル（「集計」「合計」行をまとめたテーブル）。
    マスタの親値にはそのテーブルのセクションタイトルを使用。
  - 多段階集計がある場合は、統合対象の詳細データに対して直近の上位テーブルを親に指定。
  - 上位集計テーブルが存在しない軸は null をセット。

【その他の分析基準】
- is_minimum_granularity_candidate: ファイル内の他テーブルを集計・合算した結果ではない最も詳細なデータのみtrue。「事前検証済み」で左辺に登場せず、かつサービス/商品軸でも最下位（子テーブルなし）であること。上位組織シートに集計テーブルが存在しても、その元となる詳細データはtrue
- has_external_info: 上位テーブルにしか存在しない情報列がある場合true
- recommended_for_extraction: 説明書きシート・意味のない小テーブル・summaryテーブル（詳細で代替可能な場合）はfalseも可"""

USER_PROMPT = """以下のExcelファイルから検出されたテーブルを分析してください。

=== シート一覧 ===
{sheet_list}

{sheet_level_hints}
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

    # Include out-of-table notes so the AI can infer aggregation structure
    # (e.g. "※ TypeC-total = C-1 + C-2" tells the AI this table is a mid-level aggregate).
    notes_line = ""
    for note in (t.notes or [])[:3]:
        short = note[:120] + ("…" if len(note) > 120 else "")
        notes_line += f"  テーブル外注記: {short}\n"

    return (
        f"[{s['table_id']}]\n"
        f"  シート: {s['sheet_name']}  位置: {s.get('position', '不明')}\n"
        f"{title_line}"
        f"  サイズ: {s['row_count']}行 × {s['col_count']}列\n"
        f"  列名: {cols_str}\n"
        f"{total_line}"
        f"{sample_lines}"
        f"{tail_lines}"
        f"{notes_line}"
    )


# ---------------------------------------------------------------------------
# Cross-sheet synthesis prompts (used in chunked mode, Phase 2)
# ---------------------------------------------------------------------------

CROSS_SHEET_SYSTEM = """あなたはExcelデータ構造の分析専門家です。
複数シートにまたがるテーブルのcross-sheet統合推奨のみを生成してください。

必ず以下のJSON形式のみで回答してください（説明文は不要）:
{
  "integration_recommendations": [
    {
      "recommendation_id": "IR_CS_1",
      "group_name": "統合後テーブルの表示名",
      "description": "統合の内容説明",
      "table_ids": ["統合対象テーブルIDのリスト（全シートの対応テーブル）"],
      "new_column_names": ["識別軸の列名（例: 拠点名）"],
      "new_column_values": {"テーブルID": ["値（例: 東京拠点）"]},
      "axis_parent_table_ids": [null],
      "axis_parent_label_columns": [null],
      "reasoning": "統合を推奨する理由"
    }
  ]
}

【重要】シート内の統合（intra-sheet）は既に処理済みです。
異なるシートに存在する同一構造・同一目的のテーブルをまとめる統合推奨のみを出力してください。

【統合軸の一貫性確保（最重要）】
シート内統合（intra-sheet）の参考情報を確認し、全テーブルで一貫した統合軸を定義してください。

▼ 2軸統合のパターン（推奨）:
  シート内で何らかの区分軸での統合が提案されている場合（例: 商品区分・サービス区分・期間等）:
  → クロスシート統合でシート識別軸（例: 拠点・部門・担当等）を追加し、2軸統合を提案
  → これにより全データが1つの統合テーブルにまとまる

▼ axis_parent_table_ids の設定（各軸のマスタを生成するために必須）:
  cross-sheet 統合の各識別軸（new_column_names の各要素）に対して、
  上位集計テーブルが存在する場合は axis_parent_table_ids に設定する。

  - シート識別軸（例: 拠点・部門等）の親:
    「シート間集計構造（推定）」の集計シート候補にある同構造テーブルのID。
    これにより「子単位 → 上位集計単位」のマスタが生成される。
  - 内容区分軸（例: 商品・サービス区分等）の親:
    シート内統合参考情報に axis_parent_table_ids が設定されている場合はその値を使用。
    サービス/商品の上位集計テーブル（summary）のIDを設定する。
    これにより「詳細区分 → 上位集計区分」のマスタが生成される。

  両軸に親を設定することで、シート識別軸と内容区分軸それぞれのマスタが両方生成される。

▼ 統合の一貫性チェック:
  同一シートに同一テーブル構造が存在する場合は全て同じ統合グループに含めること。
  異なるバッチで別々に提案されていても、最終的に1つの統合テーブルになるよう統合すること。"""

CROSS_SHEET_USER = """以下は各シートのテーブル分析結果サマリーです。
複数シートにまたがる同一構造のテーブルがある場合、cross-sheet統合推奨を生成してください。

=== 全テーブル一覧（{n_tables}件） ===
{table_summaries}

=== シート内統合（参考：既処理済み） ===
{intra_recs_summary}

{relation_facts}

上記を踏まえ、cross-sheet統合推奨のみをJSON形式で返してください。"""


# ---------------------------------------------------------------------------
# API call wrapper with exponential backoff for 429 errors
# ---------------------------------------------------------------------------

def _call_api(
    client: Any,
    model: str,
    messages: List[Dict],
    max_completion_tokens: int,
    timeout: float = 360.0,
) -> str:
    """Call the chat completion API with retry on 429 / rate-limit errors."""
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
                    f"GPT レスポンスがトークン上限に達して途中で切れました。"
                    f"(finish_reason=length)"
                )
            return choice.message.content or ""
        except Exception as exc:
            err = str(exc)
            if "429" in err or "too_many_requests" in err.lower() or "rate_limit" in err.lower():
                last_exc = exc
                if attempt < 3:
                    wait = 15 * (2 ** attempt)  # 15 s, 30 s, 60 s
                    time.sleep(wait)
                    continue
            raise
    raise RuntimeError(
        f"API呼び出しが最大リトライ回数を超えました: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Lightweight table summary for large-set prompts
# ---------------------------------------------------------------------------

def _format_table_light(
    t: DetectedTable,
    granularity: str = "",
    display_name: str = "",
) -> str:
    """Single-line summary used in cross-sheet synthesis prompts."""
    cols = [str(c) for c in (t.df.columns.tolist()[:8] if t.df is not None else [])]
    cols_str = "、".join(cols)
    if t.df is not None and len(t.df.columns) > 8:
        cols_str += f"…ほか{len(t.df.columns) - 8}列"
    title = f" [{t.title}]" if t.title else ""
    gran = f" ({granularity})" if granularity else ""
    name = f" 表示名:「{display_name}」" if display_name else ""
    return (
        f"[{t.table_id}] シート:{t.sheet_name}{title}{gran}{name} | "
        f"{t.row_count}行×{t.col_count}列 | 列: {cols_str}"
    )


# ---------------------------------------------------------------------------
# Per-batch helper (Phase 1 in chunked mode)
# Caller ensures len(batch_tables) ≤ _MAX_TABLES_PER_BATCH so output always
# fits within GPT-4o's 16 384-token output ceiling.
# ---------------------------------------------------------------------------

def _analyze_sheet_batch(
    client: Any,
    model: str,
    batch_tables: List[DetectedTable],
    sheet_names: List[str],
    relation_facts: str,
    sheet_level_hints: str = "",
) -> Dict[str, Any]:
    """Analyze a single sub-batch of tables. Returns the raw JSON dict."""
    n = len(batch_tables)
    max_sample = 3 if n > 7 else 5
    max_tail   = 2 if n > 7 else 3

    sheet_list = "\n".join(f"- {name}" for name in sheet_names)
    table_details = "\n\n".join(
        _format_table_detail(t, max_sample=max_sample, max_tail=max_tail)
        for t in batch_tables
    )

    user_msg = USER_PROMPT.format(
        sheet_list=sheet_list,
        sheet_level_hints=sheet_level_hints,
        table_details=table_details,
        relation_facts=relation_facts,
    )

    # Always request the maximum output tokens because ≤ 10 tables generate
    # roughly 3 000–6 000 output tokens — well within 16 384 limit.
    content = _call_api(
        client, model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_completion_tokens=16384,
    )

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"バッチ分析レスポンスが不正な JSON です: {e}\n先頭300文字: {content[:300]}"
        ) from e


# ---------------------------------------------------------------------------
# Merge helper: combine Phase 1 (per-sheet) + Phase 2 (cross-sheet) results
# ---------------------------------------------------------------------------

def _merge_raws(
    phase1_raws: List[Dict[str, Any]],
    phase2_raw: Dict[str, Any],
    tables: List[DetectedTable],
) -> "AIAnalysisResult":
    all_sheet_cls:   List[Dict] = []
    all_table_ana:   List[Dict] = []
    all_int_recs:    List[Dict] = []
    all_masters:     List[Dict] = []
    summaries:       List[str]  = []

    for raw in phase1_raws:
        if raw is None:
            continue
        all_sheet_cls.extend(raw.get("sheet_classifications", []))
        all_table_ana.extend(raw.get("table_analyses", []))
        all_int_recs.extend(raw.get("integration_recommendations", []))
        all_masters.extend(raw.get("master_tables", []))
        if raw.get("summary"):
            summaries.append(raw["summary"])

    all_int_recs.extend(phase2_raw.get("integration_recommendations", []))

    # Renumber recommendation IDs to avoid duplicates across batches
    for i, rec in enumerate(all_int_recs, 1):
        rec["recommendation_id"] = f"IR_{i}"

    merged: Dict[str, Any] = {
        "sheet_classifications": all_sheet_cls,
        "table_analyses": all_table_ana,
        "integration_recommendations": all_int_recs,
        "master_tables": all_masters,
        "summary": " / ".join(summaries[:3]) if summaries else "複数シートの分析完了",
    }

    return _parse_response(merged, tables)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# ≤ _SINGLE_CALL_LIMIT tables → single API call (legacy path)
_SINGLE_CALL_LIMIT = 50
# Each sub-batch in chunked mode must not exceed this.
# 10 tables ≈ 3 000–6 000 output tokens << GPT-4o 16 384-token output limit.
_MAX_TABLES_PER_BATCH = 10


def analyze_tables(
    tables: List[DetectedTable],
    sheet_names: List[str],
) -> "AIAnalysisResult":
    """Analyze table relationships using OpenAI or Azure OpenAI.

    For large table sets (> _SINGLE_CALL_LIMIT tables) the analysis is split
    into two phases to avoid exceeding context-window and rate limits:

    Phase 1 – Per-sheet parallel calls: each sheet's tables are sent in a
              separate request (≤ 30 tables per call, 3 concurrent workers).
    Phase 2 – Cross-sheet synthesis: a single lightweight call that receives
              one-line summaries of all tables and produces integration
              recommendations that span multiple sheets.
    """
    client, model = _make_client()

    # Pre-compute aggregation relationships once for all tables
    relations = detect_sum_relations(tables)
    relation_facts = format_relation_facts(relations, tables)
    sheet_level_hints = format_sheet_level_hints(detect_sheet_levels(tables))

    n = len(tables)
    if n <= _SINGLE_CALL_LIMIT:
        return _analyze_single(client, model, tables, sheet_names, relation_facts, sheet_level_hints)
    return _analyze_chunked(client, model, tables, sheet_names, relation_facts, sheet_level_hints)


def _analyze_single(
    client: Any,
    model: str,
    tables: List[DetectedTable],
    sheet_names: List[str],
    relation_facts: str,
    sheet_level_hints: str = "",
) -> "AIAnalysisResult":
    """Single-call path (≤ _SINGLE_CALL_LIMIT tables)."""
    n = len(tables)
    max_sample = 3 if n > 25 else (4 if n > 15 else 5)
    max_tail   = 2 if n > 25 else 3

    sheet_list = "\n".join(f"- {name}" for name in sheet_names)
    table_details = "\n\n".join(
        _format_table_detail(t, max_sample=max_sample, max_tail=max_tail) for t in tables
    )

    user_msg = USER_PROMPT.format(
        sheet_list=sheet_list,
        sheet_level_hints=sheet_level_hints,
        table_details=table_details,
        relation_facts=relation_facts,
    )

    completion_tokens = min(16384, max(8192, n * 300))
    content = _call_api(
        client, model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_completion_tokens=completion_tokens,
    )

    try:
        raw: Dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"GPT のレスポンスが不正な JSON です: {e}\n"
            f"レスポンス先頭500文字: {content[:500]}"
        ) from e

    return _parse_response(raw, tables)


def _analyze_chunked(
    client: Any,
    model: str,
    tables: List[DetectedTable],
    sheet_names: List[str],
    relation_facts: str,
    sheet_level_hints: str = "",
) -> "AIAnalysisResult":
    """Two-phase chunked path for large table sets.

    Phase 1: Tables are grouped by sheet and each sheet is split into
             sub-batches of _MAX_TABLES_PER_BATCH (≤ 10).  All sub-batches
             run in parallel with 3 concurrent workers.
    Phase 2: A single lightweight cross-sheet synthesis call uses one-line
             summaries (not full table data) to produce integration
             recommendations that span multiple sheets.
    """
    # Group tables by sheet, then split each sheet into sub-batches
    by_sheet: Dict[str, List[DetectedTable]] = {}
    for t in tables:
        by_sheet.setdefault(t.sheet_name, []).append(t)

    batches: List[List[DetectedTable]] = []
    for sheet_tables in by_sheet.values():
        for i in range(0, len(sheet_tables), _MAX_TABLES_PER_BATCH):
            batches.append(sheet_tables[i : i + _MAX_TABLES_PER_BATCH])

    batch_raws: List[Optional[Dict[str, Any]]] = [None] * len(batches)

    def _run_batch(args: Tuple[int, List[DetectedTable]]) -> Tuple[int, Dict[str, Any]]:
        idx, batch_tables = args
        raw = _analyze_sheet_batch(
            client, model, batch_tables, sheet_names, relation_facts, sheet_level_hints
        )
        return idx, raw

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_run_batch, (i, batch)): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            idx, raw = future.result()
            batch_raws[idx] = raw

    # ── Phase 2: cross-sheet synthesis ────────────────────────────────────────
    ta_by_id: Dict[str, Dict] = {}
    for raw in batch_raws:
        if raw is None:
            continue
        for ta in raw.get("table_analyses", []):
            ta_by_id[ta.get("table_id", "")] = ta

    table_summaries = "\n".join(
        _format_table_light(
            t,
            granularity=ta_by_id.get(t.table_id, {}).get("granularity_level", ""),
            display_name=ta_by_id.get(t.table_id, {}).get("display_name", ""),
        )
        for t in tables
    )

    intra_recs: List[Dict] = []
    for raw in batch_raws:
        if raw:
            intra_recs.extend(raw.get("integration_recommendations", []))

    intra_summary_lines = [
        f"  [{r.get('recommendation_id')}] {r.get('group_name')} : "
        + " + ".join(r.get("table_ids", [])[:4])
        + ("…" if len(r.get("table_ids", [])) > 4 else "")
        for r in intra_recs[:30]
    ]
    intra_recs_summary = "\n".join(intra_summary_lines) or "（なし）"

    cs_user = CROSS_SHEET_USER.format(
        n_tables=len(tables),
        table_summaries=table_summaries,
        intra_recs_summary=intra_recs_summary,
        relation_facts=relation_facts,
    )

    try:
        cs_content = _call_api(
            client, model,
            [
                {"role": "system", "content": CROSS_SHEET_SYSTEM},
                {"role": "user",   "content": cs_user},
            ],
            max_completion_tokens=16384,
        )
        phase2_raw: Dict[str, Any] = json.loads(cs_content)
    except Exception:
        # Cross-sheet synthesis is best-effort; fall back to sub-batch results only
        phase2_raw = {"integration_recommendations": []}

    return _merge_raws(batch_raws, phase2_raw, tables)


def _parse_response(raw: Dict[str, Any], tables: List[DetectedTable]) -> AIAnalysisResult:
    valid_ids = {t.table_id for t in tables}
    id_to_table = {t.table_id: t for t in tables}

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
        # Ensure display_name always includes the sheet name.
        # The AI sometimes omits it for summary tables; patch it here so the UI
        # can always distinguish tables from different sheets at a glance.
        dn = ta.get("display_name") or tid
        dt = id_to_table.get(tid)
        if dt and dt.sheet_name and dt.sheet_name not in dn:
            dn = f"{dt.sheet_name} {dn}"
        table_analyses.append(
            TableAnalysisResult(
                table_id=tid,
                display_name=dn,
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
