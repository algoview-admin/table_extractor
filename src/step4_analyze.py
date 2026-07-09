import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from itertools import combinations, groupby
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .models import (
    AIAnalysisResult,
    DetectedTable,
    IntegrationRecommendation,
    MasterTableInfo,
    SheetClassification,
    TableAnalysisResult,
)


# ---------------------------------------------------------------------------
# テーブル間合計関係の事前計算（旧 step4_aggregation.py）
# ---------------------------------------------------------------------------

def detect_sum_relations(tables: List[DetectedTable]) -> List[Dict]:
    """
    検証済みの全合計関係を返す:
      [{"parent_id": str, "child_ids": [str, ...], "match_ratio": float}, ...]

    関係の意味: parent.numeric_values ≈ 子テーブルの要素ごとの合計。
    同一の列スキーマを持つテーブルのみを比較する。
    1つの親に対して複数の関係（異なる集計軸）が存在する場合もある。
    """
    groups = _group_by_columns(tables)
    relations: List[Dict] = []
    for group in groups:
        if len(group) >= 3:
            relations.extend(_find_relations_in_group(group))
    return _drop_supersets(relations)


def format_relation_facts(relations: List[Dict], tables: List[DetectedTable]) -> str:
    """
    事前計算済みの合計関係をAI向けのプロンプトセクションとしてフォーマットする。
    関係が見つからない場合は空文字列を返す。
    """
    if not relations:
        return ""

    id_to_title = {t.table_id: (t.title or "") for t in tables}

    lines = [
        "=== 事前検証済み：数値合計関係 ===",
        "以下の親子関係はシステムが数値計算で確認済みです（AIによる推測不要）。",
        "この情報を最優先の根拠として granularity_level / parent_child_ids / is_minimum_granularity_candidate を決定してください。",
        "",
    ]
    for r in relations:
        pid = r["parent_id"]
        cids = r["child_ids"]
        pct = int(r["match_ratio"] * 100)
        p_label = f"{pid}" + (f"（{id_to_title[pid]}）" if id_to_title.get(pid) else "")
        c_parts = [f"{c}" + (f"（{id_to_title[c]}）" if id_to_title.get(c) else "") for c in cids]
        lines.append(f"  {p_label}  ≈  " + " + ".join(c_parts) + f"   [数値一致率 {pct}%]")

    lines += [
        "",
        "【この検証結果から導かれる判定基準】",
        "- 上記で「左辺（親）」になっているテーブル → granularity_level=summary, is_minimum_granularity_candidate=false",
        "- 上記で「右辺（子）」のみのテーブル（いずれの親にも属さない） → granularity_level=detail, is_minimum_granularity_candidate=true",
        "- 上記で「右辺（子）」であり且つ別の関係では「左辺（親）」のテーブル → granularity_level=summary, is_minimum_granularity_candidate=false",
    ]

    return "\n".join(lines)


def detect_sheet_levels(tables: List[DetectedTable]) -> List[Dict]:
    """
    同一スキーマのテーブルグループ内でsheet単位の数値合計を比較し、
    集計sheetの候補を特定する。
    """
    groups = _group_by_columns(tables)

    aggregate_votes: Dict[str, int] = {}
    smaller_peers: Dict[str, Set[str]] = {}

    for group in groups:
        sheet_sums: Dict[str, float] = {}
        for t in group:
            arr = _numeric_array(t)
            if arr is None:
                continue
            total = float(np.nansum(np.abs(arr)))
            if total < 1.0:
                continue
            sheet_sums[t.sheet_name] = sheet_sums.get(t.sheet_name, 0.0) + total

        if len(sheet_sums) < 3:
            continue

        sorted_items = sorted(sheet_sums.items(), key=lambda x: x[1], reverse=True)
        top_sheet, top_total = sorted_items[0]
        other_totals = [v for _, v in sorted_items[1:]]
        avg_other = sum(other_totals) / len(other_totals)

        if avg_other > 0 and top_total >= avg_other * 1.5 and len(other_totals) >= 2:
            aggregate_votes[top_sheet] = aggregate_votes.get(top_sheet, 0) + 1
            if top_sheet not in smaller_peers:
                smaller_peers[top_sheet] = set()
            for other_sheet, _ in sorted_items[1:]:
                smaller_peers[top_sheet].add(other_sheet)

    return [
        {
            "aggregate_sheet": sheet,
            "source_sheets": sorted(smaller_peers.get(sheet, set())),
        }
        for sheet, votes in aggregate_votes.items()
        if votes >= 1 and len(smaller_peers.get(sheet, set())) >= 2
    ]


def format_sheet_level_hints(hints: List[Dict]) -> str:
    """集計sheetのヒントをAI向けのプロンプトセクションとしてフォーマットする。"""
    if not hints:
        return ""

    lines = [
        "=== シート間集計構造（数値比較による推定） ===",
        "以下のシートは、同一構造を持つ複数の下位シートの数値合計シートと推定されます。",
        "【絶対ルール】これらのシートの全テーブルは granularity_level=summary かつ",
        "is_minimum_granularity_candidate=false と確定してください。",
        "「事前検証済み：数値合計関係」で右辺のみ・左辺のいずれの状況であっても、",
        "このシート判定を他の全ての推論・ルールより優先してください。",
        "",
    ]
    for h in hints:
        srcs = h["source_sheets"]
        src_str = "、".join(srcs[:6]) + ("…" if len(srcs) > 6 else "")
        lines.append(
            f"  【集計シート候補】 {h['aggregate_sheet']}"
            f" ← 集計元候補シート: {src_str}"
        )

    return "\n".join(lines) + "\n"


def _group_by_columns(tables: List[DetectedTable]) -> List[List[DetectedTable]]:
    """完全に同一の列名（同一スキーマ）を持つテーブルをグループ化する。"""
    buckets: Dict[Tuple, List[DetectedTable]] = {}
    for t in tables:
        if t.df is None or t.df.empty:
            continue
        key = tuple(str(c).strip() for c in t.df.columns)
        buckets.setdefault(key, []).append(t)
    return [g for g in buckets.values() if len(g) >= 2]


def _numeric_array(t: DetectedTable) -> Optional[np.ndarray]:
    if t.df is None:
        return None
    num = t.df.select_dtypes(include=[np.number])
    return num.values.astype(float) if not num.empty else None


def _total(t: DetectedTable) -> float:
    arr = _numeric_array(t)
    return float(np.nansum(np.abs(arr))) if arr is not None else 0.0


def _match_ratio(parent: DetectedTable, children: List[DetectedTable], tol: float = 0.03) -> float:
    """
    相対許容誤差 `tol` の範囲内で parent_value ≈ sum(child_values) を満たす、
    意味のある数値セルの割合を返す。
    形状が一致しない場合は -1.0 を返す。
    """
    p = _numeric_array(parent)
    if p is None:
        return -1.0

    child_arrs = [_numeric_array(c) for c in children]
    if any(a is None or a.shape != p.shape for a in child_arrs):
        return -1.0

    child_sum = np.zeros_like(p)
    for a in child_arrs:
        child_sum += np.nan_to_num(a, nan=0.0)

    p_clean = np.nan_to_num(p, nan=0.0)
    mask = np.abs(p_clean) > 0.5

    if mask.sum() == 0:
        return -1.0

    rel_diff = np.abs(p_clean[mask] - child_sum[mask]) / (np.abs(p_clean[mask]) + 1e-9)
    return float((rel_diff <= tol).mean())


def _find_relations_in_group(group: List[DetectedTable]) -> List[Dict]:
    """構造的に同一のテーブル集合内で全ての合計関係を検出する。"""
    MAX_SMALLER = 14
    MAX_K = 9
    COMBO_CAP = 50_000

    totals = {t.table_id: _total(t) for t in group}
    sorted_group = sorted(group, key=lambda t: totals[t.table_id])
    n = len(sorted_group)
    relations: List[Dict] = []

    for i in range(2, n):
        parent = sorted_group[i]
        p_total = totals[parent.table_id]
        if p_total < 1.0:
            continue

        smaller = [t for t in sorted_group[:i] if totals[t.table_id] > 0.1]
        if len(smaller) < 2:
            continue

        if len(smaller) > MAX_SMALLER:
            same = sorted(
                [t for t in smaller if t.sheet_name == parent.sheet_name],
                key=lambda t: totals[t.table_id], reverse=True,
            )
            other = sorted(
                [t for t in smaller if t.sheet_name != parent.sheet_name],
                key=lambda t: totals[t.table_id], reverse=True,
            )
            n_same = min(len(same), MAX_SMALLER)
            smaller = same[:n_same] + other[: MAX_SMALLER - n_same]

        max_k = min(len(smaller), MAX_K)

        while max_k > 2:
            n_combos = sum(math.comb(len(smaller), k) for k in range(2, max_k + 1))
            if n_combos <= COMBO_CAP:
                break
            max_k -= 1

        found_sets: List[Tuple[frozenset, float]] = []
        all_sheets_in_group = {t.sheet_name for t in group}

        for k in range(2, max_k + 1):
            for subset in combinations(smaller, k):
                s_total = sum(totals[t.table_id] for t in subset)
                if s_total > p_total * 1.15 or s_total < p_total * 0.72:
                    continue

                cross_kids = [t for t in subset if t.sheet_name != parent.sheet_name]
                if cross_kids:
                    if len(cross_kids) != len(subset):
                        continue
                    required = all_sheets_in_group - {parent.sheet_name}
                    if {t.sheet_name for t in cross_kids} != required:
                        continue

                ratio = _match_ratio(parent, list(subset))
                if ratio >= 0.88:
                    found_sets.append((frozenset(t.table_id for t in subset), ratio))

        minimal = [
            (s, r) for s, r in found_sets
            if not any(other < s for other, _ in found_sets)
        ]
        for child_set, ratio in minimal:
            relations.append({
                "parent_id": parent.table_id,
                "child_ids": sorted(child_set),
                "match_ratio": ratio,
            })

    return relations


def _drop_supersets(relations: List[Dict]) -> List[Dict]:
    """各親に対して上位集合となる子集合のみを除去し、全ての最小分解を保持する。"""
    result: List[Dict] = []
    keyfn = lambda r: r["parent_id"]
    for _, group_iter in groupby(sorted(relations, key=keyfn), key=keyfn):
        group_rels = list(group_iter)
        sets = [(frozenset(r["child_ids"]), r) for r in group_rels]
        result.extend(
            r for s, r in sets
            if not any(other < s for other, _ in sets)
        )
    return result


def _make_client() -> Tuple[Any, str]:
    """OPENAI_API_TYPE 環境変数に基づいて (client, model_or_deployment) を返す。

    Azure OpenAI を使用する場合は OPENAI_API_TYPE=azure を設定する。省略または "openai" を設定すると
    標準の OpenAI API が使用される。すべての認証情報は環境変数から読み込まれるため、
    関数の引数を通じてシークレットを渡す必要はない。
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

▼ 多段階集計のマスタ生成（集計構造を後から再現するための階層マスタ）【最重要】:
  「事前検証済み：数値合計関係」に集計構造がある場合（単層・多層ともに対象）、
  各集計テーブル（summary）の「直接の子テーブル全て（detail・summary を問わない）」を
  即時親ごとにグループ化し、グループごとに integration_recommendation を生成すること。
  ※ 多層ネスト（例: 上位集計→中間集計→詳細）の場合も各集計レベルの子を独立したグループとして扱う

  手順（全ての summary テーブルに対して繰り返す）:
    1. 「事前検証済み：数値合計関係」から各 summary テーブルの直接の子テーブルを全て取得
       （child_table_ids。detail・summary どちらも含める。is_minimum_granularity_candidate は問わない）
    2. 子テーブルが2件以上の場合、integration_recommendation を1件生成:
       - table_ids: 直接の子テーブル全てのIDリスト（summary/detailを混在させてよい）
       - new_column_names: ["区分"]（実際の意味に応じて設定）
       - new_column_values: {各テーブルID: [そのテーブルの識別値]}
       - axis_parent_table_ids: [即時親の summary テーブルID]
       - axis_parent_label_columns: [null]
    3. 子テーブルが1件のみの場合は省略

  ✅ 正例（多層構造）: 上位集計A → [中間集計B(summary), 詳細C(detail), 詳細D(detail)]
     → グループ: {B, C, D} → integration_recommendation 生成（summaryのBも含める）
     → さらに 中間集計B → [詳細E, 詳細F] も別途 integration_recommendation 生成
  ❌ 誤例: is_minimum_granularity_candidate=true の子のみをグループ化（summaryの中間集計が抜ける）

  ※ これにより「各集計レベルの子区分→直近上位集計」のマスタが集計レベルごとに生成され、
     多層の集計構造を後から完全に再現可能になる

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

【axis_parent_label_columns の名称統一ルール（必須）】
同一ファイル内で同一の親子軸（例: 下位単位→上位集計単位）に対する axis_parent_label_columns の列名は、
全ての integration_recommendation で完全に統一すること。
- 同じ親子関係に対して異なる名称（例: 「上位集計単位」と「集計先」）を混在させない
- 最初に決めた列名を全推奨で一貫して使用する
- 同一のマスタ内容を列名だけ変えて重複生成しない

【シート内統合が cross-sheet フェーズで2軸化されることへの対応（重要）】
同じ new_column_names を持つシート内統合（intra-sheet）が複数の下位シートで並行して存在し、
かつそれらが同一の集計シートに収束する場合、cross-sheet フェーズでその統合は2軸化される。
その前提で、シート内統合の axis_parent_table_ids を以下のルールに従って正確に設定すること:
- 内容区分軸（シート内の既存軸）の親: シート内の上位集計テーブルID（従来どおり）
- シート識別軸（cross-sheet で追加される軸）は cross-sheet フェーズで設定されるため、
  シート内推奨では null のままでよい
この設定が正確であることで、cross-sheet フェーズが元の axis_parent_table_ids[0] を正しく引き継げる。

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

    # AI がテーブル間の合計値を比較できるよう末尾行も含める
    tail_lines = ""
    if t.df is not None and len(t.df) > max_sample:
        tail = t.df.tail(max_tail).copy()
        for i, (_, row) in enumerate(tail.iterrows(), 1):
            items = "、".join(
                f"{k}={v}" for k, v in list(row.items())[:10] if v is not None and str(v).strip()
            )
            tail_lines += f"  末尾行{i}: {items}\n"

    title_line = f"  セクションタイトル: {s['title']}\n" if s.get("title") else ""

    # 合計・小計行と思われる行にフラグを立て、AI がテーブル比較時の
    # 構造整合性チェック（判定B）に利用できるようにする。
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

    # AI が集計構造を推定できるよう、テーブル外の注記も含める
    # （例: "※ TypeC-total = C-1 + C-2" はこのテーブルが中間集計であることを AI に伝える）。
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
# クロスシート統合候補の事前計算（Python 処理、AI 不使用）
# ---------------------------------------------------------------------------

def _format_merge_candidates(
    relations: List[Dict],
    tables: List[DetectedTable],
    aggregate_sheets: Optional[Set[str]] = None,
) -> str:
    """
    複数の支店シートにまたがって存在する同一タイプの親子構造を特定する。
    Phase 2 のクロスシートプロンプト用に整形されたセクションを返す。

    Python 側でこれらのグループを事前計算（数値確認済みの関係から）することで、
    Phase 2 は出力をフォーマットするだけでよく、グループを探索する必要がない。
    これにより AI が 2 軸統合を見落としたり部分的にしか生成しないことを防ぐ。
    """
    if not relations:
        return ""

    if aggregate_sheets is None:
        aggregate_sheets = set()

    id_to_table: Dict[str, DetectedTable] = {t.table_id: t for t in tables}
    id_to_title: Dict[str, str] = {t.table_id: (t.title or "").strip() for t in tables}

    # (parent_title, k) でグループ化 — 同タイトル + 同子数 = 同一タイプ
    by_type: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for r in relations:
        pid = r["parent_id"]
        parent = id_to_table.get(pid)
        if parent is None:
            continue
        ptitle = id_to_title.get(pid, "")
        if not ptitle:
            continue
        k = len(r["child_ids"])
        if k < 2:
            continue
        by_type[(ptitle, k)].append({
            "parent_id": pid,
            "child_ids": r["child_ids"],
            "sheet": parent.sheet_name,
        })

    cross_sheet_groups = []
    for (ptitle, k), entries in by_type.items():
        branch_entries = [e for e in entries if e["sheet"] not in aggregate_sheets]
        branch_sheets = {e["sheet"] for e in branch_entries}
        if len(branch_sheets) < 2:
            continue

        all_child_ids = [cid for e in branch_entries for cid in e["child_ids"]]
        parent_ids = [e["parent_id"] for e in branch_entries]
        example_titles = [id_to_title.get(cid, cid) for cid in branch_entries[0]["child_ids"]]

        # このタイプの集計シート親（axis_parent_table_ids のシート軸用）
        agg_parent_ids = [
            e["parent_id"] for e in entries if e["sheet"] in aggregate_sheets
        ]

        cross_sheet_groups.append({
            "parent_title": ptitle,
            "k": k,
            "sheets": sorted(branch_sheets),
            "parent_ids": parent_ids,
            "all_child_ids": all_child_ids,
            "example_titles": example_titles,
            "agg_parent_ids": agg_parent_ids,
        })

    if not cross_sheet_groups:
        return ""

    lines = [
        "=== 【必須】クロスシート2軸統合候補（数値確認済み・必ずそのまま使用すること） ===",
        "以下の各グループは、同名の親テーブルが複数シートに存在し、同数の子テーブルを持つことが",
        "数値計算で確認済みです。",
        "【★絶対ルール】各グループの「★全子テーブルIDs」をそのまま table_ids とした",
        "2軸 integration_recommendation を1グループにつき1件生成してください。",
        "シートごとに分割した1軸統合は禁止。部分的なシートのみの統合も禁止。",
        "",
    ]

    for i, g in enumerate(cross_sheet_groups, 1):
        lines.append(
            f"【統合グループ{i}】親タイプ「{g['parent_title']}」"
            f"（子{g['k']}個 × {len(g['sheets'])}シート）"
        )
        lines.append(f"  対象シート: {' / '.join(g['sheets'])}")
        lines.append(f"  各シートの親テーブルID: {', '.join(g['parent_ids'])}")
        if g["agg_parent_ids"]:
            lines.append(
                f"  集計シートの親テーブルID（シート識別軸の axis_parent_table_ids 候補）:"
                f" {', '.join(g['agg_parent_ids'])}"
            )
        lines.append(
            f"  ★全子テーブルIDs（これをそのまま table_ids に使うこと）:"
            f" {', '.join(g['all_child_ids'])}"
        )
        lines.append(f"  子テーブルの区分例: {', '.join(g['example_titles'])}")
        lines.append(
            f"  → new_column_names: [\"シート識別軸の名称\", \"内容区分軸の名称\"]"
            f"（実際の意味に合わせた適切な名称にすること）"
        )
        lines.append("")

    return "\n".join(lines)


def _compute_merge_recommendations(
    relations: List[Dict],
    tables: List[DetectedTable],
    aggregate_sheets: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    数値確認済みの関係から直接、2 軸統合推奨の dict を生成する — AI 不要。

    _format_merge_candidates と同じグループ化ロジックを使用する。同一の親タイトルが
    2 つ以上の非集計シートに同数の直接子テーブルを持つ場合にグループを形成する。
    各グループは全シートの子テーブルを対象とした、そのまま使用可能な
    integration_recommendation を 1 件生成する。

    Python 生成の推奨は（数値的根拠に基づき）正確であることが保証されており、
    AI 生成の支店別推奨より優先される。
    """
    if not relations:
        return []
    if aggregate_sheets is None:
        aggregate_sheets = set()

    id_to_table: Dict[str, DetectedTable] = {t.table_id: t for t in tables}
    id_to_title: Dict[str, str] = {t.table_id: (t.title or "").strip() for t in tables}

    by_type: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for r in relations:
        pid = r["parent_id"]
        parent = id_to_table.get(pid)
        if parent is None:
            continue
        ptitle = id_to_title.get(pid, "")
        if not ptitle:
            continue
        k = len(r["child_ids"])
        if k < 2:
            continue
        by_type[(ptitle, k)].append({
            "parent_id": pid,
            "child_ids": r["child_ids"],
            "sheet": parent.sheet_name,
        })

    recs: List[Dict] = []
    for i, ((ptitle, k), entries) in enumerate(by_type.items(), 1):
        branch_entries = [e for e in entries if e["sheet"] not in aggregate_sheets]
        branch_sheets = {e["sheet"] for e in branch_entries}
        if len(branch_sheets) < 2:
            continue

        all_child_ids = [cid for e in branch_entries for cid in e["child_ids"]]

        # new_column_values を構築: {child_id: [sheet_name, content_title]}
        new_col_vals: Dict[str, List[str]] = {}
        for e in branch_entries:
            for cid in e["child_ids"]:
                child = id_to_table.get(cid)
                if child:
                    new_col_vals[cid] = [child.sheet_name, id_to_title.get(cid, cid)]

        # axis_parent_table_ids:
        #   index 0 → シート軸の親（集計シートの同タイプテーブル、存在する場合）
        #   index 1 → 内容軸の親（最初の支店シートの親）
        agg_entries = [e for e in entries if e["sheet"] in aggregate_sheets]
        sheet_axis_parent: Optional[str] = agg_entries[0]["parent_id"] if agg_entries else None
        content_axis_parent: Optional[str] = (
            branch_entries[0]["parent_id"] if branch_entries else None
        )

        recs.append({
            "recommendation_id": f"IR_AUTO_{i}",
            "group_name": f"{ptitle} 支店横断統合",
            "description": (
                f"{len(branch_sheets)}シートに存在する「{ptitle}」の直接子テーブルを、"
                f"シート識別と内容区分の2軸で統合する。"
            ),
            "table_ids": all_child_ids,
            "new_column_names": ["シート識別", "内容区分"],
            "new_column_values": new_col_vals,
            "axis_parent_table_ids": [sheet_axis_parent, content_axis_parent],
            "axis_parent_label_columns": [None, None],
            "reasoning": (
                f"「{ptitle}」が{len(branch_sheets)}シートにわたって同一構造"
                f"（直接子{k}件）を持つことが数値計算で確認済み。"
                f"シート識別軸と内容区分軸の2軸統合を自動生成。"
            ),
        })

    return recs


# ---------------------------------------------------------------------------
# クロスシート統合プロンプト（チャンクモードの Phase 2 で使用）
# ---------------------------------------------------------------------------

CROSS_SHEET_SYSTEM = """あなたはExcelデータ構造の分析専門家です。
複数シートにまたがるテーブルのcross-sheet統合推奨のみを生成してください。

【★最優先タスク：クロスシート2軸統合候補への対応】
プロンプトに「【必須】クロスシート2軸統合候補」セクションが含まれる場合:
  1. 各グループの「★全子テーブルIDs」をそのまま table_ids として使用する
  2. 1グループ = 1つの integration_recommendation（シートごとに分割しない）
  3. new_column_names: [シート識別軸の名称, 内容区分軸の名称]（適切な名称を設定）
  4. axis_parent_table_ids: [集計シートの親テーブルID（あれば）, 各シートの親テーブルID（1つ）]
  5. これらのグループの生成を最優先し、省略・分割・部分的な生成は禁止
上記グループを全て生成した後、さらに追加すべきcross-sheet統合があれば生成する。

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

▼ シート内統合の自動2軸化（最重要・必ず実行）:
  シート内統合（参考情報）を走査し、複数の異なるシートで同一（または同等）の
  new_column_names を持つ統合が並行して存在し、かつそれらが同一の集計シートに
  集約される場合は、それら全テーブルを1件の2軸統合推奨にまとめること。
  - new_column_names: ["シート識別軸の名称", "元の内容区分軸の名称"]（適切な名称を設定）
  - new_column_values: 各テーブルに [そのテーブルのシート名, 元の区分値] の2要素リスト
  - axis_parent_table_ids: [集計シートの対応テーブルID, 元の axis_parent_table_ids[0]]
    ・集計シートの対応テーブルIDが不明な場合は null
    ・元の axis_parent_table_ids[0] がない場合は null
  - 元のシート内単軸推奨と重複させないこと（この2軸統合がその代替・上位推奨となる）
  - シート内推奨が1シート分だけ存在しても、同一構造が他シートにあれば必ず全シート分統合する

▼ 2軸統合のパターン（必須・省略不可）:
  シート内統合（intra-sheet）に含まれる各グループが複数シートにまたがって存在する場合、
  以下を必ず実行すること（省略・簡略化は禁止）:

  a) 同一の区分軸（new_column_names が同一または同等）でのシート内統合が
     複数シートに存在する場合 → 全シートの対応テーブルを1つのcross-sheet統合にまとめる
     ・部分的なシートのみ（例: 1支店分だけ）の統合は禁止
     ・シート内統合で1シート分しか提案されていない場合でも、
       同一構造のテーブルが他シートに存在するなら必ず全シート含む統合を生成

  b) 統合軸はシート識別軸（拠点・部門等）を既存の区分軸に追加した2軸以上で定義する
     ・例: シート内が「サービス区分」1軸 → cross-sheetで「シート識別軸」+「サービス区分」の2軸

  c) 各集計レベルの統合グループは独立して生成すること
     ・上位レベルのcross-sheet統合を生成した場合でも、下位レベルのcross-sheet統合も別途生成
     ・例: 上位サービス合計のcross-sheet統合とは別に、その下の詳細サービス区分のcross-sheet統合も生成

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

{merge_candidates}
{relation_facts}

上記を踏まえ、cross-sheet統合推奨のみをJSON形式で返してください。"""


# ---------------------------------------------------------------------------
# 429 エラー時に指数バックオフを行う API 呼び出しラッパー
# ---------------------------------------------------------------------------

def _call_api(
    client: Any,
    model: str,
    messages: List[Dict],
    max_completion_tokens: int,
    timeout: float = 360.0,
) -> str:
    """429 / レートリミットエラー時にリトライしながらチャット補完 API を呼び出す。"""
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
                    wait = 15 * (2 ** attempt)  # 15 秒、30 秒、60 秒
                    time.sleep(wait)
                    continue
            raise
    raise RuntimeError(
        f"API呼び出しが最大リトライ回数を超えました: {last_exc}"
    )


# ---------------------------------------------------------------------------
# 大規模プロンプト用の軽量テーブルサマリー
# ---------------------------------------------------------------------------

def _format_table_light(
    t: DetectedTable,
    granularity: str = "",
    display_name: str = "",
) -> str:
    """クロスシート統合プロンプトで使用する 1 行サマリー。"""
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
# バッチ単位のヘルパー（チャンクモードの Phase 1）
# 呼び出し元が len(batch_tables) ≤ _MAX_TABLES_PER_BATCH を保証することで、
# 出力が常に GPT-4o の 16,384 トークン上限に収まる。
# ---------------------------------------------------------------------------

def _analyze_sheet_batch(
    client: Any,
    model: str,
    batch_tables: List[DetectedTable],
    sheet_names: List[str],
    relation_facts: str,
    sheet_level_hints: str = "",
) -> Dict[str, Any]:
    """テーブルの 1 サブバッチを分析する。生の JSON dict を返す。"""
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

    # テーブル数が 10 件以下の場合の出力は約 3,000～6,000 トークンで 16,384 上限に
    # 十分収まるため、常に最大出力トークン数をリクエストする。
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
# マージヘルパー: Phase 1（シート別）と Phase 2（クロスシート）の結果を結合する
# ---------------------------------------------------------------------------

def _merge_raws(
    phase1_raws: List[Dict[str, Any]],
    phase2_raw: Dict[str, Any],
    tables: List[DetectedTable],
    auto_recs: Optional[List[Dict]] = None,
) -> "AIAnalysisResult":
    """Phase 1、Phase 2、および自動生成の推奨をマージする。

    優先順位 / 重複排除ルール:
      1. auto_recs   — Python 計算による 2 軸推奨（決定論的、常に最初に含める）
      2. phase2_recs — auto_recs にまだ含まれていない AI のクロスシート推奨
      3. phase1_recs — table_ids が auto_rec または phase2_rec のいずれのサブセットにも
                       なっていないバッチ別推奨（多支店 2 軸推奨に包含される
                       単一支店推奨は暗黙に除外される）
    """
    all_sheet_cls:    List[Dict] = []
    all_table_ana:    List[Dict] = []
    phase1_int_recs:  List[Dict] = []
    all_masters:      List[Dict] = []
    summaries:        List[str]  = []

    for raw in phase1_raws:
        if raw is None:
            continue
        all_sheet_cls.extend(raw.get("sheet_classifications", []))
        all_table_ana.extend(raw.get("table_analyses", []))
        phase1_int_recs.extend(raw.get("integration_recommendations", []))
        all_masters.extend(raw.get("master_tables", []))
        if raw.get("summary"):
            summaries.append(raw["summary"])

    phase2_int_recs = phase2_raw.get("integration_recommendations", [])
    auto = auto_recs or []

    # auto + phase2 推奨からスーパーセットプールを構築する。
    # table_ids がプールエントリのサブセットになっている Phase 1 推奨は冗長
    # （多支店推奨ですでにカバーされた単一支店スライスを表す）。
    superset_pool = [
        frozenset(r.get("table_ids", []))
        for r in (auto + phase2_int_recs)
        if r.get("table_ids")
    ]

    filtered_phase1 = [
        r for r in phase1_int_recs
        if not any(
            frozenset(r.get("table_ids", [])) <= pool_set
            for pool_set in superset_pool
        )
    ]

    # table_ids が auto 推奨と完全一致する Phase 2 推奨を除外する（完全重複）
    auto_id_sets = {frozenset(r.get("table_ids", [])) for r in auto}
    filtered_phase2 = [
        r for r in phase2_int_recs
        if frozenset(r.get("table_ids", [])) not in auto_id_sets
    ]

    all_int_recs = auto + filtered_phase2 + filtered_phase1

    # バッチ間での重複を避けるため推奨 ID を採番し直す
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
# 公開エントリポイント
# ---------------------------------------------------------------------------

# テーブル数が _SINGLE_CALL_LIMIT 以下 → 単一 API 呼び出し（レガシーパス）
_SINGLE_CALL_LIMIT = 50
# チャンクモードの各サブバッチはこの値を超えてはならない。
# テーブル 10 件 ≈ 3,000～6,000 出力トークン << GPT-4o の 16,384 トークン上限。
_MAX_TABLES_PER_BATCH = 10


def analyze_tables(
    tables: List[DetectedTable],
    sheet_names: List[str],
) -> "AIAnalysisResult":
    """OpenAI または Azure OpenAI を使用してテーブルの関係を分析する。

    テーブル数が多い場合（> _SINGLE_CALL_LIMIT 件）、コンテキストウィンドウと
    レートリミットの超過を避けるため、分析を 2 フェーズに分割する:

    Phase 1 – シート別並列呼び出し: 各シートのテーブルを個別リクエストで送信
              （1 呼び出しあたり最大 30 件、同時ワーカー 3 件）。
    Phase 2 – クロスシート統合: 全テーブルの 1 行サマリーを受け取り、
              複数シートにまたがる統合推奨を生成する単一の軽量呼び出し。
    """
    client, model = _make_client()

    # 全テーブルに対して集計関係を一度だけ事前計算する
    relations = detect_sum_relations(tables)
    relation_facts = format_relation_facts(relations, tables)
    sheet_hints_raw = detect_sheet_levels(tables)
    sheet_level_hints = format_sheet_level_hints(sheet_hints_raw)

    n = len(tables)
    if n <= _SINGLE_CALL_LIMIT:
        return _analyze_single(client, model, tables, sheet_names, relation_facts, sheet_level_hints)
    return _analyze_chunked(
        client, model, tables, sheet_names,
        relations, sheet_hints_raw,
        relation_facts, sheet_level_hints,
    )


def _analyze_single(
    client: Any,
    model: str,
    tables: List[DetectedTable],
    sheet_names: List[str],
    relation_facts: str,
    sheet_level_hints: str = "",
) -> "AIAnalysisResult":
    """単一呼び出しパス（_SINGLE_CALL_LIMIT 件以下のテーブル）。"""
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
    relations: List[Dict],
    sheet_hints_raw: List[Dict],
    relation_facts: str,
    sheet_level_hints: str = "",
) -> "AIAnalysisResult":
    """大規模テーブルセット向けの 2 フェーズチャンクパス。

    Phase 1: テーブルをシート別にグループ化し、各シートを _MAX_TABLES_PER_BATCH（最大 10 件）の
             サブバッチに分割する。全サブバッチは 3 つの同時ワーカーで並列実行される。
    Phase 2: 1 行サマリー（フルテーブルデータではない）を使用した単一の軽量
             クロスシート統合呼び出しで、複数シートにまたがる統合推奨を生成する。
    """
    # テーブルをシート別にグループ化し、各シートをサブバッチに分割する
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

    # ── Phase 2: クロスシート統合 ────────────────────────────────────────
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

    aggregate_sheets: Set[str] = {h["aggregate_sheet"] for h in sheet_hints_raw}

    # Python 計算による 2 軸推奨を生成する（決定論的、AI 不使用）
    auto_recs = _compute_merge_recommendations(relations, tables, aggregate_sheets)

    merge_candidates = _format_merge_candidates(relations, tables, aggregate_sheets)

    cs_user = CROSS_SHEET_USER.format(
        n_tables=len(tables),
        table_summaries=table_summaries,
        intra_recs_summary=intra_recs_summary,
        merge_candidates=merge_candidates,
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
        # クロスシート統合はベストエフォート。サブバッチ結果のみにフォールバックする
        phase2_raw = {"integration_recommendations": []}

    return _merge_raws(batch_raws, phase2_raw, tables, auto_recs=auto_recs)


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
        # display_name に常にシート名が含まれるようにする。
        # AI がサマリーテーブルのシート名を省略することがあるため、
        # UI が異なるシートのテーブルを一目で区別できるようここで補完する。
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

    # 全ての集計軸において最小粒度にあるテーブル。
    # 最小粒度のデータのみがパイプラインの後続に流れるよう、
    # 統合・マスタ提案をこれらのテーブルに限定する。
    min_gran_ids = {
        ta.table_id for ta in table_analyses if ta.is_minimum_granularity_candidate
    }

    integration_recs: List[IntegrationRecommendation] = []
    seen_relations: set = set()  # 重複排除のため (parent_id, frozenset(child_ids)) を追跡する
    for ir in raw.get("integration_recommendations", []):
        valid_tids = [x for x in ir.get("table_ids", []) if x in valid_ids]
        if len(valid_tids) < 2:
            continue

        # 階層が検出された場合は最小粒度の統合のみに限定する。
        # 最小粒度候補が存在しない場合（フラットファイル）は全て保持する。
        if min_gran_ids and not all(tid in min_gran_ids for tid in valid_tids):
            continue

        parent_tid = ir.get("parent_table_id")
        if parent_tid and parent_tid not in valid_ids:
            parent_tid = None

        # 重複排除: 同じ親 + 同じ子セット = 冗長
        relation_key = (parent_tid, frozenset(valid_tids))
        if relation_key in seen_relations:
            continue
        seen_relations.add(relation_key)

        # 旧形式（new_column_name 文字列）と新形式（new_column_names リスト）の両方をサポートする
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

        # 軸ごとの親情報（新しい多軸形式）
        raw_axis_parents = ir.get("axis_parent_table_ids", []) or []
        raw_axis_parent_cols = ir.get("axis_parent_label_columns", []) or []
        axis_parents: List[Optional[str]] = [
            (p if (p and p in valid_ids) else None) for p in raw_axis_parents
        ]
        axis_parent_cols: List[Optional[str]] = [
            (c if c else None) for c in raw_axis_parent_cols
        ]
        # col_names の長さに合わせてパディングする
        while len(axis_parents) < len(col_names):
            axis_parents.append(None)
        while len(axis_parent_cols) < len(col_names):
            axis_parent_cols.append(None)

        # 後方互換: axis_parent_table_ids がないが旧 parent_table_id がある場合、axis 0 に使用する
        if not any(axis_parents) and parent_tid:
            axis_parents[0] = parent_tid
            old_pcol = ir.get("parent_label_column") or None
            if old_pcol and not axis_parent_cols[0]:
                axis_parent_cols[0] = old_pcol

        # 軸リストから後方互換用の単軸フィールドを導出する
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
