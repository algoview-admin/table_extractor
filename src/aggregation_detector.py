"""
数値に基づいてテーブル間の階層的な合計関係を事前計算するモジュール。

「親」テーブルとは、同一の列スキーマを持つ他のテーブルの部分集合の要素ごとの合計に
（許容誤差の範囲内で）等しい数値セルを持つテーブルである。
これらの関係はアルゴリズムにより検出され、検証済みの事実としてAIプロンプトに
注入されるため、AIがキーワードや名称から推測する必要がなくなる。
"""

import math
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .models import DetectedTable


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    mask = np.abs(p_clean) > 0.5  # 意味のある値を持つセルのみを比較する

    if mask.sum() == 0:
        return -1.0

    rel_diff = np.abs(p_clean[mask] - child_sum[mask]) / (np.abs(p_clean[mask]) + 1e-9)
    return float((rel_diff <= tol).mean())


def _find_relations_in_group(group: List[DetectedTable]) -> List[Dict]:
    """構造的に同一のテーブル集合内で全ての合計関係を検出する。

    大規模グループでの組み合わせ爆発を防ぐためのパフォーマンス制限:
    - MAX_SMALLER: 親ごとの子候補を合計値の大きい上位 MAX_SMALLER 件に制限する
      （直接の子はほぼ常に大きい小計を持つ）
    - MAX_K: 実際の集計チェーンで MAX_K を超える直接の子は稀
    - COMBO_CAP: 探索空間がこの値を超える場合は max_k を動的に削減する
    """
    MAX_SMALLER = 14    # 親ごとの候補プール数
    MAX_K = 9           # 実際には直接の子が9を超えることは稀
    COMBO_CAP = 50_000  # 親ごとの組み合わせ上限; 超過時は max_k を削減

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

        # 候補プールを制限: 親と同一sheetのテーブルを優先し、残りのスロットは
        # 他sheetのうち合計値の大きいテーブルで埋める。
        # これを行わないと、複数sheetのグループで他sheetの大きな中間集計が
        # 同一sheet内の小さいが正しい直接の子を押しのけてしまう
        # （例: 2つの小さい子が兄弟ブランチのテーブルに追い出されて
        # 5つの子関係の検出が失敗するケース）。
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

        # 組み合わせ総数が COMBO_CAP 以内に収まるまで max_k を動的に削減する。
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
                # 合計値レベルでの高速枝刈り（以前より狭いウィンドウ）
                if s_total > p_total * 1.15 or s_total < p_total * 0.72:
                    continue

                # sheet横断カバレッジのガード:
                # 有効な関係は (a) 親と同一sheet内のみ、または
                # (b) 他の全sheetにそれぞれ1つの子が存在する純粋なsheet横断、のいずれかに限る。
                # これにより、ブランチレベルのリーフテーブルが兄弟ブランチの一部における
                # 同じリーフテーブルの数値合計に偶然一致してしまう誤検知を防ぐ
                # （小さい合計値でよく起こるケース）。
                cross_kids = [t for t in subset if t.sheet_name != parent.sheet_name]
                if cross_kids:
                    # 同一sheet内と他sheetの混在を除外する。
                    if len(cross_kids) != len(subset):
                        continue
                    # 部分的なsheet横断を除外: 他の全sheetが必ず含まれていなければならない。
                    required = all_sheets_in_group - {parent.sheet_name}
                    if {t.sheet_name for t in cross_kids} != required:
                        continue

                ratio = _match_ratio(parent, list(subset))
                if ratio >= 0.88:
                    found_sets.append((frozenset(t.table_id for t in subset), ratio))

        # 最小の有効部分集合のみを保持する（他の見つかった部分集合の上位集合を除去する）
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


def detect_sheet_levels(tables: List[DetectedTable]) -> List[Dict]:
    """
    同一スキーマのテーブルグループ内でsheet単位の数値合計を比較し、
    集計sheetの候補を特定する。

    sheetの合計数値が少なくとも2つのスキーマグループにわたって
    他のsheetの平均値の1.5倍以上になる場合、そのsheetを集計候補としてフラグを立てる。
    これにより、元sheetの一部しか存在しない場合でも、組織的なロールアップ
    （例: 支店sheetを集計する部門sheet）を検出できる。

    {"aggregate_sheet": str, "source_sheets": [str]} の dict のリストを返す。
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
    """集計sheetのヒントをAI向けのプロンプトセクションとしてフォーマットする。

    集計sheetが検出されなかった場合は空文字列を返す。
    """
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


def _drop_supersets(relations: List[Dict]) -> List[Dict]:
    """
    各親に対して、同じ親に対する別の有効な集合の真の上位集合となる子集合のみを除去する。
    これにより、異なる集計軸を表すものも含め、全ての最小分解を保持する。
    例えば、地理軸（3支店）と商品軸（4サービス種別）はカーディナリティが異なっても
    両方が保持される。

    以前の動作（最小サイズのみ保持）では、大きい軸の関係を誤って破棄していた。
    その結果、AIが以下のような多階層構造を見逃していた:
      • C-1 + C-2 → TypeC-total → Next-total → ServiceA-total
    ここで TypeC-total は子（Next-total の）であると同時に親（C-1/C-2 の）でもある。
    """
    from itertools import groupby

    result: List[Dict] = []
    keyfn = lambda r: r["parent_id"]
    for _, group_iter in groupby(sorted(relations, key=keyfn), key=keyfn):
        group_rels = list(group_iter)
        sets = [(frozenset(r["child_ids"]), r) for r in group_rels]
        # 同じ親に対して r の子の真部分集合を持つ別の関係が存在しない場合のみ r を保持する
        # （つまり、r が別の関係の上位集合でない場合）。
        result.extend(
            r for s, r in sets
            if not any(other < s for other, _ in sets)
        )
    return result
