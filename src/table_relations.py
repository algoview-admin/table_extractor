"""
Pre-compute hierarchical (sum) relationships between tables based on numeric values.

A "parent" table is one whose numeric cell values equal (within tolerance) the
element-wise sum of a subset of other tables that share the same column schema.
These relationships are detected algorithmically and injected into the AI prompt
as verified facts, so the AI does not need to infer them from keywords or names.
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import DetectedTable


def detect_sum_relations(tables: List[DetectedTable]) -> List[Dict]:
    """
    Return all verified sum relationships:
      [{"parent_id": str, "child_ids": [str, ...], "match_ratio": float}, ...]

    A relation means: parent.numeric_values ≈ element-wise sum of children.
    Only tables that share identical column schemas are compared.
    Multiple relations per parent are possible (different aggregation axes).
    """
    groups = _group_by_columns(tables)
    relations: List[Dict] = []
    for group in groups:
        if len(group) >= 3:
            relations.extend(_find_relations_in_group(group))
    return _drop_supersets(relations)


def format_relation_facts(relations: List[Dict], tables: List[DetectedTable]) -> str:
    """
    Format pre-computed sum relations as a prompt section for the AI.
    Returns empty string if no relations found.
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
    """Group tables that have the exact same column names (same schema)."""
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
    Returns the fraction of non-trivial numeric cells where
    parent_value ≈ sum(child_values), within relative tolerance `tol`.
    Returns -1.0 if shapes are incompatible.
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
    mask = np.abs(p_clean) > 0.5  # only compare cells with meaningful values

    if mask.sum() == 0:
        return -1.0

    rel_diff = np.abs(p_clean[mask] - child_sum[mask]) / (np.abs(p_clean[mask]) + 1e-9)
    return float((rel_diff <= tol).mean())


def _find_relations_in_group(group: List[DetectedTable]) -> List[Dict]:
    """
    Find all sum relationships within a set of structurally identical tables.
    A parent must be numerically larger than any single child.
    """
    totals = {t.table_id: _total(t) for t in group}
    sorted_group = sorted(group, key=lambda t: totals[t.table_id])
    n = len(sorted_group)
    relations: List[Dict] = []

    for i in range(2, n):  # candidate parent must have at least 2 smaller tables
        parent = sorted_group[i]
        p_total = totals[parent.table_id]
        if p_total < 1.0:
            continue

        smaller = [t for t in sorted_group[:i] if totals[t.table_id] > 0.1]
        if len(smaller) < 2:
            continue

        max_k = min(len(smaller), 8)
        found_sets: List[Tuple[frozenset, float]] = []

        for k in range(2, max_k + 1):
            for subset in combinations(smaller, k):
                s_total = sum(totals[t.table_id] for t in subset)
                # Fast total-level pruning before expensive cell-level check
                if s_total > p_total * 1.20 or s_total < p_total * 0.55:
                    continue
                ratio = _match_ratio(parent, list(subset))
                if ratio >= 0.88:
                    found_sets.append((frozenset(t.table_id for t in subset), ratio))

        # Keep only minimal valid subsets (remove supersets of other found subsets)
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
    """
    For each parent, keep only the smallest-cardinality valid child sets.
    Larger subsets are redundant (cross-axis mixed decompositions) and add noise.
    If multiple subsets share the minimum size, all are kept — each likely
    represents a distinct aggregation axis (e.g. geographic vs. product axis).
    """
    from itertools import groupby

    result: List[Dict] = []
    keyfn = lambda r: r["parent_id"]
    for _, group_iter in groupby(sorted(relations, key=keyfn), key=keyfn):
        group_rels = list(group_iter)
        min_size = min(len(r["child_ids"]) for r in group_rels)
        result.extend(r for r in group_rels if len(r["child_ids"]) == min_size)
    return result
