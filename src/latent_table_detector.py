"""
Latent table detection from out-of-table notes and annotations.

Notes/annotations adjacent to detected tables often hint at tables that
exist elsewhere in the file but were not captured — either because they are
in a section that was not parsed, or because they lie outside the visible
scroll range of a single session.

This module extracts "entity references" (potential table names) from any
trailing note, cross-references them against the detected table list, and
returns a LatentTableProposal for every case where:
  - At least one referenced entity matches a detected table   (confirming relevance)
  - At least one referenced entity has NO match               (the latent table)

Note-type coverage (rule-based, no API required):
  1. Aggregation    : "A、B、Cの合計"  →  C might be missing
  2. Enumeration    : "以下4種: A・B・C・D"  →  D might be missing
  3. Exclusion      : "AとBを除いた値"  →  A / B might be separate tables
  4. Reference      : "「A」および「B」を参照"  →  A / B might be tables
  5. Parallel list  : any 、-separated list where some items match tables

For complex natural-language notes that fall outside the patterns above,
callers can optionally pass them to an LLM (not done here; kept pure rule-based
to avoid extra latency).
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .models import DetectedTable, DerivedLatentTable


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LatentTableProposal:
    """A proposed table that is referenced in a note but not yet detected."""

    proposal_id: str
    source_table_id: str        # Table whose trailing note triggered this proposal
    source_title: str           # Human-readable title of the source table
    note_text: str              # Full original note text
    note_type: str              # Inferred note type (aggregation / enumeration / reference / general)
    all_referenced: List[str]   # All entity names extracted from the note
    detected_table_ids: List[str]   # table_ids of entities that matched detected tables
    detected_names: List[str]       # Referenced names that matched detected tables
    missing_names: List[str]        # Referenced names with NO detected-table match
    reasoning: str


# ---------------------------------------------------------------------------
# Note-text normalisation & entity extraction
# ---------------------------------------------------------------------------

_NOTE_PREFIX_RE = re.compile(r"^[※＊\*注）注\)（注\(注意＜<]+\s*")
_SPLIT_RE = re.compile(r"[、，,・／/＋+及びおよびとや]+")

# Keywords that suggest an aggregation/summation relationship
_AGG_KEYWORDS = ("合計", "合算", "総計", "小計", "集計", "の計", "sum")
# Suffixes to strip before splitting (aggregation variants)
_AGG_SUFFIX_RE = re.compile(
    r"[のをにおける]*(合計|合算|総計|小計|集計|計)\s*$", re.IGNORECASE
)
# Keywords that suggest an exclusion relationship
_EXCL_KEYWORDS = ("除く", "除いた", "除外", "を除")
# Keywords that suggest a reference/cross-reference relationship
_REF_KEYWORDS = ("参照", "参考", "を見る", "については", "に記載")


def _detect_note_type(text: str) -> str:
    """Classify the note's semantic type for display / filtering purposes."""
    if any(k in text for k in _AGG_KEYWORDS):
        return "aggregation"
    if any(k in text for k in _EXCL_KEYWORDS):
        return "exclusion"
    if any(k in text for k in _REF_KEYWORDS):
        return "reference"
    return "general"


def _extract_entities(note_text: str) -> List[str]:
    """Extract candidate entity names from a note.

    Handles the following patterns:
      - 「quoted names」
      - (parenthesised names)
      - Delimiter-separated lists after stripping aggregation suffixes
    """
    text = _NOTE_PREFIX_RE.sub("", note_text).strip()

    candidates: List[str] = []

    # 1. 「Japanese quote」extraction
    candidates.extend(re.findall(r"「([^」]{2,40})」", text))

    # 2. (parenthesised) extraction
    candidates.extend(re.findall(r"[（(]([^）)]{2,40})[）)]", text))

    # 3. Delimiter-separated enumeration
    #    Strip aggregation suffix so "AとBの合計" → "AとB" before split
    clean = _AGG_SUFFIX_RE.sub("", text).strip()
    # Also strip exclusion/reference tails
    clean = re.sub(r"[をにの](除く|除いた|除外|参照|参考|含む|合わせた|まとめた).*$", "", clean).strip()
    parts = _SPLIT_RE.split(clean)
    candidates.extend(p.strip() for p in parts if 2 <= len(p.strip()) <= 60)

    # Deduplicate while preserving order; filter very short/noisy tokens
    _NOISE_RE = re.compile(r"^(その|この|以下|上記|なお|ただし|また|※|注|合計|合算|小計|総計|集計)$")
    seen = set()
    result = []
    for c in candidates:
        c = c.strip()
        if c and c not in seen and not _NOISE_RE.match(c) and len(c) >= 2:
            seen.add(c)
            result.append(c)

    return result


# ---------------------------------------------------------------------------
# Table matching
# ---------------------------------------------------------------------------


def _exact_match(name: str, tables: List[DetectedTable]) -> Optional[str]:
    """Exact or substring match only. Returns table_id or None."""
    for t in tables:
        title = t.title or ""
        if name == title or name == t.table_id:
            return t.table_id
    for t in tables:
        title = t.title or ""
        if title and (name in title or title in name):
            return t.table_id
    return None


def _fuzzy_match(name: str, tables: List[DetectedTable]) -> Optional[str]:
    """Fuzzy match against a (pre-filtered) table list. Returns table_id or None.

    Threshold is intentionally high (0.92) to avoid false positives between
    near-homograph names that differ only in a series suffix (e.g. "X-3" vs
    "X合計" or "X-D" share a long common prefix but are genuinely different).
    """
    best_ratio = 0.0
    best_id: Optional[str] = None
    for t in tables:
        for candidate in filter(None, [t.title, t.table_id]):
            ratio = difflib.SequenceMatcher(None, name, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = t.table_id
    return best_id if best_ratio >= 0.92 else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_latent_tables(tables: List[DetectedTable]) -> List[LatentTableProposal]:
    """Scan all detected tables' trailing notes and titles for latent-table proposals.

    A proposal is generated when a note (or aggregation-style title) mentions N ≥ 2
    entity names AND:
      - at least 1 entity matches a detected table   (confirms the note is relevant)
      - at least 1 entity has NO detected-table match (the latent/missing table)

    Two scan passes:
      1. Trailing notes  (t.notes)  — primary path after excel_parser fix
      2. Table titles    (t.title)  — fallback for notes that were misclassified as
                                      titles in older project files; only fires when
                                      the title looks like an enumeration/aggregation

    Matching uses a two-pass strategy to avoid cross-contamination between
    similarly-named series entries (e.g. C-1 / C-2 / C-3):
      Pass 1 — exact / substring match for ALL entities, building a
                "reserved" set of already-claimed table_ids.
      Pass 2 — fuzzy match only for entities that had no exact match,
                searching only tables NOT claimed in pass 1.
    """
    proposals: List[LatentTableProposal] = []
    # Track (source_table_id, frozenset(entities)) already proposed to avoid duplicates
    _seen: set = set()

    # Aggregation/enumeration pattern — same regex used in excel_parser._AGG_ENUM_RE
    _title_note_re = re.compile(
        r"[・、,＋+].+[のをにおける]*(合計|内訳|合算|総計|小計|集計|含む|合わせた|除く|除外)"
    )

    def _scan_text(t: DetectedTable, text: str, is_title: bool = False) -> None:
        """Process a single note/title text against the full table list."""
        entities = _extract_entities(text)
        if len(entities) < 2:
            return

        key = (t.table_id, frozenset(entities))
        if key in _seen:
            return
        _seen.add(key)

        # ── Pass 1: exact / substring matches ──────────────────────────
        exact_map: dict = {}
        reserved_ids: set = {t.table_id}
        for entity in entities:
            mid = _exact_match(entity, tables)
            if mid and mid != t.table_id:
                exact_map[entity] = mid
                reserved_ids.add(mid)

        # ── Pass 2: fuzzy on un-reserved tables only ────────────────────
        available = [t2 for t2 in tables if t2.table_id not in reserved_ids]

        # Pre-compute name prefixes of exact-matched tables (first 10 chars).
        # When at least one entity has already been exact-matched, a fuzzy
        # candidate that shares the same long prefix is very likely a sibling
        # in the same series (e.g. "TypeC合計" shares prefix with "TypeC-1").
        # Accepting such a fuzzy match would incorrectly consume a missing
        # series member — so we reject it and let it fall through to missing.
        exact_title_prefixes: List[str] = []
        if exact_map:
            id_to_table = {tb.table_id: tb for tb in tables}
            for eid in exact_map.values():
                tb = id_to_table.get(eid)
                title = (tb.title or "") if tb else ""
                if len(title) >= 10:
                    exact_title_prefixes.append(title[:10])

        def _prefix_conflict(candidate_table: DetectedTable) -> bool:
            """True when the candidate shares a 10-char prefix with an exact-matched table."""
            if not exact_title_prefixes:
                return False
            c_title = candidate_table.title or ""
            if len(c_title) < 10:
                return False
            c_pre = c_title[:10]
            return c_pre in exact_title_prefixes

        fuzzy_map: dict = {}
        for entity in entities:
            if entity not in exact_map:
                mid = _fuzzy_match(entity, available)
                if mid:
                    # Reject matches where the candidate shares a long-prefix
                    # with already exact-matched tables (series-sibling guard).
                    cand = next((tb for tb in available if tb.table_id == mid), None)
                    if cand is not None and _prefix_conflict(cand):
                        mid = None
                if mid:
                    fuzzy_map[entity] = mid

        # ── Classify each entity ────────────────────────────────────────
        detected_ids: List[str] = []
        detected_names: List[str] = []
        missing_names: List[str] = []

        for entity in entities:
            mid = exact_map.get(entity) or fuzzy_map.get(entity)
            if mid:
                detected_ids.append(mid)
                detected_names.append(entity)
            else:
                missing_names.append(entity)

        if not detected_names or not missing_names:
            return

        source_title = t.title or t.table_id
        note_type = _detect_note_type(text)
        note_short = text[:100] + ("…" if len(text) > 100 else "")

        type_label = {
            "aggregation": "集計注記",
            "exclusion":   "除外注記",
            "reference":   "参照注記",
            "general":     "注記",
        }.get(note_type, "注記")

        origin = "テーブル表題" if is_title else type_label

        reasoning = (
            f"テーブル「{source_title}」の{origin}「{note_short}」に "
            f"{len(entities)} 件の名称が列挙されています。"
            f"うち {len(detected_names)} 件（{', '.join(detected_names)}）は検出済みですが、"
            f"{len(missing_names)} 件（{', '.join(missing_names)}）は未検出です。"
            f"これらのテーブルが実際に存在する可能性があります。"
        )

        proposals.append(
            LatentTableProposal(
                proposal_id=f"LP_{len(proposals) + 1}",
                source_table_id=t.table_id,
                source_title=source_title,
                note_text=text,
                note_type=note_type,
                all_referenced=entities,
                detected_table_ids=detected_ids,
                detected_names=detected_names,
                missing_names=missing_names,
                reasoning=reasoning,
            )
        )

    for t in tables:
        # Primary: trailing notes
        for note in getattr(t, "notes", None) or []:
            _scan_text(t, note, is_title=False)

        # Fallback: aggregation-style titles (catches notes misclassified by older parser)
        title = getattr(t, "title", None) or ""
        if title and _title_note_re.search(title):
            _scan_text(t, title, is_title=True)

    return proposals


# ---------------------------------------------------------------------------
# Derived latent table generation (numeric subtraction)
# ---------------------------------------------------------------------------


def derive_latent_tables(
    tables: List[DetectedTable],
) -> List[DerivedLatentTable]:
    """
    For each latent table proposal where at least ONE component is missing
    from a set of tables related by a note, attempt to derive the missing
    table's data by numeric subtraction:

        missing ≈ aggregate_table − sum(detected_components)

    The aggregate table (parent) is identified as the candidate with the
    largest absolute numeric total. When more than one component is missing,
    the computed residual represents their combined total.

    Shape flexibility: only the numeric columns common to ALL candidates are
    used, so tables that differ only in non-numeric or extra columns can still
    participate. Row counts must match.

    Returns a list of DerivedLatentTable instances (empty if none can be derived).
    """
    latent_proposals = find_latent_tables(tables)
    id_to_table = {t.table_id: t for t in tables}
    derived: List[DerivedLatentTable] = []

    for lp in latent_proposals:
        try:
            _try_derive_one(lp, id_to_table, derived)
        except Exception:
            pass  # Skip any proposal that raises an unexpected error

    return derived


def _try_derive_one(
    lp: "LatentTableProposal",
    id_to_table: dict,
    derived: List[DerivedLatentTable],
) -> None:
    """Attempt to derive one latent table from a LatentTableProposal.
    Appends to `derived` in-place; raises on unrecoverable errors (caller catches)."""

    # Need at least one missing component to derive
    if not lp.missing_names:
        return

    # Accept any note type — aggregation wording is not required for the math
    # to work, and some notes phrase the relationship differently ("C-3を含む").

    # Gather all tables in the relationship: the note source + every detected ref
    candidate_ids: List[str] = [lp.source_table_id] + list(lp.detected_table_ids)
    candidate_ids_unique = list(dict.fromkeys(candidate_ids))  # preserve order, dedup
    candidates = [id_to_table.get(cid) for cid in candidate_ids_unique]
    if any(c is None or c.df is None or c.df.empty for c in candidates):
        return

    # Find numeric columns common to ALL candidates
    common_num_cols: Optional[List] = None
    for c in candidates:
        nc = list(c.df.select_dtypes(include=[np.number]).columns)
        if not nc:
            return  # This candidate has no numeric data
        if common_num_cols is None:
            common_num_cols = nc
        else:
            # Keep only columns present in both, in the order of the first candidate
            common_num_cols = [col for col in common_num_cols if col in nc]

    if not common_num_cols:
        return  # No shared numeric columns

    # Row counts must match for element-wise subtraction
    row_counts = [len(c.df) for c in candidates]
    if len(set(row_counts)) != 1:
        return

    # Extract numeric arrays for common columns
    arrays: List[Tuple["DetectedTable", np.ndarray]] = []
    for c in candidates:
        arr = np.nan_to_num(
            c.df[common_num_cols].values.astype(float), nan=0.0
        )
        arrays.append((c, arr))

    # Identify the PARENT as the candidate with the largest absolute numeric total
    totals = [float(np.nansum(np.abs(arr))) for _, arr in arrays]
    if max(totals) < 1e-9:
        return  # All zeros — nothing to derive

    parent_idx = int(np.argmax(totals))
    parent_table, parent_arr = arrays[parent_idx]
    children: List[Tuple["DetectedTable", np.ndarray]] = [
        (c, arr) for i, (c, arr) in enumerate(arrays) if i != parent_idx
    ]
    if not children:
        return

    # Sanity check: parent total must be ≥ 80% of sum of child totals.
    child_sum_total = sum(float(np.nansum(np.abs(arr))) for _, arr in children)
    if child_sum_total > 0 and totals[parent_idx] < child_sum_total * 0.8:
        return  # Parent smaller than children — wrong table identified as parent

    # Compute: missing = parent − sum(detected_children)
    c_sum = np.zeros_like(parent_arr, dtype=float)
    for _, arr in children:
        c_sum += arr
    derived_arr = parent_arr - c_sum

    # Build the derived DataFrame: parent's full structure, numeric cols replaced
    derived_df = parent_table.df.copy()
    for ci, col in enumerate(common_num_cols):
        derived_df[col] = derived_arr[:, ci]

    # Human-readable names and formula
    parent_label = parent_table.title or parent_table.table_id
    child_labels = [c.title or c.table_id for c, _ in children]
    missing_name = (
        lp.missing_names[0]
        if len(lp.missing_names) == 1
        else "（" + " + ".join(lp.missing_names) + "）合算"
    )
    formula = (
        f"{missing_name}  ≈  {parent_label}"
        f"  −  ( {' + '.join(child_labels)} )"
    )

    derived.append(
        DerivedLatentTable(
            proposal_id=f"DLT_{len(derived) + 1}",
            derived_name=missing_name,
            df=derived_df,
            parent_table_id=parent_table.table_id,
            parent_title=parent_label,
            detected_child_ids=[c.table_id for c, _ in children],
            note_text=lp.note_text,
            derivation_formula=formula,
            source_display_order=[parent_table.table_id] + [c.table_id for c, _ in children],
            reasoning=(
                f"注記「{lp.note_text[:100]}{'…' if len(lp.note_text) > 100 else ''}」に"
                f"よれば「{missing_name}」は「{parent_label}」の構成要素として記載されているが"
                f"未検出。検出済み構成要素（{', '.join(child_labels)}）を集計テーブルから"
                f"差し引くことで推定データを算出した。"
            ),
        )
    )
