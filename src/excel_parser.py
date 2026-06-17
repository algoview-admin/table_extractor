import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import openpyxl
import pandas as pd

from .models import DetectedTable


# ---------------------------------------------------------------------------
# Grid building
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Formula evaluation helpers
# ---------------------------------------------------------------------------


def _col_to_num(col: str) -> int:
    """'A'→1, 'Z'→26, 'AA'→27 (case-insensitive)."""
    n = 0
    for ch in col.upper():
        n = n * 26 + (ord(ch) - 64)
    return n


def _parse_cell_ref(ref: str) -> Tuple[Optional[int], Optional[int]]:
    """'D5', '$D$5' → (row=5, col=4). Returns (None, None) on failure."""
    m = re.match(r"\$?([A-Z]+)\$?(\d+)$", ref.strip().upper())
    if m:
        return int(m.group(2)), _col_to_num(m.group(1))
    return None, None


def _eval_formula(formula: str, grid: List[List[Any]]) -> Optional[float]:
    """
    Evaluate simple Excel formulas against an already-built value grid.

    Supported forms:
      =SUM(A1:B3)              range sum
      =SUM(A1:B3, C5:D7)      multi-range sum
      =A1                      single cell reference
      =A1+B1  / =A1-B1  / =A1*B1   two-cell arithmetic
    Returns None when the formula is unsupported or required cells are still None.
    """
    if not formula or not isinstance(formula, str):
        return None
    expr = formula.lstrip("=").strip()

    def _get(r: int, c: int) -> Optional[float]:
        if 1 <= r < len(grid) and 1 <= c < len(grid[r]):
            v = grid[r][c]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return None

    def _range_sum(ref1: str, ref2: str) -> Optional[float]:
        r1, c1 = _parse_cell_ref(ref1)
        r2, c2 = _parse_cell_ref(ref2)
        if None in (r1, c1, r2, c2):
            return None
        total = 0.0
        found = False
        for r in range(min(r1, r2), max(r1, r2) + 1):
            for c in range(min(c1, c2), max(c1, c2) + 1):
                v = _get(r, c)
                if v is not None:
                    total += v
                    found = True
        return total if found else None

    # SUM(arg, arg, ...) where each arg is a range ref1:ref2 or a single cell ref
    m = re.fullmatch(r"SUM\((.+)\)", expr, re.IGNORECASE)
    if m:
        args = [a.strip() for a in m.group(1).split(",")]
        grand = 0.0
        found = False
        for arg in args:
            parts = arg.split(":")
            v = _range_sum(parts[0], parts[1]) if len(parts) == 2 else None
            if v is None and len(parts) == 1:
                r, c = _parse_cell_ref(parts[0])
                v = _get(r, c) if r else None
            if v is not None:
                grand += v
                found = True
        return grand if found else None

    # Single cell reference
    if re.fullmatch(r"\$?[A-Z]+\$?\d+", expr, re.IGNORECASE):
        r, c = _parse_cell_ref(expr)
        return _get(r, c) if r else None

    # Two-cell arithmetic
    m = re.fullmatch(
        r"(\$?[A-Z]+\$?\d+)\s*([+\-\*])\s*(\$?[A-Z]+\$?\d+)", expr, re.IGNORECASE
    )
    if m:
        r1, c1 = _parse_cell_ref(m.group(1))
        op = m.group(2)
        r2, c2 = _parse_cell_ref(m.group(3))
        v1 = _get(r1, c1) if r1 else None
        v2 = _get(r2, c2) if r2 else None
        if v1 is not None and v2 is not None:
            return v1 + v2 if op == "+" else (v1 - v2 if op == "-" else v1 * v2)

    return None


def _fill_formula_cells(
    grid: List[List[Any]], max_row: int, max_col: int, ws_formulas
) -> None:
    """
    Fill None cells in *grid* by evaluating Excel formulas from *ws_formulas*
    (a worksheet loaded with data_only=False).

    Runs up to 5 passes so that sub-totals that sum other sub-totals are
    resolved correctly (each pass makes previously-None cells available to
    formulas in the next pass).
    """
    formula_map: Dict[Tuple[int, int], str] = {}
    for row in ws_formulas.iter_rows(
        min_row=1, max_row=max_row, min_col=1, max_col=max_col
    ):
        for cell in row:
            v = cell.value
            if isinstance(v, str) and v.strip().startswith("="):
                formula_map[(cell.row, cell.column)] = v.strip()

    for _ in range(5):
        progress = False
        for (r, c), formula in formula_map.items():
            if grid[r][c] is None:
                result = _eval_formula(formula, grid)
                if result is not None:
                    grid[r][c] = result
                    progress = True
        if not progress:
            break  # nothing new resolved — stop early


# ---------------------------------------------------------------------------
# Grid building
# ---------------------------------------------------------------------------


def _build_value_grid(ws, ws_formulas=None) -> Tuple[List[List[Any]], int, int]:
    """
    Build a 1-indexed 2D grid from an openpyxl worksheet.

    - Merged cell values are propagated across the entire merged range.
    - When *ws_formulas* is provided (the same sheet loaded with
      data_only=False), cells whose cached value is None are re-evaluated
      from their formula string.  This handles files saved without formula
      recalculation (e.g. exported from LibreOffice / Google Sheets).
    """
    max_row = ws.max_row or 0
    max_col = ws.max_column or 0

    if max_row == 0 or max_col == 0:
        return [[]], 0, 0

    grid: List[List[Any]] = [[None] * (max_col + 1) for _ in range(max_row + 1)]

    for row in ws.iter_rows(min_row=1, max_row=max_row, min_col=1, max_col=max_col):
        for cell in row:
            if cell.value is not None:
                grid[cell.row][cell.column] = cell.value

    # Propagate merged cell top-left value across the merged range
    for merge_range in ws.merged_cells.ranges:
        top_val = grid[merge_range.min_row][merge_range.min_col]
        for r in range(merge_range.min_row, merge_range.max_row + 1):
            for c in range(merge_range.min_col, merge_range.max_col + 1):
                grid[r][c] = top_val

    # Evaluate uncached formula cells (e.g. SUM rows where cache is missing)
    if ws_formulas is not None:
        _fill_formula_cells(grid, max_row, max_col, ws_formulas)

    return grid, max_row, max_col


def _build_grid_from_xlrd(sheet) -> Tuple[List[List[Any]], int, int]:
    """Build a 1-indexed 2D grid from an xlrd sheet."""
    import xlrd

    nrows, ncols = sheet.nrows, sheet.ncols
    grid: List[List[Any]] = [[None] * (ncols + 1) for _ in range(nrows + 1)]

    for r in range(nrows):
        for c in range(ncols):
            cell = sheet.cell(r, c)
            if cell.ctype not in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
                grid[r + 1][c + 1] = cell.value

    return grid, nrows, ncols


# ---------------------------------------------------------------------------
# Table boundary detection
# ---------------------------------------------------------------------------


def _is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


# Characters removed before numeric parsing (Japanese minus signs, thousand separators)
_NUM_STRIP = str.maketrans("", "", ",、△▲")


def _cell_is_numeric(v: Any) -> bool:
    """Return True when the cell value can be interpreted as a number."""
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return v == v  # exclude float NaN
    try:
        float(str(v).strip().translate(_NUM_STRIP))
        return True
    except (ValueError, TypeError):
        return False


# ── Row-type constants ────────────────────────────────────────────────────────
_RT_EMPTY   = "empty"    # no filled cells
_RT_TITLE   = "title"    # 1 short text cell — table/section name candidate
_RT_COL_HDR = "col_hdr"  # ≥2 text cells, few numbers — column-axis labels
_RT_DATA    = "data"     # ≥55 % numeric — measurement rows
_RT_MIXED   = "mixed"    # doesn't fit cleanly (e.g. row-header + numbers)


def _row_content_profile(grid: List[List[Any]], r: int, max_col: int) -> Dict[str, Any]:
    """Return a content profile dict for a single grid row."""
    numeric = text = 0
    col_min: Optional[int] = None
    col_max: Optional[int] = None
    texts: List[str] = []

    for c in range(1, max_col + 1):
        v = grid[r][c]
        if not _is_filled(v):
            continue
        col_min = c if col_min is None else min(col_min, c)
        col_max = c
        if _cell_is_numeric(v):
            numeric += 1
        else:
            text += 1
            texts.append(str(v).strip())

    return {
        "numeric": numeric,
        "text": text,
        "filled": numeric + text,
        "col_min": col_min,
        "col_max": col_max,
        "texts": texts,
    }


def _classify_row(p: Dict[str, Any]) -> str:
    """Classify a row profile into one of the _RT_* constants."""
    filled = p["filled"]
    if filled == 0:
        return _RT_EMPTY

    n_ratio = p["numeric"] / filled

    # Mostly numeric → data row
    if n_ratio >= 0.55:
        return _RT_DATA

    # Single short text cell → title candidate
    if filled == 1 and p["text"] == 1:
        return _RT_TITLE if (p["texts"] and len(p["texts"][0]) <= 40) else _RT_COL_HDR

    # Multiple text cells, very few numbers → column-header row
    if p["text"] >= 2 and n_ratio <= 0.30:
        return _RT_COL_HDR

    return _RT_MIXED


def _find_column_groups(
    grid: List[List[Any]],
    start_row: int,
    end_row: int,
    max_col: int,
    gap_threshold: int = 2,
) -> List[Tuple[int, int]]:
    """Return (start_col, end_col) pairs within a row span, splitting on column gaps.

    Used to separate side-by-side tables that share the same row range.
    """
    col_has_content = [False] * (max_col + 1)
    for r in range(start_row, end_row + 1):
        for c in range(1, max_col + 1):
            if _is_filled(grid[r][c]):
                col_has_content[c] = True

    groups: List[Tuple[int, int]] = []
    group_start: Optional[int] = None
    empty_streak = 0

    for c in range(1, max_col + 1):
        if col_has_content[c]:
            if group_start is None:
                group_start = c
            empty_streak = 0
        else:
            if group_start is not None:
                empty_streak += 1
                if empty_streak >= gap_threshold:
                    groups.append((group_start, c - empty_streak))
                    group_start = None
                    empty_streak = 0

    if group_start is not None:
        last_filled = max(
            (c for c in range(group_start, max_col + 1) if col_has_content[c]),
            default=group_start,
        )
        groups.append((group_start, last_filled))

    return groups


def _detect_table_regions(
    grid: List[List[Any]], max_row: int, max_col: int
) -> List[Dict[str, Any]]:
    """Detect candidate table regions using structural row-by-row analysis.

    Algorithm
    ---------
    1. Classify every row as empty / title / col_hdr / data / mixed.
    2. Run a state machine that accumulates rows into regions.  Key transition
       rules that go beyond simple "empty row = table boundary":

       • A col_hdr row appearing *after data rows* without an empty separator
         signals a structural discontinuity (new table header) → flush the
         current region and start a new one.

       • A title row appearing after data rows also signals a new table.

       • Up to 4 consecutive empty rows are tolerated between a title and the
         header/data of its table.  More than 4 clears the pending title.

    3. Within each region the actual column span is tracked from every
       contributing row so that row-header columns to the left of the value
       area are always included.

    Returns a list of region dicts, each with:
        band_start  – first row index of the region
        band_end    – last row index
        col_start   – leftmost non-empty column across all region rows
        col_end     – rightmost non-empty column
        title_rows  – [(row_idx, text)] for title rows in/above the region
        header_rows – [row_idx] for col_hdr rows
        data_rows   – [row_idx] for data/mixed rows
    """
    if max_row == 0 or max_col == 0:
        return []

    # ── Phase 1: classify all rows ──
    profiles: Dict[int, Dict] = {}
    row_types: Dict[int, str] = {}
    for r in range(1, max_row + 1):
        p = _row_content_profile(grid, r, max_col)
        profiles[r] = p
        row_types[r] = _classify_row(p)

    # ── Phase 2: state machine ──
    def _mk() -> Dict:
        return {
            "band_start": None,
            "band_end": None,
            "col_start": max_col + 1,
            "col_end": 0,
            "title_rows": [],   # [(row_idx, text)]
            "header_rows": [],  # [row_idx]
            "data_rows": [],    # [row_idx]
        }

    def _upd_cols(reg: Dict, r: int) -> None:
        p = profiles[r]
        if p["col_min"] is not None:
            reg["col_start"] = min(reg["col_start"], p["col_min"])
        if p["col_max"] is not None:
            reg["col_end"] = max(reg["col_end"], p["col_max"])

    def _flush(reg: Dict, last_row: int, regions: List) -> None:
        if not (reg["data_rows"] or len(reg["header_rows"]) >= 2):
            return
        if reg["band_start"] is None or reg["col_end"] < reg["col_start"]:
            return
        reg["band_end"] = last_row
        regions.append(reg)

    regions: List[Dict] = []
    cur = _mk()
    pending_titles: List[Tuple[int, str]] = []  # title rows not yet attached
    consec_empty = 0
    last_filled = 0

    for r in range(1, max_row + 1):
        rt = row_types[r]

        # ── empty row ──────────────────────────────────────────────────────
        if rt == _RT_EMPTY:
            consec_empty += 1
            if cur["data_rows"] and consec_empty >= 1:
                # Gap after data rows → close this table
                _flush(cur, last_filled, regions)
                cur = _mk()
                pending_titles = []
            elif consec_empty > 4:
                # Long blank stretch → abandon pending titles
                pending_titles = []
            continue

        consec_empty = 0
        last_filled = r
        p = profiles[r]

        # ── title row ──────────────────────────────────────────────────────
        if rt == _RT_TITLE:
            if cur["data_rows"]:
                # Title appearing after data rows → end current table
                _flush(cur, r - 1, regions)
                cur = _mk()
                pending_titles = []
            text = " ".join(p["texts"])
            pending_titles.append((r, text))
            continue

        # ── column-header row ──────────────────────────────────────────────
        if rt == _RT_COL_HDR:
            if cur["data_rows"]:
                # Header appearing after data without a blank row
                # → structural discontinuity: close current, start fresh
                _flush(cur, r - 1, regions)
                cur = _mk()
                # pending_titles may belong to this new table — keep them

            if cur["band_start"] is None:
                # Attach pending titles and open the region
                for tr, tt in pending_titles:
                    cur["title_rows"].append((tr, tt))
                    _upd_cols(cur, tr)
                pending_titles = []
                cur["band_start"] = cur["title_rows"][0][0] if cur["title_rows"] else r

            cur["header_rows"].append(r)
            _upd_cols(cur, r)
            continue

        # ── data / mixed row ───────────────────────────────────────────────
        if cur["band_start"] is None:
            for tr, tt in pending_titles:
                cur["title_rows"].append((tr, tt))
                _upd_cols(cur, tr)
            pending_titles = []
            cur["band_start"] = cur["title_rows"][0][0] if cur["title_rows"] else r

        cur["data_rows"].append(r)
        _upd_cols(cur, r)

    # Final flush
    if cur["band_start"] is not None:
        _flush(cur, last_filled, regions)

    return regions


def _classify_table_quality(
    df: "pd.DataFrame",
    grid: List[List[Any]],
    band_start: int,
    band_end: int,
    col_start: int,
    col_end: int,
) -> str:
    """Classify a detected rectangular region into one of three quality tiers.

    Returns
    -------
    "ok"       — suitable for analysis
    "metadata" — structured but not analytical (menus, index/TOC tables)
    "discard"  — not a table (free-text paragraphs, notes, empty fragments)

    Signal mapping
    --------------
    1. Fill rate < 20 %                       → discard
    2. Fewer than 2 dense data rows           → discard
    3. Wide + short + all-text (≤5 rows, ≥5 cols, 0 numeric) → discard
    4. Long text (col header or cell > 80 ch, OR > 25 % of text cells > 40 ch)
       • + zero numeric cells                 → discard (free-text block)
       • + some numeric cells                 → discard (index/TOC table)
    """
    # Signal 1: raw fill rate
    raw_total = (band_end - band_start + 1) * (col_end - col_start + 1)
    if raw_total == 0:
        return "discard"
    raw_filled = sum(
        1
        for r in range(band_start, band_end + 1)
        for c in range(col_start, col_end + 1)
        if _is_filled(grid[r][c])
    )
    if raw_filled / raw_total < 0.20:
        return "discard"

    # Signal 2: dense rows (≥25 % of columns filled)
    if df.empty:
        return "discard"
    n_cols = max(len(df.columns), 1)
    dense_rows = sum(
        1
        for _, row in df.iterrows()
        if sum(1 for v in row if v is not None and str(v).strip()) / n_cols >= 0.25
    )
    if dense_rows < 2:
        return "discard"

    # Signal 3: wide + short + all-text → selection menus / form grids
    if df.shape[0] <= 5 and df.shape[1] >= 5:
        num_ct = sum(
            int(pd.to_numeric(df[c], errors="coerce").notna().sum()) for c in df.columns
        )
        if num_ct == 0:
            return "discard"

    # Signal 4: long-text content (check column headers AND cell values)
    txt_ct = lng_ct = 0
    max_len = 0

    def _tally(raw: str) -> None:
        nonlocal txt_ct, lng_ct, max_len
        s = raw.strip()
        if not s or s.lower() == "nan":
            return
        try:
            float(s.translate(_NUM_STRIP))
            return
        except (ValueError, TypeError):
            pass
        txt_ct += 1
        n = len(s)
        if n > max_len:
            max_len = n
        if n > 40:
            lng_ct += 1

    for col_name in df.columns:
        _tally(str(col_name))
    for _, row in df.iterrows():
        for v in row:
            if not pd.isna(v):
                _tally(str(v))

    if max_len > 80 or (txt_ct >= 3 and lng_ct / txt_ct > 0.25):
        return "discard"

    return "ok"


# ---------------------------------------------------------------------------
# DataFrame extraction from grid region
# ---------------------------------------------------------------------------


def _detect_header_rows(rows: List[List[Any]]) -> Tuple[int, int]:
    """
    Detect leading title rows and column header rows.

    Returns (n_title, n_header):
      n_title  — rows at the top that are section titles (single-cell string,
                 followed by a wider row). These are skipped before building the df.
      n_header — rows that form the column header (1 or 2 levels).
    """
    if not rows:
        return 0, 0

    def _nn(row: List[Any]) -> List[Any]:
        return [v for v in row if v is not None]

    def _str_count(vals: List[Any]) -> int:
        return sum(1 for v in vals if isinstance(v, str))

    def _num_count(vals: List[Any]) -> int:
        return sum(
            1 for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)
        )

    # --- Step 1: skip leading title rows (and blank spacer rows between them) ---
    # A title row has exactly 1 non-empty cell (a string).
    # All-empty rows that appear before the first wide row are treated as
    # spacers within the title block and are also skipped, so that a column
    # header row separated from its title by one or more blank rows is still
    # recognised correctly (e.g. title in row 2, blank in row 3, headers in row 4).
    n_title = 0
    while n_title < len(rows) - 1:
        nn_curr = _nn(rows[n_title])

        # All-empty row — skip as a spacer if a wider row still exists below
        if not nn_curr:
            has_wider_below = any(
                len(_nn(rows[j])) >= 2 for j in range(n_title + 1, len(rows))
            )
            if has_wider_below:
                n_title += 1
                continue
            else:
                break

        if not (len(nn_curr) == 1 and isinstance(nn_curr[0], str)):
            break
        # Accept this row as a title only if a wider row exists further down
        has_wider_below = any(
            len(_nn(rows[j])) >= 2 for j in range(n_title + 1, len(rows))
        )
        if has_wider_below:
            n_title += 1
        else:
            break

    remaining = rows[n_title:]
    if not remaining:
        return n_title, 0

    # --- Step 2: detect 1- or 2-level column headers in remaining rows ---
    nn_first = _nn(remaining[0])
    if not nn_first:
        return n_title, 0

    # First row is a header if ≥50 % of non-empty cells are strings
    if _str_count(nn_first) / len(nn_first) < 0.5:
        return n_title, 0

    # Second row is ALSO a header only when ALL of these hold:
    #   1. No numeric values in the second row
    #   2. High string ratio in the second row
    #   3. The first row contains DUPLICATE values — the hallmark of merged-cell
    #      spanning headers (e.g. "東京支社|東京支社|大阪支社|大阪支社").
    #      If all first-row values are unique, row 1 is almost certainly a
    #      normal single-level header and row 2 is data (even when all-string).
    if len(remaining) > 1:
        nn_second = _nn(remaining[1])
        first_strs = [str(v) for v in nn_first]
        first_has_duplicates = len(first_strs) != len(set(first_strs))
        if (
            nn_second
            and _num_count(nn_second) == 0
            and _str_count(nn_second) / len(nn_second) >= 0.8
            and first_has_duplicates
        ):
            return n_title, 2

    return n_title, 1


def _make_unique_columns(columns: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    result: List[str] = []
    for col in columns:
        col = col if col else "列"
        if col in seen:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            result.append(col)
    return result


def _extract_dataframe(
    grid: List[List[Any]],
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Extract a rectangular region as a DataFrame.

    Returns (df, title) where title is the section heading text detected
    immediately above the table data.
    """
    if start_row > end_row or start_col > end_col:
        return pd.DataFrame(), None

    rows = [
        [grid[r][c] for c in range(start_col, end_col + 1)]
        for r in range(start_row, end_row + 1)
    ]

    if not rows:
        return pd.DataFrame(), None

    n_title, n_header = _detect_header_rows(rows)
    num_cols = end_col - start_col + 1

    # Collect title text from skipped title rows
    title: Optional[str] = None
    if n_title > 0:
        title_parts = []
        for i in range(n_title):
            vals = [v for v in rows[i] if v is not None]
            if vals:
                title_parts.append(str(vals[0]).strip())
        title = " / ".join(title_parts) if title_parts else None

    # Rows available for header + data
    remaining = rows[n_title:]

    if n_header == 0:
        columns = [f"列{i + 1}" for i in range(num_cols)]
        df = pd.DataFrame(remaining, columns=columns)
    elif n_header == 1:
        header = _make_unique_columns(
            [str(v) if v is not None else "" for v in remaining[0]]
        )
        df = pd.DataFrame(remaining[1:], columns=header)
    else:
        combined = []
        for c in range(num_cols):
            parts = [
                str(remaining[h][c]).strip()
                for h in range(n_header)
                if remaining[h][c] is not None and str(remaining[h][c]).strip()
            ]
            combined.append("_".join(parts) if parts else f"列{c + 1}")
        header = _make_unique_columns(combined)
        df = pd.DataFrame(remaining[n_header:], columns=header)

    return df.dropna(how="all").reset_index(drop=True), title


# ---------------------------------------------------------------------------
# Per-worksheet detection
# ---------------------------------------------------------------------------


def _detect_tables_in_grid(
    grid: List[List[Any]],
    max_row: int,
    max_col: int,
    sheet_name: str,
    table_counter: Dict[str, int],
) -> List[DetectedTable]:
    """Detect tables using structural row classification and state machine."""
    if max_row == 0 or max_col == 0:
        return []

    regions = _detect_table_regions(grid, max_row, max_col)
    detected: List[DetectedTable] = []

    for reg in regions:
        band_start = reg["band_start"]
        band_end = reg["band_end"]

        col_groups = _find_column_groups(
            grid, band_start, band_end, max_col, gap_threshold=2
        )

        for col_start, col_end in col_groups:
            if (band_end - band_start + 1) < 2:
                continue

            df, inner_title = _extract_dataframe(
                grid, band_start, band_end, col_start, col_end
            )
            if df.empty or len(df) == 0:
                continue

            quality = _classify_table_quality(
                df, grid, band_start, band_end, col_start, col_end
            )
            if quality != "ok":
                continue

            effective_title = inner_title
            if effective_title is None and reg["title_rows"]:
                effective_title = " / ".join(t for _, t in reg["title_rows"])

            safe = sheet_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            table_counter[safe] = table_counter.get(safe, 0) + 1
            table_id = f"{safe}_T{table_counter[safe]}"

            detected.append(
                DetectedTable(
                    table_id=table_id,
                    sheet_name=sheet_name,
                    start_row=band_start,
                    end_row=band_end,
                    start_col=col_start,
                    end_col=col_end,
                    df=df,
                    title=effective_title,
                )
            )

    return detected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_excel(
    file_content: bytes, filename: str
) -> Tuple[List[DetectedTable], List[str]]:
    """Parse .xlsx/.xlsm file and detect all table regions."""
    ext = Path(filename).suffix.lower()
    all_tables: List[DetectedTable] = []
    table_counter: Dict[str, int] = {}

    if ext in (".xlsx", ".xlsm", ".xlsb"):
        raw = io.BytesIO(file_content)
        wb_data = openpyxl.load_workbook(raw, data_only=True, read_only=False)
        raw.seek(0)
        wb_form = openpyxl.load_workbook(raw, data_only=False, read_only=False)
        sheet_names = wb_data.sheetnames

        for sheet_name in sheet_names:
            ws_data = wb_data[sheet_name]
            ws_form = wb_form[sheet_name] if sheet_name in wb_form.sheetnames else None
            grid, max_row, max_col = _build_value_grid(ws_data, ws_formulas=ws_form)
            tables = _detect_tables_in_grid(
                grid, max_row, max_col, sheet_name, table_counter
            )
            all_tables.extend(tables)

    elif ext == ".xls":
        import xlrd

        wb = xlrd.open_workbook(file_contents=file_content)
        sheet_names = wb.sheet_names()

        for sheet_name in sheet_names:
            ws = wb.sheet_by_name(sheet_name)
            grid, max_row, max_col = _build_grid_from_xlrd(ws)
            tables = _detect_tables_in_grid(
                grid, max_row, max_col, sheet_name, table_counter
            )
            all_tables.extend(tables)
    else:
        raise ValueError(f"Unsupported Excel format: {ext}")

    return all_tables, sheet_names


def parse_csv(
    file_content: bytes, filename: str
) -> Tuple[List[DetectedTable], List[str]]:
    """Parse a CSV file as a single table."""
    for enc in ("utf-8-sig", "cp932", "shift_jis", "utf-8"):
        try:
            df = pd.read_csv(io.BytesIO(file_content), encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        df = pd.read_csv(io.BytesIO(file_content), encoding="latin-1")

    safe = Path(filename).stem.replace(" ", "_")
    table = DetectedTable(
        table_id=f"{safe}_T1",
        sheet_name="CSV",
        start_row=1,
        end_row=len(df) + 1,
        start_col=1,
        end_col=len(df.columns),
        df=df,
    )
    return [table], ["CSV"]
