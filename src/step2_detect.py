"""
ステップ2 テーブル検出モジュール。

処理概要: 生グリッドまたは DataFrame を受け取り、行分類ステートマシンでテーブル領域を特定する。
          ファイル読み込み・整形・分析は行わない。
入力    : List[SheetGrid]（step1_upload が構築した生グリッド）/ pd.DataFrame（CSV の場合）、ファイル名
出力    : List[DetectedTable]（検出されたテーブルごとの位置・DataFrame・タイトル・注記を含む）
"""

import re
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .keywords import STAT_PLACEHOLDERS as _STAT_PLACEHOLDERS
from .models import DetectedTable, SheetGrid
from .step3_normalize import (
    detect_cross_table,
    detect_header_roles,
    fill_grouping_cols,
    merge_header_rows,
    remove_aggregates,
    stack_cross_table,
)


# ---------------------------------------------------------------------------
# セル値ヘルパー
# ---------------------------------------------------------------------------

def _is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


_NUM_STRIP = str.maketrans("", "", ",、△▲")


def _cell_is_numeric(v: Any) -> bool:
    """セルの値が数値として解釈できる場合に True を返す。
    統計秘匿マーカー（***、X 等）も数値扱いとする。"""
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return v == v
    s = str(v).strip()
    if s in _STAT_PLACEHOLDERS:
        return True
    try:
        float(s.translate(_NUM_STRIP))
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# 行分類
# ---------------------------------------------------------------------------

_RT_EMPTY   = "empty"
_RT_TITLE   = "title"
_RT_COL_HDR = "col_hdr"
_RT_DATA    = "data"
_RT_MIXED   = "mixed"
_RT_NOTE    = "note"

_NOTE_PREFIXES: Tuple[str, ...] = (
    "※", "＊", "*", "注）", "注)", "（注", "(注", "注意", "＜注", "<注",
    "note:", "NOTE:", "＊注", "*注",
)

_AGG_ENUM_RE = re.compile(
    r"[・、,＋+].+[のをにおける]*(合計|内訳|合算|総計|小計|集計|含む|合わせた|除く|除外)"
)


def _row_content_profile(grid: List[List[Any]], r: int, max_col: int) -> Dict[str, Any]:
    """grid の1行分のコンテンツプロファイル辞書を返す。"""
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
    """行のプロファイルを _RT_* 定数のいずれかに分類する。"""
    filled = p["filled"]
    if filled == 0:
        return _RT_EMPTY

    n_ratio = p["numeric"] / filled

    if n_ratio >= 0.55:
        return _RT_DATA

    if filled == 1 and p["text"] == 1:
        txt = p["texts"][0] if p["texts"] else ""
        if txt.startswith(_NOTE_PREFIXES) or len(txt) > 60:
            return _RT_NOTE
        if _AGG_ENUM_RE.search(txt):
            return _RT_NOTE
        return _RT_TITLE

    if p["numeric"] == 0 and p["text"] >= 2:
        unique = set(p["texts"])
        if len(unique) == 1:
            txt = next(iter(unique))
            if txt.startswith(_NOTE_PREFIXES) or len(txt) > 60:
                return _RT_NOTE
            if _AGG_ENUM_RE.search(txt):
                return _RT_NOTE
            if p["text"] >= 4:
                return _RT_COL_HDR
            return _RT_TITLE if len(txt) <= 40 else _RT_COL_HDR

    if p["texts"] and max(len(t) for t in p["texts"]) > 80 and n_ratio <= 0.30:
        return _RT_NOTE

    if p["text"] >= 2 and n_ratio <= 0.30:
        return _RT_COL_HDR

    return _RT_MIXED


# ---------------------------------------------------------------------------
# テーブル領域検出
# ---------------------------------------------------------------------------

def _find_column_groups(
    grid: List[List[Any]],
    start_row: int,
    end_row: int,
    max_col: int,
    gap_threshold: int = 2,
) -> List[Tuple[int, int]]:
    """行スパン内で列の空白を境に (start_col, end_col) のペアを返す。"""
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
    """構造的な行ごとの分析を使用してテーブル候補領域を検出する。"""
    if max_row == 0 or max_col == 0:
        return []

    profiles: Dict[int, Dict] = {}
    row_types: Dict[int, str] = {}
    for r in range(1, max_row + 1):
        p = _row_content_profile(grid, r, max_col)
        profiles[r] = p
        row_types[r] = _classify_row(p)

    def _mk() -> Dict:
        return {
            "band_start": None,
            "band_end": None,
            "col_start": max_col + 1,
            "col_end": 0,
            "title_rows": [],
            "header_rows": [],
            "data_rows": [],
            "trailing_notes": [],
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
    pending_titles: List[Tuple[int, str]] = []
    consec_empty = 0
    last_filled = 0

    for r in range(1, max_row + 1):
        rt = row_types[r]

        if rt == _RT_EMPTY:
            consec_empty += 1
            if cur["data_rows"] and consec_empty >= 2:
                _flush(cur, last_filled, regions)
                cur = _mk()
                pending_titles = []
            elif consec_empty > 4:
                pending_titles = []
            continue

        consec_empty = 0
        last_filled = r
        p = profiles[r]

        if rt == _RT_TITLE:
            if cur["data_rows"]:
                _flush(cur, r - 1, regions)
                cur = _mk()
                pending_titles = []
            elif len(cur["header_rows"]) >= 2:
                _flush(cur, r - 1, regions)
                cur = _mk()
                pending_titles = []
            text = " ".join(p["texts"])
            pending_titles.append((r, text))
            continue

        if rt == _RT_COL_HDR:
            if cur["data_rows"]:
                tbl_width = cur["col_end"] - cur["col_start"] + 1
                row_width = (
                    p["col_max"] - p["col_min"] + 1
                    if p["col_min"] is not None and p["col_max"] is not None
                    else 0
                )
                filled_in_span = (
                    sum(
                        1 for c in range(cur["col_start"], cur["col_end"] + 1)
                        if _is_filled(grid[r][c])
                    )
                    if tbl_width > 0
                    else 0
                )
                fill_density = filled_in_span / tbl_width if tbl_width > 0 else 0

                if tbl_width > 0 and (
                    row_width / tbl_width < 0.5 or fill_density < 0.30
                ):
                    cur["data_rows"].append(r)
                    _upd_cols(cur, r)
                    continue

                _flush(cur, r - 1, regions)
                cur = _mk()

            if cur["band_start"] is None:
                for tr, tt in pending_titles:
                    cur["title_rows"].append((tr, tt))
                    _upd_cols(cur, tr)
                pending_titles = []
                cur["band_start"] = cur["title_rows"][0][0] if cur["title_rows"] else r

            cur["header_rows"].append(r)
            _upd_cols(cur, r)
            continue

        if rt == _RT_NOTE:
            unique_note_texts = list(dict.fromkeys(profiles[r]["texts"]))
            note_text = unique_note_texts[0] if unique_note_texts else ""
            if note_text:
                if cur["data_rows"]:
                    _flush(cur, r - 1, regions)
                    if regions:
                        regions[-1]["trailing_notes"].append(note_text)
                    cur = _mk()
                    pending_titles = []
                elif cur["band_start"] is None and regions:
                    regions[-1]["trailing_notes"].append(note_text)
            continue

        if cur["band_start"] is None:
            for tr, tt in pending_titles:
                cur["title_rows"].append((tr, tt))
                _upd_cols(cur, tr)
            pending_titles = []
            cur["band_start"] = cur["title_rows"][0][0] if cur["title_rows"] else r

        cur["data_rows"].append(r)
        _upd_cols(cur, r)

    if cur["band_start"] is not None:
        _flush(cur, last_filled, regions)

    return regions


# ---------------------------------------------------------------------------
# テーブル品質フィルタ
# ---------------------------------------------------------------------------

def _classify_table_quality(
    df: pd.DataFrame,
    grid: List[List[Any]],
    band_start: int,
    band_end: int,
    col_start: int,
    col_end: int,
) -> str:
    """検出された矩形領域を "ok" / "metadata" / "discard" に分類する。"""
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

    if df.shape[0] <= 5 and df.shape[1] >= 5:
        num_ct = sum(
            int(pd.to_numeric(df[c], errors="coerce").notna().sum()) for c in df.columns
        )
        if num_ct == 0:
            return "discard"

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
        total_numeric = sum(
            int(pd.to_numeric(df[c], errors="coerce").notna().sum())
            for c in df.columns
        )
        numeric_density = total_numeric / raw_total if raw_total > 0 else 0
        if total_numeric < 10 and numeric_density < 0.15:
            return "discard"

    return "ok"


# ---------------------------------------------------------------------------
# DataFrame 抽出
# ---------------------------------------------------------------------------

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
) -> Tuple[pd.DataFrame, Optional[str], Optional[pd.DataFrame]]:
    """矩形領域を DataFrame として抽出する。

    Returns (df, title, raw_df)
    """
    if start_row > end_row or start_col > end_col:
        return pd.DataFrame(), None, None

    rows = [
        [grid[r][c] for c in range(start_col, end_col + 1)]
        for r in range(start_row, end_row + 1)
    ]

    if not rows:
        return pd.DataFrame(), None, None

    n_title, header_roles = detect_header_roles(rows)
    num_cols = end_col - start_col + 1

    title: Optional[str] = None
    if n_title > 0:
        title_parts = []
        for i in range(n_title):
            vals = [v for v in rows[i] if v is not None]
            if vals:
                title_parts.append(str(vals[0]).strip())
        title = " / ".join(title_parts) if title_parts else None

    remaining = rows[n_title:]
    n_header = len(header_roles)

    if n_header == 0:
        columns = [f"列{i + 1}" for i in range(num_cols)]
        df = pd.DataFrame(remaining, columns=columns)
        return df.dropna(how="all").reset_index(drop=True), title, None

    if n_header == 1:
        header = _make_unique_columns(
            [str(v) if v is not None else "" for v in remaining[0]]
        )
        df = pd.DataFrame(remaining[1:], columns=header)
        return df.dropna(how="all").reset_index(drop=True), title, None

    raw_header = _make_unique_columns(
        [str(v) if v is not None else "" for v in remaining[0]]
    )
    raw_df = pd.DataFrame(remaining[1:], columns=raw_header).dropna(how="all").reset_index(drop=True)

    header_data = [remaining[i] for i in range(n_header)]
    merged = merge_header_rows(header_data, header_roles, num_cols)
    header = _make_unique_columns(merged)
    df = pd.DataFrame(remaining[n_header:], columns=header)

    return df.dropna(how="all").reset_index(drop=True), title, raw_df


# ---------------------------------------------------------------------------
# シート内テーブル検出
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\d{4}")
_SECTION_ONLY_RE = re.compile(r"^\s*\d{4}[年度]?\s*$")


def _propagate_sheet_title(detected: List[DetectedTable]) -> None:
    """同一シート内で複数テーブルを検出した際、先頭テーブルタイトルの
    シート共通部分を後続テーブルに引き継ぐ。"""
    if not detected:
        return

    for _sheet, group_iter in groupby(detected, key=lambda t: t.sheet_name):
        tables = list(group_iter)
        if len(tables) < 2:
            continue

        first_title = tables[0].title or ""
        if " / " not in first_title:
            continue

        parts = [p.strip() for p in first_title.split("/")]
        sheet_parts = [p for p in parts if not _YEAR_RE.search(p) and p]
        section_parts = [p for p in parts if _YEAR_RE.search(p) and p]

        if not sheet_parts:
            continue

        sheet_title = " ".join(sheet_parts)

        propagated = False
        for t in tables[1:]:
            raw = t.title or ""
            if not _SECTION_ONLY_RE.match(raw):
                continue
            t.title = f"{sheet_title}_{raw.strip()}"
            propagated = True

        if propagated and section_parts:
            tables[0].title = f"{sheet_title}_{section_parts[0]}"


def _apply_cross_table_detection(tables: List[DetectedTable], filename: str) -> None:
    """テーブルリスト全体にクロス集計検出と縦持ち変換を適用する。"""
    from .step3_normalize import _is_agg_label

    for t in tables:
        if t.df is None or t.df.empty:
            continue
        info = detect_cross_table(t.df, title=t.title, filename=filename)
        if info:
            t.stack_info = info
            stacked = stack_cross_table(t.df, info)
            var_name = info.get("var_name", "")
            if var_name and var_name in stacked.columns:
                agg_mask = stacked[var_name].astype(str).apply(_is_agg_label)
                if agg_mask.any():
                    stacked = stacked[~agg_mask].reset_index(drop=True)
            t.stacked_df = stacked


def _detect_tables_in_grid(
    grid: List[List[Any]],
    max_row: int,
    max_col: int,
    sheet_name: str,
    table_counter: Dict[str, int],
) -> List[DetectedTable]:
    """グリッドからテーブルを検出して DetectedTable のリストを返す。"""
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

        first_header_row = min(reg["header_rows"]) if reg["header_rows"] else band_start

        for col_start, col_end in col_groups:
            if (band_end - first_header_row + 1) < 2:
                continue

            df, inner_title, raw_df = _extract_dataframe(
                grid, first_header_row, band_end, col_start, col_end
            )
            if df.empty or len(df) == 0:
                continue

            quality = _classify_table_quality(
                df, grid, first_header_row, band_end, col_start, col_end
            )
            if quality != "ok":
                continue

            effective_title = inner_title
            if effective_title is None and reg["title_rows"]:
                effective_title = " / ".join(t for _, t in reg["title_rows"])

            safe = sheet_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            table_counter[safe] = table_counter.get(safe, 0) + 1
            table_id = f"{safe}_T{table_counter[safe]}"

            pre_fill_df_candidate = df
            df, filled_cols = fill_grouping_cols(df)
            pre_fill_df = pre_fill_df_candidate if filled_cols else None

            (
                cleaned_df,
                agg_rows,
                agg_cols,
                agg_row_positions,
                agg_row_meta,
                agg_col_meta,
            ) = remove_aggregates(df)
            pre_agg_df = df if (agg_rows or agg_cols) else None

            detected.append(
                DetectedTable(
                    table_id=table_id,
                    sheet_name=sheet_name,
                    start_row=first_header_row,
                    end_row=band_end,
                    start_col=col_start,
                    end_col=col_end,
                    df=cleaned_df,
                    title=effective_title,
                    notes=reg.get("trailing_notes", []),
                    raw_df=raw_df,
                    pre_agg_df=pre_agg_df,
                    agg_rows_removed=agg_rows,
                    agg_cols_removed=agg_cols,
                    agg_rows_removed_positions=agg_row_positions,
                    agg_removed_row_metadata=agg_row_meta,
                    agg_removed_col_metadata=agg_col_meta,
                    filled_cols=filled_cols,
                    pre_fill_df=pre_fill_df,
                )
            )

    _propagate_sheet_title(detected)
    return detected


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def detect_tables(
    sheets: List[SheetGrid], filename: str
) -> Tuple[List[DetectedTable], List[str]]:
    """SheetGrid のリストからテーブルを検出する。"""
    all_tables: List[DetectedTable] = []
    table_counter: Dict[str, int] = {}

    for sheet in sheets:
        tables = _detect_tables_in_grid(
            sheet.grid, sheet.max_row, sheet.max_col,
            sheet.sheet_name, table_counter,
        )
        all_tables.extend(tables)

    _apply_cross_table_detection(all_tables, filename)
    sheet_names = [s.sheet_name for s in sheets]
    return all_tables, sheet_names


def detect_from_csv(df: pd.DataFrame, filename: str) -> Tuple[List[DetectedTable], List[str]]:
    """DataFrame（CSV 読み込み済み）を単一テーブルとして検出する。"""
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
    _apply_cross_table_detection([table], filename)
    return [table], ["CSV"]


# ---------------------------------------------------------------------------
# UI 表示補助（steps/step2_detect.py から利用）
# ---------------------------------------------------------------------------

def get_original_df(t: DetectedTable):
    """整形処理適用前の生 DataFrame を返す。"""
    for candidate in [
        t.raw_df,
        getattr(t, "pre_fill_df", None),
        t.pre_agg_df,
        t.df,
    ]:
        if candidate is not None and not candidate.empty:
            return candidate
    return None


def build_tree_text(
    filename: str,
    sheets: List[str],
    tables_by_sheet: dict,
) -> str:
    """ファイル・シート・テーブルのツリー表示テキストを生成する。"""
    lines = [f"📁 {filename}"]
    for i, sheet in enumerate(sheets):
        sh_tables = tables_by_sheet.get(sheet, [])
        cnt_str = f"{len(sh_tables)} テーブル" if sh_tables else "テーブルなし"
        is_last_sh = i == len(sheets) - 1
        sh_pfx = "└── " if is_last_sh else "├── "
        lines.append(f"{sh_pfx}📋 {sheet}  ({cnt_str})")
        ch_pfx = "    " if is_last_sh else "│   "
        for j, t in enumerate(sh_tables):
            is_last_t = j == len(sh_tables) - 1
            t_pfx = ch_pfx + ("└── " if is_last_t else "├── ")
            dims = f"{t.row_count}行×{t.col_count}列"
            title_part = f"  [{t.title}]" if t.title else ""
            lines.append(f"{t_pfx}📊 {t.table_id}  {dims}{title_part}")
    return "\n".join(lines)


def group_tables_by_sheet(tables: List[DetectedTable]) -> dict:
    """テーブルリストをシート名でグループ化して返す。"""
    by_sheet: dict = {}
    for t in tables:
        by_sheet.setdefault(t.sheet_name, []).append(t)
    return by_sheet
