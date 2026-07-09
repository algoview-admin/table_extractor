"""
ステップ1 ファイル読み込みモジュール。

処理概要: ファイルのバイト列を受け取り、セル値を保持した生グリッド（1-indexed 2次元配列）を構築する。
          数式評価・結合セル伝播を行う。テーブル検出・整形・分析は行わない。
入力    : bytes（Excel / CSV ファイルの内容）、ファイル名
出力    : List[SheetGrid]（シートごとの生グリッド）/ pd.DataFrame（CSV の場合）
"""

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openpyxl
import pandas as pd

from .models import SheetGrid


# ---------------------------------------------------------------------------
# 数式評価ヘルパー
# openpyxl を data_only=True でロードすると、Excel で一度も開かれていないファイルの
# 数式セルは None になる。そのままでは step2 にスカスカのグリッドが渡るため、
# ここで数式を自前評価してセル値を補完する（「ファイルの値を正確に取り出す」読み込みの一部）。
# ---------------------------------------------------------------------------

def _col_to_num(col: str) -> int:
    """'A'→1, 'Z'→26, 'AA'→27（大文字・小文字を区別しない）。"""
    n = 0
    for ch in col.upper():
        n = n * 26 + (ord(ch) - 64)
    return n


def _parse_cell_ref(ref: str) -> Tuple[Optional[int], Optional[int]]:
    """'D5', '$D$5' → (row=5, col=4)。失敗時は (None, None) を返す。"""
    m = re.match(r"\$?([A-Z]+)\$?(\d+)$", ref.strip().upper())
    if m:
        return int(m.group(2)), _col_to_num(m.group(1))
    return None, None


def _eval_formula(formula: str, grid: List[List[Any]]) -> Optional[float]:
    """
    構築済みの値 grid に対してシンプルな Excel 数式を評価する。

    サポートする形式:
      =SUM(A1:B3)              範囲の合計
      =SUM(A1:B3, C5:D7)      複数範囲の合計
      =A1                      単一セル参照
      =A1+B1  / =A1-B1  / =A1*B1   2セル間の四則演算
    数式が未サポートの場合、または必要なセルがまだ None の場合は None を返す。
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

    if re.fullmatch(r"\$?[A-Z]+\$?\d+", expr, re.IGNORECASE):
        r, c = _parse_cell_ref(expr)
        return _get(r, c) if r else None

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
    *ws_formulas*（data_only=False でロードされたワークシート）の Excel 数式を評価し、
    *grid* 内の None セルを埋める。

    小計が他の小計を合計するケースを正しく解決できるよう、最大5パス実行する。
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
            break


# ---------------------------------------------------------------------------
# グリッド構築
# ---------------------------------------------------------------------------

def _build_value_grid(ws, ws_formulas=None) -> Tuple[List[List[Any]], int, int]:
    """
    openpyxl のワークシートから 1-indexed の2次元 grid を構築する。

    - 結合セルの値は結合範囲全体に伝播される。
    - *ws_formulas* が指定された場合、キャッシュ値が None のセルを数式から再評価する。
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

    # openpyxl の内部表現では結合セルの値は先頭セルにしか存在しない。
    # 伝播させないと step2 が結合範囲を空セルと誤判定するため、ここで補正する。
    for merge_range in ws.merged_cells.ranges:
        top_val = grid[merge_range.min_row][merge_range.min_col]
        for r in range(merge_range.min_row, merge_range.max_row + 1):
            for c in range(merge_range.min_col, merge_range.max_col + 1):
                grid[r][c] = top_val

    if ws_formulas is not None:
        _fill_formula_cells(grid, max_row, max_col, ws_formulas)

    return grid, max_row, max_col


def _build_grid_from_xlrd(sheet) -> Tuple[List[List[Any]], int, int]:
    """xlrd のシートから 1-indexed の2次元 grid を構築する。"""
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
# 公開 API
# ---------------------------------------------------------------------------

def load_excel(content: bytes, filename: str) -> Tuple[List[SheetGrid], List[str]]:
    """Excel ファイルを読み込み、シートごとの SheetGrid リストと sheet_names を返す。"""
    ext = Path(filename).suffix.lower()
    sheets: List[SheetGrid] = []

    if ext in (".xlsx", ".xlsm", ".xlsb"):
        raw = io.BytesIO(content)
        wb_data = openpyxl.load_workbook(raw, data_only=True, read_only=False)
        raw.seek(0)
        wb_form = openpyxl.load_workbook(raw, data_only=False, read_only=False)
        sheet_names = wb_data.sheetnames

        for sheet_name in sheet_names:
            ws_data = wb_data[sheet_name]
            ws_form = wb_form[sheet_name] if sheet_name in wb_form.sheetnames else None
            grid, max_row, max_col = _build_value_grid(ws_data, ws_formulas=ws_form)
            sheets.append(SheetGrid(sheet_name=sheet_name, grid=grid, max_row=max_row, max_col=max_col))

    elif ext == ".xls":
        import xlrd

        wb = xlrd.open_workbook(file_contents=content)
        sheet_names = wb.sheet_names()

        for sheet_name in sheet_names:
            ws = wb.sheet_by_name(sheet_name)
            grid, max_row, max_col = _build_grid_from_xlrd(ws)
            sheets.append(SheetGrid(sheet_name=sheet_name, grid=grid, max_row=max_row, max_col=max_col))

    else:
        raise ValueError(f"Unsupported Excel format: {ext}")

    return sheets, sheet_names


def load_csv(content: bytes) -> pd.DataFrame:
    """CSV ファイルを読み込み、DataFrame を返す。"""
    for enc in ("utf-8-sig", "cp932", "shift_jis", "utf-8"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    return pd.read_csv(io.BytesIO(content), encoding="latin-1")
