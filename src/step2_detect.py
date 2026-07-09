"""
ステップ2 テーブル検出ロジック。

Streamlit に依存しない純粋な処理関数を提供する。
UI 層 (steps/step2_detect.py) はこのモジュールの関数を呼び出す。
"""

from typing import List, Optional

import pandas as pd

from .models import DetectedTable


def get_original_df(t: DetectedTable) -> Optional[pd.DataFrame]:
    """整形処理適用前の生 DataFrame を返す。

    優先順位: 多段ヘッダー統合前 → ffill 前 → 集計除去前 → 最終 df
    """
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
