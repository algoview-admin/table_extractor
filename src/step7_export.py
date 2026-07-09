"""
ステップ7 エクスポートロジック。

Streamlit に依存しない純粋な処理関数を提供する。
UI 層 (steps/step7_export.py) はこのモジュールの関数を呼び出す。
"""

import io
import zipfile
from typing import Dict, Tuple

import pandas as pd


def safe_filename(display_name: str) -> str:
    """表示名をファイル名として安全な文字列に変換する。"""
    return (
        display_name
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """DataFrame を UTF-8 BOM 付き CSV のバイト列に変換する。"""
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def build_export_zip(selected: Dict[str, dict]) -> Tuple[bytes, Dict[str, bytes]]:
    """選択テーブルを ZIP にまとめる。

    Returns:
        (zip_bytes, file_map)  — file_map は {filename: csv_bytes}
    """
    zip_buf = io.BytesIO()
    file_map: Dict[str, bytes] = {}

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for tid, info in selected.items():
            df: pd.DataFrame = info["df"]
            fname = f"{safe_filename(info['display_name'])}.csv"
            csv_bytes = df_to_csv_bytes(df)
            file_map[fname] = csv_bytes
            zf.writestr(fname, csv_bytes)

    zip_buf.seek(0)
    return zip_buf.getvalue(), file_map
