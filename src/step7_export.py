"""
ステップ7 エクスポートモジュール。

処理概要: 選択済みテーブルを CSV に変換し、ZIP アーカイブとして梱包する。
          集計除去の監査用メタデータ（agg_removed_row_metadata /
          agg_removed_col_metadata）が付与されたテーブルは
          同名の _metadata.json も同梱する。
入力    : Dict[str, dict]（選択済みテーブル。キー = テーブル ID、値 = df・display_name・
          任意で agg_removed_row_metadata / agg_removed_col_metadata（List[dict]）
          等を含む辞書）
出力    : ZIP バイト列（全テーブルをまとめた一括ダウンロード用）、
          Dict[str, bytes]（ファイル名 → 個別バイト列。CSV と、存在すればメタデータ JSON）
"""

import io
import json
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


def metadata_to_json_bytes(
    agg_removed_row_metadata: list, agg_removed_col_metadata: list
) -> bytes:
    """集計除去の監査用メタデータ（行・列）を JSON バイト列に変換する。"""
    return json.dumps(
        {
            "aggregate_rows_removed": agg_removed_row_metadata,
            "aggregate_columns_removed": agg_removed_col_metadata,
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    ).encode("utf-8")


def build_export_zip(selected: Dict[str, dict]) -> Tuple[bytes, Dict[str, bytes]]:
    """選択テーブルを ZIP にまとめる。

    各テーブルの info に agg_removed_row_metadata / agg_removed_col_metadata
    （除去した集計行・集計列の監査用メタデータ）のいずれかが含まれる場合は、
    同名の "<表示名>_metadata.json" も同梱する。

    Returns:
        (zip_bytes, file_map)  — file_map は {filename: bytes}（CSV とメタデータ JSON）
    """
    zip_buf = io.BytesIO()
    file_map: Dict[str, bytes] = {}

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for tid, info in selected.items():
            df: pd.DataFrame = info["df"]
            safe_name = safe_filename(info["display_name"])

            csv_fname = f"{safe_name}.csv"
            csv_bytes = df_to_csv_bytes(df)
            file_map[csv_fname] = csv_bytes
            zf.writestr(csv_fname, csv_bytes)

            row_meta = info.get("agg_removed_row_metadata") or []
            col_meta = info.get("agg_removed_col_metadata") or []
            if row_meta or col_meta:
                meta_fname = f"{safe_name}_metadata.json"
                meta_bytes = metadata_to_json_bytes(row_meta, col_meta)
                file_map[meta_fname] = meta_bytes
                zf.writestr(meta_fname, meta_bytes)

    zip_buf.seek(0)
    return zip_buf.getvalue(), file_map
