"""
ステップ6 テーブル選択モジュール。

処理概要: AI 分析結果と統合推奨をもとに、エクスポート対象テーブルの
          グルーピング・粒度バッジ生成・ファイル名変換を行う。
入力    : AIAnalysisResult、final_tables（統合済みテーブル辞書 Dict[str, dict]）
出力    : グループ分類辞書（統合・最小粒度・マスタ・推奨・非推奨の5区分）、
          バッジ HTML 文字列、安全なファイル名文字列
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd


def granularity_badge(info: dict) -> str:
    """テーブルの粒度に応じた HTML バッジ文字列を返す。"""
    if info["is_integrated"]:
        return "<span class='step-badge badge-integrated'>🔀 統合</span>"
    g = info.get("granularity", "unknown")
    if info.get("is_master"):
        return "<span class='step-badge badge-master'>📚 マスタ</span>"
    if info.get("is_minimum"):
        return "<span class='step-badge badge-detail'>⭐ 最小粒度</span>"
    if g == "summary":
        return "<span class='step-badge badge-summary'>📈 集計</span>"
    if g == "detail":
        return "<span class='step-badge badge-detail'>🔍 詳細</span>"
    return "<span class='step-badge badge-ref'>📄 その他</span>"


def group_final_tables(final: dict) -> Tuple[dict, dict, dict, dict, dict]:
    """final_tables を統合・最小粒度・マスタ・その他推奨・非推奨の5グループに分類して返す。

    Returns:
        (integrated, min_tables, master_tables, other_rec, non_rec)
    """
    integrated = {k: v for k, v in final.items() if v["is_integrated"]}

    min_tables = {
        k: v
        for k, v in final.items()
        if not v["is_integrated"] and v.get("is_minimum")
    }

    master_tables = {
        k: v
        for k, v in final.items()
        if not v["is_integrated"] and not v.get("is_minimum") and v.get("is_master")
    }

    shown = set(integrated) | set(min_tables) | set(master_tables)

    other_rec = {
        k: v
        for k, v in final.items()
        if k not in shown and v.get("recommended") and not v["is_integrated"]
    }

    shown |= set(other_rec)
    non_rec = {k: v for k, v in final.items() if k not in shown}

    return integrated, min_tables, master_tables, other_rec, non_rec


def group_integrated_by_columns(integrated: dict) -> List[List[Tuple[str, dict]]]:
    """統合テーブルを列シグネチャでグループ化する。代表を先頭に、類似を後続に並べる。"""
    groups: List[List[Tuple[str, dict]]] = []
    sig_to_group: Dict[frozenset, List[Tuple[str, dict]]] = {}
    for tid, info in integrated.items():
        df = info.get("df")
        sig = frozenset(df.columns) if df is not None else frozenset()
        if sig not in sig_to_group:
            group: List[Tuple[str, dict]] = [(tid, info)]
            groups.append(group)
            sig_to_group[sig] = group
        else:
            sig_to_group[sig].append((tid, info))
    return groups


def safe_table_filename(display_name: str) -> str:
    """テーブルの表示名をファイル名として安全な文字列に変換する。"""
    return (
        display_name
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )
