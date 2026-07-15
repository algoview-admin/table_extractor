"""
辞書ファイルローダー。

処理概要: src/dictionaries/ 以下の YAML ファイルを読み込み、
          各ステップが使う定数として公開する。
          モジュール初回インポート時に一度だけ評価される。
入力    : src/dictionaries/*.yaml
出力    : UNIT_VOCAB, AGG_KEYWORDS, UCHI_PREFIXES, NOTE_ROW_PREFIXES,
          TIME_PATTERNS, VAR_NAME_MAP, VALUE_KEYWORDS, NOTE_AGG_KEYWORDS,
          NOTE_EXCL_KEYWORDS, NOTE_REF_KEYWORDS, STAT_PLACEHOLDERS,
          STAT_NA_MARKERS
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import yaml

_DIR = Path(__file__).parent / "dictionaries"


def _load(name: str):
    path = _DIR / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 単位語彙 (step3 / stage1) ──────────────────────────────────────────────
# 比較は小文字で行うため、ロード時に全エントリを小文字化する
UNIT_VOCAB: FrozenSet[str] = frozenset(
    str(v).lower() for v in _load("unit_vocab.yaml")
)

# ── 集計キーワード (step3 / stage2) ────────────────────────────────────────
AGG_KEYWORDS: FrozenSet[str] = frozenset(_load("agg_keywords.yaml"))

# ── 内訳接頭辞 (step3 / stage2.5) ──────────────────────────────────────────
UCHI_PREFIXES: Tuple[str, ...] = tuple(_load("uchi_prefixes.yaml"))

# ── タイトル候補行の注記判定接頭辞 (step2 / 行分類) ────────────────────────
NOTE_ROW_PREFIXES: Tuple[str, ...] = tuple(_load("note_row_prefixes.yaml"))

# ── 時系列パターン (step3 / stage4) ────────────────────────────────────────
_time_raw = _load("time_patterns.yaml")

TIME_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(
            item["pattern"],
            getattr(re, item["flags"]) if "flags" in item else 0,
        ),
        item["kind"],
    )
    for item in _time_raw["patterns"]
]

VAR_NAME_MAP: Dict[str, str] = {
    item["kind"]: item["var_name"] for item in _time_raw["patterns"]
}

VAR_NAME_FALLBACK: str = _time_raw.get("fallback_var_name", "期間")

# ── 値列名推定キーワード (step3 / stage4) ──────────────────────────────────
VALUE_KEYWORDS: Dict[str, str] = _load("value_keywords.yaml")

# ── 注記分類キーワード (step5) ──────────────────────────────────────────────
_note_raw = _load("note_keywords.yaml")
NOTE_AGG_KEYWORDS: Tuple[str, ...] = tuple(_note_raw["aggregation"])
NOTE_EXCL_KEYWORDS: Tuple[str, ...] = tuple(_note_raw["exclusion"])
NOTE_REF_KEYWORDS: Tuple[str, ...] = tuple(_note_raw["reference"])

# ── 統計秘匿プレースホルダー (step2) ───────────────────────────────────────
STAT_PLACEHOLDERS: FrozenSet[str] = frozenset(
    str(v) for v in _load("stat_placeholders.yaml")
)

# ── 秘匿・欠損マーカー (step3 / stage2) ─────────────────────────────────────
# 比較は小文字化して行うため、ロード時に全エントリを小文字化する
STAT_NA_MARKERS: FrozenSet[str] = frozenset(
    str(v).lower() for v in _load("stat_na_markers.yaml")
)
