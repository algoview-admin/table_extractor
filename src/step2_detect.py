"""
ステップ2 テーブル検出モジュール。

処理概要: 生グリッド（List[SheetGrid]。Excel も CSV も step1_upload が同じ grid 形式で
          返す）を受け取り、行分類ステートマシンでテーブル領域を特定する。
          ヘッダー行が何行あるか・どの行が name/unit かという構造の認識まではここで行うが、
          複数ヘッダー行を1つの列名へ統合する・軸展開するといった列名の整形は一切行わない
          （表を「表として認識する」ことと「表を整形する」ことを分離するため）。
          整形は src/step3_normalize_determ.py の normalize_tables() が Step3 側で適用する。
入力    : List[SheetGrid]（step1_upload が構築した生グリッド）
出力    : List[DetectedTable]（検出されたテーブルごとの位置・DataFrame・タイトル・注記を含む。
          整形前の生 DataFrame のみを保持する）
"""

import re
from itertools import groupby
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .keywords import (
    NOTE_ROW_PREFIXES as _NOTE_PREFIXES,
    STAT_PLACEHOLDERS as _STAT_PLACEHOLDERS,
    UNIT_VOCAB,
)
from .models import DetectedTable, SheetGrid


# ---------------------------------------------------------------------------
# ヘッダー行の構造認識（name/unit の役割判定のみ。列名の統合は行わない）
# ---------------------------------------------------------------------------
# ここではテーブル領域のうち何行がヘッダーで何行からがデータかを判定する
# ためだけに name/unit の役割を使う（構造認識であり、値の意味理解は不要）。
# 複数ヘッダー行を実際に1つの列名へ統合する処理（多段ヘッダーの検出と
# 解決機能・単純統合／軸展開）は Step3 の step3_normalize_determ.py が行う。


_UNIT_PERIOD_SUFFIX_RE = re.compile(
    r"[／/](日|月|年|週|時|day|month|year|week|hour|hr)$", re.IGNORECASE
)


def _strip_unit_period_suffix(v: str) -> str:
    """「立方メートル／日」「cubic meters/day」のような単位＋期間表記から
    期間部分を取り除く（"立方メートル"/"cubic meters" を残す）。
    UNIT_VOCAB には基本単位のみを登録すればよく、期間との組み合わせを
    総当たりで列挙しなくても済むようにするための正規化。"""
    return _UNIT_PERIOD_SUFFIX_RE.sub("", v)


def _is_unit_row(row: List[Any]) -> bool:
    """行が単位ラベルのみで構成されているかを判定する。

    以下の条件をすべて満たす場合に True を返す:
      - 全ての非空セルが 25 文字以内（単位は必ず短い）
      - 非空セルの 50% 以上が既知単位語彙に一致（「立方メートル／日」のような
        期間サフィックス付きは基本単位部分で照合）、
        または全セルが 15 文字以内かつ繰り返し率が高い（同一単位が複数列に並ぶ）
    """
    vals = [
        str(v).strip()
        for v in row
        if v is not None and str(v).strip() and str(v).strip().lower() != "nan"
    ]
    if not vals:
        return False
    if any(len(v) > 25 for v in vals):
        return False
    vocab_hits = sum(
        1
        for v in vals
        if v.lower() in UNIT_VOCAB
        or _strip_unit_period_suffix(v.lower()) in UNIT_VOCAB
    )
    if vocab_hits / len(vals) >= 0.5:
        return True
    # 繰り返し率が高い場合の補助判定（「百万円」が複数列に並ぶケース等）。
    # ただし語彙ヒットが 0 件の場合は「年初在庫額」「土地以外のもの」のような
    # サブカラム名と区別できないため除外する。
    unique_ratio = len({v.lower() for v in vals}) / len(vals)
    return vocab_hits > 0 and unique_ratio <= 0.4 and all(len(v) <= 15 for v in vals)


def detect_header_roles(rows: List[List[Any]]) -> Tuple[int, List[str]]:
    """先頭のタイトル行と列ヘッダー行を検出する。

    Returns (n_title, header_roles):
      n_title      — 先頭にあるセクションタイトルの行数。
      header_roles — ヘッダー行ごとの役割リスト（"name" または "unit"）。
                     空リストはヘッダー行なし。

    連続するヘッダー行を最大 _MAX_HEADER_ROWS 行まで検出し、各行を分類する:
      "name" — 列ラベル行（日本語名、英語名など）
      "unit" — 単位ラベル行（人、百万円、mil. yen など）

    2 番目以降の "name" 行を受け入れる条件:
      - 直前が "unit" 行（別言語の名前層）、または
      - 最初の行に重複値がある（結合セルスパン型の多段ヘッダー）
    """
    if not rows:
        return 0, []

    def _nn(row: List[Any]) -> List[Any]:
        return [v for v in row if v is not None]

    def _str_count(vals: List[Any]) -> int:
        return sum(1 for v in vals if isinstance(v, str))

    def _num_count(vals: List[Any]) -> int:
        return sum(
            1 for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)
        )

    # --- ステップ1: 先頭のタイトル行をスキップ ---
    n_title = 0
    while n_title < len(rows) - 1:
        nn_curr = _nn(rows[n_title])
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
        has_wider_below = any(
            len(_nn(rows[j])) >= 2 for j in range(n_title + 1, len(rows))
        )
        if has_wider_below:
            n_title += 1
        else:
            break

    remaining = rows[n_title:]
    if not remaining:
        return n_title, []

    # --- ステップ2: 最初の行がヘッダーか確認（文字列比率 50% 以上）---
    nn_first = _nn(remaining[0])
    if not nn_first or _str_count(nn_first) / len(nn_first) < 0.5:
        return n_title, []

    # 1 行目に重複値があるか（結合セルスパン型の判定）
    first_strs = [str(v) for v in nn_first]
    first_has_dups = len(first_strs) != len(set(first_strs))

    # --- ステップ3: 連続するヘッダー行を検出 ---
    # 日本語5段＋英語5段のような深いヘッダー（10行）に対応するため余裕をもたせる
    _MAX_HEADER_ROWS = 12
    roles: List[str] = []

    for i, row in enumerate(remaining[:_MAX_HEADER_ROWS]):
        nn = _nn(row)
        if not nn:
            break
        num_cnt = _num_count(nn)
        if num_cnt / len(nn) >= 0.40:
            break
        role = "unit" if _is_unit_row(row) else "name"
        if i > 0 and role == "name":
            # 2 番目以降の name 行: 直前が unit か、または結合セルスパン型のみ受け入れる
            if roles[-1] != "unit" and not first_has_dups:
                break
        roles.append(role)

    if not roles or roles[0] != "name":
        return n_title, []

    return n_title, roles


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
            """
            全セルが同一テキスト＝結合セルの複製（openpyxl の結合範囲展開で
            同じ値が全列に複製される）である可能性が高い。この場合は列数に
            関わらず文字列長のみで判定する（列数で col_hdr 側に倒すと、
            4列以上に結合されたタイトル行が列ヘッダーと誤認識されてしまう）。
            """
            txt = next(iter(unique))
            if txt.startswith(_NOTE_PREFIXES) or len(txt) > 60:
                return _RT_NOTE
            if _AGG_ENUM_RE.search(txt):
                return _RT_NOTE
            if p["col_min"] is not None and p["col_min"] > 1:
                """
                先頭列（ラベル列想定）が空白で、途中の列から結合テキストが
                始まっている場合は、行全体を覆うドキュメントタイトルではなく
                「販売実績（AP数）」のような値列だけをまとめる列グループ見出し
                （多段ヘッダーの一部）である可能性が高いため、列ヘッダーとして扱う。
                """
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
    row_widths: Optional[List[int]] = None,
    content_rows: Optional[Set[int]] = None,
) -> List[Tuple[int, int]]:
    """行スパン内で列の空白を境に (start_col, end_col) のペアを返す。

    末尾の空列（この行スパン内で最後にデータがある列より右側）は、この
    テーブル自身の行が本来持っていた幅までは無条件に最後のグループへ含める
    （黙って切り捨てると、不可逆な列削除がユーザー確認なしに Step2 で
    行われてしまい、Step3 の無効カラム検出・削除機能（ユーザー確認必須）を
    経由できなくなるため）。実際に削除するかどうかの判断は Step3 側の
    ユーザー確認に委ねる。

    ここでの max_col はシート全体で共有される絶対列位置の上限であり、
    シート内の別の行帯（別テーブル）がたまたま広い場合にそこまで含めて
    しまうと、無関係な列を巻き込んでしまう（列は全テーブルで共通の絶対
    座標のため）。row_widths（CSV由来。行ごとの本来のフィールド数）が
    渡された場合は、この行帯自身の行が実際に持っていた最大幅までしか
    拡張しない。row_widths が無い場合（Excel。行ごとの「本来の幅」という
    概念がない）は、内側の空列と同じ許容量（gap_threshold 未満）だけ
    保守的に末尾へ残す。

    内側の空列は、従来通り gap_threshold 以上連続した場合のみ別テーブル
    領域として分割する（側並びの複数テーブルを区別するため）。

    content_rows が指定された場合、列の充填有無はこの行集合（通常は
    ヘッダー行＋データ行）のみで判定する。タイトル行・注記行は結合セルの
    複製で行全体（本来のテーブル幅より広い範囲）に同一テキストが埋まる
    ことがあり、これを列充填の根拠にすると実データの無い列（例:
    タイトルのマージ範囲がテーブル本体より1列広いだけの列）が本物の列と
    誤認識され、後続処理で存在しない列名（例: 重複列名への連番付与）が
    生成されてしまう。指定がない場合は従来通り行帯全体を対象にする。
    """
    col_has_content = [False] * (max_col + 1)
    scan_rows = content_rows if content_rows is not None else range(start_row, end_row + 1)
    for r in scan_rows:
        if r < start_row or r > end_row:
            continue
        for c in range(1, max_col + 1):
            if _is_filled(grid[r][c]):
                col_has_content[c] = True

    last_content_col = max(
        (c for c in range(1, max_col + 1) if col_has_content[c]), default=0
    )
    if last_content_col == 0:
        return []

    groups: List[Tuple[int, int]] = []
    group_start: Optional[int] = None
    empty_streak = 0

    # 末尾の空列は区切りになり得ないため、実データがある最後の列までのみを
    # 対象に gap_threshold 判定を行う（それより右は下記で個別に処理する）。
    for c in range(1, last_content_col + 1):
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
        groups.append((group_start, last_content_col))

    if groups and row_widths is not None and last_content_col < max_col:
        # CSV: 行が実際に持っていたフィールド数という実証拠があるためのみ延長する。
        # Excel はこの根拠を持たない（末尾列が content_rows で無内容と判定された
        # 時点で、ヘッダー文字列すら存在しない = 延長を正当化する材料が原理的に
        # 存在しない）ため、以前あった「保守的に1列だけ延長する」処理は行わない。
        # この延長は、タイトル行の結合セルがテーブル本体より広い場合に
        # 実データのない列（例: 存在しない列名 "○○_1"）を生んでいた
        # （sheet全体の max_col がタイトル行の結合範囲だけで押し上げられるため）。
        local_max_col = max(
            (row_widths[r] for r in range(start_row, min(end_row, len(row_widths) - 1) + 1)),
            default=last_content_col,
        )
        trailing_end = max(last_content_col, min(local_max_col, max_col))
        gs, _ge = groups[-1]
        groups[-1] = (gs, trailing_end)

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
            # 結合セルの複製で同一テキストが複数列に並ぶ場合があるため、
            # _RT_NOTE と同様に重複排除してから結合する。
            unique_title_texts = list(dict.fromkeys(p["texts"]))
            text = " ".join(unique_title_texts)
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

                # 既存の救済条件（row_width/tbl_width<0.5 or fill_density<0.30）は
                # 「表の一部だけを埋める疎な副見出し行」を想定している。だが
                # (キー列,属性名,値) のような狭い縦持ち表では、値が非数値
                # （区分名・評点・月表記等）の行は毎回 filled_in_span=tbl_width の
                # 全列充填になり、この救済条件に一切引っかからず、データ行が
                # 次々と新規ヘッダーとして誤検出され表が細切れになる。
                # 表幅が狭く（tbl_width<=4）候補行が確立済みの表幅を過不足なく
                # 満たしている場合は、疎な副見出しとは逆の「既存データ行と同じ
                # 形状」を示す強いシグナルのため、データ行として救済する。
                is_narrow_full_row = tbl_width <= 4 and fill_density >= 0.9

                if tbl_width > 0 and (
                    row_width / tbl_width < 0.5
                    or fill_density < 0.30
                    or is_narrow_full_row
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
                elif pending_titles:
                    # 直前に検出済みのタイトル行に続く説明的な注記行（例:「※CRM
                    # システムからエクスポート」）。まだ次のテーブルのヘッダー/
                    # データ行が始まっていない段階なので、これは直前に完了した
                    # 別テーブルの末尾注記ではなく、これから始まるテーブルの
                    # タイトルに付随する説明文である。複数行タイトルの結合と
                    # 同様に pending_titles に連結し、正しいテーブルに帰属させる。
                    pending_titles.append((r, note_text))
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
) -> Tuple[
    pd.DataFrame,
    Optional[str],
    Optional[pd.DataFrame],
    Optional[List[List[Any]]],
    Optional[List[str]],
]:
    """矩形領域を DataFrame として抽出する。

    Returns (df, title, raw_df, raw_header_rows, raw_header_roles)

    ヘッダー行が2行以上ある場合、df の列名は暫定的に先頭ヘッダー行のみを
    使う（列名の統合という整形判断はStep2では行わないため）。実際の列名
    統合（単純統合／軸展開）は raw_header_rows/raw_header_roles を使って
    Step3 の多段ヘッダーの検出と解決機能（step3_normalize_determ.py）が行う。
    """
    if start_row > end_row or start_col > end_col:
        return pd.DataFrame(), None, None, None, None

    rows = [
        [grid[r][c] for c in range(start_col, end_col + 1)]
        for r in range(start_row, end_row + 1)
    ]

    if not rows:
        return pd.DataFrame(), None, None, None, None

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
        return df.dropna(how="all").reset_index(drop=True), title, None, None, None

    if n_header == 1:
        header = _make_unique_columns(
            [str(v) if v is not None else "" for v in remaining[0]]
        )
        df = pd.DataFrame(remaining[1:], columns=header)
        return df.dropna(how="all").reset_index(drop=True), title, None, None, None

    raw_header = _make_unique_columns(
        [str(v) if v is not None else "" for v in remaining[0]]
    )
    raw_df = pd.DataFrame(remaining[1:], columns=raw_header).dropna(how="all").reset_index(drop=True)

    header_data = [remaining[i] for i in range(n_header)]
    # 列名統合はStep3の仕事なので、ここでは先頭ヘッダー行のみを暫定列名にする
    # （n_header==1 の場合と同じ扱い）。データ開始位置はヘッダー行数を正しく
    # 反映する（先頭行以外のヘッダー行をデータに混入させないため）。
    header = raw_header
    df = pd.DataFrame(remaining[n_header:], columns=header)

    raw_header_rows = [list(row) for row in header_data] if n_header >= 2 else None
    raw_header_roles = list(header_roles) if n_header >= 2 else None

    return (
        df.dropna(how="all").reset_index(drop=True),
        title,
        raw_df,
        raw_header_rows,
        raw_header_roles,
    )


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


def _detect_tables_in_grid(
    grid: List[List[Any]],
    max_row: int,
    max_col: int,
    sheet_name: str,
    table_counter: Dict[str, int],
    row_widths: Optional[List[int]] = None,
) -> List[DetectedTable]:
    """グリッドからテーブルを検出して DetectedTable のリストを返す。"""
    if max_row == 0 or max_col == 0:
        return []

    regions = _detect_table_regions(grid, max_row, max_col)
    detected: List[DetectedTable] = []

    for reg in regions:
        band_start = reg["band_start"]
        band_end = reg["band_end"]

        content_rows = set(reg["header_rows"]) | set(reg["data_rows"])
        col_groups = _find_column_groups(
            grid,
            band_start,
            band_end,
            max_col,
            gap_threshold=2,
            row_widths=row_widths,
            content_rows=content_rows,
        )

        first_header_row = min(reg["header_rows"]) if reg["header_rows"] else band_start

        for col_start, col_end in col_groups:
            if (band_end - first_header_row + 1) < 2:
                continue

            df, inner_title, raw_df, raw_header_rows, raw_header_roles = _extract_dataframe(
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

            detected.append(
                DetectedTable(
                    table_id=table_id,
                    sheet_name=sheet_name,
                    start_row=first_header_row,
                    end_row=band_end,
                    start_col=col_start,
                    end_col=col_end,
                    df=df,
                    title=effective_title,
                    notes=reg.get("trailing_notes", []),
                    raw_df=raw_df,
                    raw_header_rows=raw_header_rows,
                    raw_header_roles=raw_header_roles,
                )
            )

    _propagate_sheet_title(detected)
    return detected


# ---------------------------------------------------------------------------
# ファイル単位のテーブル検出エントリポイント（Excel経路 / CSV経路）
# ---------------------------------------------------------------------------

def detect_tables(
    sheets: List[SheetGrid],
) -> Tuple[List[DetectedTable], List[str]]:
    """SheetGrid のリストからテーブルを検出する（整形処理は行わない）。"""
    all_tables: List[DetectedTable] = []
    table_counter: Dict[str, int] = {}

    for sheet in sheets:
        tables = _detect_tables_in_grid(
            sheet.grid, sheet.max_row, sheet.max_col,
            sheet.sheet_name, table_counter,
            row_widths=sheet.row_widths,
        )
        all_tables.extend(tables)

    sheet_names = [s.sheet_name for s in sheets]
    return all_tables, sheet_names


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
