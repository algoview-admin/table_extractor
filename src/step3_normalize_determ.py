"""
ステップ3 テーブル正規化モジュール（決定論的処理）。

処理概要: 検出された生 DataFrame を分析に適した形式に正規化する。
          多段ヘッダーの統合・グルーピング列の補完・「うち」書きの内訳
          別テーブル分離・集計行列の除去・単位混在の分離（指標マスタ生成）・
          クロス集計/Wide_to_longの縦持ち変換など、正規表現・語彙辞書のみで
          完結する（LLMを使わない）処理を扱う。
          LLMを使用する処理（Transpose検出等）は
          src/step3_normalize_llm.py を参照。
          normalize_tables() が両者を正しい順序で呼び出す統括関数。
入力    : DetectedTable.df（step2_detect が構築した生 DataFrame）
出力    : 正規化済み DataFrame、整形メタ情報
          （transpose_info, filled_cols, uchi_split_info, stack_info,
          agg_rows_removed, unit_split_info 等を DetectedTable に付与）
"""

import re as _re
from typing import Any, Dict, List, Optional, Tuple

from .keywords import (
    AGG_KEYWORDS,
    AXIS_DELIMITERS,
    STAT_NA_MARKERS as _STAT_NA_MARKERS,
    TIME_PATTERNS as _TIME_PATTERNS,
    UCHI_PREFIXES,
    UNIT_VOCAB,
    VALUE_KEYWORDS as _VALUE_KEYWORDS,
    VAR_NAME_MAP as _VAR_NAME_MAP,
    VAR_NAME_FALLBACK as _VAR_NAME_FALLBACK,
)
from .step3_normalize_llm import (
    apply_transpose,
    detect_category_axis,
    detect_dimension_axes,
    detect_transpose,
    make_transpose_client,
)


def _is_unit_row(row: List[Any]) -> bool:
    """行が単位ラベルのみで構成されているかを判定する。

    以下の条件をすべて満たす場合に True を返す:
      - 全ての非空セルが 25 文字以内（単位は必ず短い）
      - 非空セルの 50% 以上が既知単位語彙に一致、
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
    vocab_hits = sum(1 for v in vals if v.lower() in UNIT_VOCAB)
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


def _detect_row_language(row: List[Any], n_cols: int) -> str:
    """ヘッダー行の主言語を検出する。'ja'、'en'、'other' のいずれかを返す。

    非空セルの過半数に CJK 文字（ひらがな・カタカナ・漢字）が含まれる場合は 'ja'、
    ASCII アルファベットのみで構成されるセルが過半数の場合は 'en'、
    それ以外は 'other' を返す。
    """

    def _v(ci: int) -> str:
        if ci < len(row) and row[ci] is not None:
            s = str(row[ci]).strip()
            return s if s and s.lower() != "nan" else ""
        return ""

    texts = [_v(ci) for ci in range(n_cols) if _v(ci)]
    if not texts:
        return "other"

    def _has_cjk(s: str) -> bool:
        return any("぀" <= c <= "鿿" or "豈" <= c <= "﫿" for c in s)

    cjk_cells = sum(1 for t in texts if _has_cjk(t))
    ascii_cells = sum(
        1
        for t in texts
        if not _has_cjk(t) and any(c.isalpha() and c.isascii() for c in t)
    )

    if cjk_cells / len(texts) >= 0.5:
        return "ja"
    if ascii_cells / len(texts) >= 0.5:
        return "en"
    return "other"


def merge_header_rows(
    header_data: List[List[Any]],
    roles: List[str],
    n_cols: int,
) -> List[str]:
    """複数のヘッダー行を 1 つの列名リストにマージする。

    roles が全て "name" の場合（結合セルスパン型）: アンダースコアで連結する。
      例: ["東京支社", "売上"] → "東京支社_売上"

    "unit" ロールを含む場合（名前+単位型）: name の直後の unit を括弧内に付記する。
    複数の name+unit ペアが存在する場合（多言語ヘッダー）は、最初の name 行の言語を
    主言語とみなし、その言語に一致するペアのみを使用する。
      例（日本語主言語）: ["従業者数","人","Number of persons","persons"] → "従業者数[人]"
    """

    def _get(row: List[Any], ci: int) -> str:
        if ci < len(row) and row[ci] is not None:
            v = str(row[ci]).strip()
            return v if v and v.lower() != "nan" else ""
        return ""

    if all(r == "name" for r in roles):
        columns = []
        for ci in range(n_cols):
            parts = [p for p in (_get(row, ci) for row in header_data) if p]
            columns.append("_".join(parts) if parts else f"列{ci + 1}")
        return columns

    # name+unit ペアリング
    pairs: List[Tuple[List[Any], Optional[List[Any]]]] = []
    i = 0
    while i < len(roles):
        if roles[i] == "name":
            if i + 1 < len(roles) and roles[i + 1] == "unit":
                pairs.append((header_data[i], header_data[i + 1]))
                i += 2
            else:
                pairs.append((header_data[i], None))
                i += 1
        else:
            pairs.append((header_data[i], None))
            i += 1

    # 複数ペアが存在する場合、主言語（最初の明確な言語）のペアのみを選択する
    if len(pairs) > 1:
        pair_langs = [_detect_row_language(name_row, n_cols) for name_row, _ in pairs]
        unique_clear = {l for l in pair_langs if l != "other"}
        if len(unique_clear) > 1:
            # 複数言語が混在 → 最初の明確な言語を主言語とする
            dominant = next((l for l in pair_langs if l != "other"), None)
            if dominant is not None:
                filtered = [
                    p for p, l in zip(pairs, pair_langs) if l in (dominant, "other")
                ]
                if filtered:
                    pairs = filtered

    columns = []
    for ci in range(n_cols):
        parts: List[str] = []
        # 名前セルが空だったペアの単位。ループ後に最後の名前パーツへ付加する。
        orphan_unit = ""
        for name_row, unit_row in pairs:
            name = _get(name_row, ci)
            unit = _get(unit_row, ci) if unit_row is not None else ""
            if name and unit:
                orphan_unit = ""
                parts.append(f"{name} [{unit}]")
            elif name:
                if orphan_unit:
                    parts.append(f"{name} [{orphan_unit}]")
                    orphan_unit = ""
                else:
                    parts.append(name)
            elif unit:
                """
                この列の名前行に名前がなく、単位行のみに値がある。
                直前に収集した名前パーツへ後で付加できるよう保持する。
                """
                orphan_unit = unit
        # 残った孤立単位を最後の名前パーツへ付加する。
        if orphan_unit and parts and " [" not in parts[-1]:
            parts[-1] = f"{parts[-1]} [{orphan_unit}]"
        columns.append("_".join(parts) if parts else f"列{ci + 1}")

    return columns


# ---------------------------------------------------------------------------
# 多軸ヘッダー展開（多段ヘッダーの検出と解決機能）
# ---------------------------------------------------------------------------
#
# merge_header_rows は複数のヘッダー行を単純にアンダースコアで連結するだけ
# だが、多段ヘッダーが「科目×支店」のように独立した複数のカテゴリ軸を
# 交差させたクロス集計形式の場合は、本来 tidy 形式（各軸を列として展開し
# 値を縦持ちに変換）に解決すべきである。
#
# 各ヘッダー行が (a) 単一値のみ（注記的な見出し。軸ではない）、
# (b) 時系列パターンに一致（既存の TIME_PATTERNS を流用、決定論的に判定可）、
# (c) 複数の異なる値がバランスよく並ぶ軸候補、のいずれかを構造的に分類する
# （意味理解は不要）。(c) が残る場合のみ、それが本当に独立した軸として
# 意味を持つか・軸名として何が適切かは値の意味を理解する必要があるため
# LLM 判定を行う（src/step3_normalize_llm.py の detect_dimension_axes）。


def _ffill_header_row(row: List[Any]) -> List[Any]:
    """ヘッダー行内の空白セルを左から右へ前方補完する。

    実際に結合されたセル（XLSX）は読み込み時点で複製済みのため冪等に働き、
    CSV等で空白のまま連結を表現している場合はこれで復元する。
    """
    out = list(row)
    last: Any = None
    for i, v in enumerate(out):
        s = str(v).strip() if v is not None else ""
        if s and s.lower() != "nan":
            last = v
        else:
            out[i] = last
    return out


def _header_row_kind(
    ffilled_row: List[Any], value_positions: Optional[set] = None
) -> Tuple[str, List[str]]:
    """前方補完済みヘッダー行を分類する。

    value_positions が指定された場合、その列位置のセルのみを対象にする
    （ラベル列の判定に使う。最終行＝リーフ行はラベル列にも実名が入っている
    ことがあるため、外側の行が空白だった列を除外しないと、ラベル列名が
    軸候補の値として混入してしまう）。

    Returns (kind, values):
      kind   — "empty" / "constant" / "time" / "candidate" / "irregular"
      values — 重複排除済みの値リスト（出現順）
    """
    non_blank = [
        str(v).strip()
        for ci, v in enumerate(ffilled_row)
        if (value_positions is None or ci in value_positions)
        and v is not None
        and str(v).strip()
        and str(v).strip().lower() != "nan"
    ]
    if not non_blank:
        return "empty", []

    seen: List[str] = []
    for v in non_blank:
        if v not in seen:
            seen.append(v)
    if len(seen) == 1:
        return "constant", seen

    from collections import Counter as _Counter

    time_kinds = [_classify_col_time(v) for v in seen]
    matched = [k for k in time_kinds if k is not None]
    if matched:
        dominant_kind, cnt = _Counter(matched).most_common(1)[0]
        if cnt / len(seen) >= _WIDE_TO_LONG_MATCH_RATIO:
            return "time", seen

    # 値の種類数が非空セル数に対して多すぎる（≒繰り返しがほぼ無く、各セルが
    # ほぼ一意）場合は、共有カテゴリを表す軸ではなく通常のラベル列名の羅列
    # である可能性が高いため対象外とする安全弁（「累計」等、他の値より出現
    # 回数が少ない値が混在すること自体は許容する）。
    if len(seen) > len(non_blank) / 2:
        return "irregular", seen

    return "candidate", seen


def _dedup_columns(columns: List[str]) -> List[str]:
    """列名重複時に連番を付与する（step2_detect._make_unique_columns と同等。
    循環importを避けるためここに複製）。"""
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


def detect_multi_axis_header(raw_header_rows: List[List[Any]]) -> Optional[Dict[str, Any]]:
    """多段ヘッダーの各行を構造的に分類し、除外可能な行・LLM判定が必要な
    軸候補行を切り分ける（決定論的、値の意味理解は不要）。

    2行未満、または分類不能な行（irregular）を含む場合は None を返し、
    呼び出し側は従来通りの単純連結（merge_header_rows）にフォールバックする。
    """
    if len(raw_header_rows) < 2:
        return None

    ffilled = [_ffill_header_row(r) for r in raw_header_rows]

    # 値列範囲の判定: 最終行（リーフ行）以外のいずれかの行で内容がある列のみを
    # 対象にする。リーフ行はラベル列にも実名（例: "帳票支店名"）を持つことが
    # あり、それを軸候補の値として誤って拾わないようにするため。
    num_cols = len(ffilled[0])
    value_positions = set()
    for r in ffilled[:-1]:
        for ci, v in enumerate(r):
            if v is not None and str(v).strip() and str(v).strip().lower() != "nan":
                value_positions.add(ci)
    if not value_positions:
        value_positions = set(range(num_cols))

    analyzed = [_header_row_kind(r, value_positions) for r in ffilled]
    kinds = [k for k, _ in analyzed]
    values = [v for _, v in analyzed]

    if any(k == "irregular" for k in kinds):
        return None

    leaf_time_idx: Optional[int] = None
    for i, k in enumerate(kinds):
        if k == "time":
            leaf_time_idx = i  # 最も下段（最も細かい粒度）の time 行を採用

    dropped = {i for i, k in enumerate(kinds) if k in ("constant", "empty")}
    if leaf_time_idx is not None:
        for i, k in enumerate(kinds):
            if i == leaf_time_idx:
                continue
            if k == "time":
                # leaf以外の time 行（稀）はより粗い粒度の冗長グルーピングとみなす
                dropped.add(i)
            elif k == "candidate" and len(values[i]) < len(values[leaf_time_idx]):
                # leaf行より値の種類数が少ない候補行は、leafの冗長な上位グルーピング
                # （例: 上期/下期 が 4月〜3月 の上位区分）とみなし除外する
                dropped.add(i)

    candidate_idxs = [
        i for i, k in enumerate(kinds) if k == "candidate" and i not in dropped
    ]

    if leaf_time_idx is not None and candidate_idxs:
        # 時系列軸と未解決の軸候補が混在する複合ケースは対象外とし、
        # 誤変換を避けるため安全側（フォールバック）に倒す
        return None

    return {
        "ffilled_rows": ffilled,
        "kinds": kinds,
        "values": values,
        "leaf_time_idx": leaf_time_idx,
        "dropped_idxs": dropped,
        "candidate_idxs": candidate_idxs,
    }


def apply_multi_axis_header(
    df: Any,
    info: Dict[str, Any],
    axis_result: Dict[str, Any],
) -> Any:
    """LLMが確定した軸名・値列名で、多軸クロス集計形式の多段ヘッダーを
    tidy形式（各軸を列として展開し値を縦持ちに変換）に展開する。
    """
    import pandas as pd

    ffilled = info["ffilled_rows"]
    candidate_idxs = info["candidate_idxs"]
    axis_names = axis_result["axis_names"]
    value_name = axis_result["value_name"]

    num_cols = len(ffilled[0])
    label_positions = [
        ci
        for ci in range(num_cols)
        if all(
            ffilled[ri][ci] is None or str(ffilled[ri][ci]).strip() == ""
            for ri in candidate_idxs
        )
    ]
    data_positions = [ci for ci in range(num_cols) if ci not in label_positions]
    label_col_names = [df.columns[ci] for ci in label_positions]

    # ラベル列がヘッダー行に一切名前を持たない場合、列名は "列N" のプレース
    # ホルダーのままになる（merge_header_rows のフォールバック）。データが
    # 時系列パターン（既存 TIME_PATTERNS）に一致する場合は、その軸名を
    # 決定論的に採用する（例: "2024年","2025年" → "年"）。
    renamed_label_names = []
    for name in label_col_names:
        new_name = name
        if _re.match(r"^列\d+$", str(name)):
            col_values = [
                str(v).strip() for v in df[name].dropna().unique() if str(v).strip()
            ]
            time_kinds = {_classify_col_time(v) for v in col_values}
            if col_values and len(time_kinds) == 1 and None not in time_kinds:
                new_name = _VAR_NAME_MAP.get(next(iter(time_kinds)), _VAR_NAME_FALLBACK)
        renamed_label_names.append(new_name)
    label_col_names = renamed_label_names

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        base = {name: row.iloc[pos] for name, pos in zip(label_col_names, label_positions)}
        for ci in data_positions:
            rec = dict(base)
            for ax_name, ri in zip(axis_names, candidate_idxs):
                rec[ax_name] = ffilled[ri][ci]
            rec[value_name] = row.iloc[ci]
            records.append(rec)

    return pd.DataFrame(records).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 集計行・集計列の除去
# ---------------------------------------------------------------------------

# 全角スペース・ゼロ幅文字など ASCII strip で取れない空白類の正規化
_WS_RE = _re.compile(r"[\s　\xa0​‌‍﻿]+")

# 末尾の括弧注記パターン: 「（参考）」「[除く海外]」「(注1)」など
# 1つ以上の括弧グループ（各種括弧の対）が末尾に続く場合にマッチ
_TRAILING_BRACKET_RE = _re.compile(
    r"[\s　]*(?:[（(【〔「\[]\s*[^)）】〕」\]]*\s*[)）】〕」\]])+\s*$"
)


def _normalize_label(s: str) -> str:
    """ラベル比較用に Unicode 空白類・制御文字を除去して正規化する。"""
    return _WS_RE.sub("", s).strip()


def _is_agg_label(s: str) -> bool:
    """値が集計ラベルか判定する。完全一致 or キーワードで終わる場合に True。

    全角スペース・ゼロ幅スペース等の不可視文字を除去してから比較する。
    末尾に括弧注記（「（参考）」「[除く○○]」等）がある場合はそれを除いて再判定する。
    例: "合計（参考）" → "合計" として判定 → True
        "計画（2023年度）" → "計画" → マッチしない → False
    """
    s = _normalize_label(str(s))
    if not s:
        return False
    sl = s.lower()
    for kw in AGG_KEYWORDS:
        kw_l = kw.lower()
        if sl == kw_l or sl.endswith(kw_l):
            return True
    # 末尾の括弧注記を除いて再判定
    stripped = _TRAILING_BRACKET_RE.sub("", s).strip()
    if stripped and stripped != s:
        sl2 = _normalize_label(stripped).lower()
        for kw in AGG_KEYWORDS:
            kw_l = kw.lower()
            if sl2 == kw_l or sl2.endswith(kw_l):
                return True
    return False


def _detect_covariates(df: Any, cols: List[str]) -> Dict[str, set]:
    """一方が他方を一意に決定する列ペア（コード↔ラベルなど）を検出する。

    例: 都道府県コード='01' ↔ 都道府県名='北海道' のように完全に対応する列。
    戻り値: {col: {共変する列名のセット}} 形式の辞書。
    """
    import pandas as pd

    covar: Dict[str, set] = {c: set() for c in cols}
    valid = [c for c in cols if c in df.columns]
    if len(df) < 2:
        return covar
    for i, c1 in enumerate(valid):
        for c2 in valid[i + 1 :]:
            try:
                # c1 が c2 を一意に決定するか、または c2 が c1 を一意に決定するか
                g1 = df.groupby(c1, dropna=False)[c2].nunique()
                g2 = df.groupby(c2, dropna=False)[c1].nunique()
                if (g1 <= 1).all() or (g2 <= 1).all():
                    covar[c1].add(c2)
                    covar[c2].add(c1)
            except Exception:
                pass
    return covar


def _is_grouping_col(series: Any, df_len: int) -> bool:
    """列がグルーピング変数（カテゴリ・コード列）かを判定する。

    集計の冗長性チェックで「コンテキスト」に使える列の条件:
      - ユニーク値の割合が 50% 未満（同じ値が複数行にまたがる）
    数値データ列（売上額、給与額など）はほぼ全行が異なる値を持つため除外される。
    全 null 列は割合が 0 なのでコンテキストに残るが、NaN 一致のため実害はない。
    """
    n_unique = series.nunique(dropna=True)
    return n_unique / max(1, df_len) < 0.5


def _is_redundant_agg_row(
    df: Any,
    idx: Any,
    col: str,
    ctx_cols: List[str],
    covar: Dict[str, set],
) -> bool:
    """集計値を持つ行が冗長かどうかを判定する。

    「冗長」とは、その集計値と同じコンテキスト（他のラベル列の値が一致する行）に
    非集計の個別データが存在する場合を指す。
    例: 都道府県名='全国計' かつ 年次=2019 → 同年に個別都道府県の行が存在 → 冗長
        都道府県名='全国計' かつ 年次=2015 → 2015年に個別都道府県の行が存在しない → 冗長でない

    ctx_cols: 数値データ列を除いた object 列（コード・ラベル列のみ）。
    covar:    ctx_cols 内の共変ペア辞書。col の共変列はコンテキストから除外する。
    """
    import pandas as pd

    context_cols = [c for c in ctx_cols if c != col and c not in covar.get(col, set())]

    if not context_cols:
        """
        コンテキスト列がない場合、対象列に非集計値が存在するなら除去できる。
        非集計値がなければ全行が集計のみのため、削除するとデータが失われる。
        """
        return any(
            not _is_agg_label(str(v))
            for v in df[col].dropna()
            if not (isinstance(v, float) and pd.isna(v))
        )

    # 同じコンテキスト値を持つ他の行を検索
    def _isnull(v: Any) -> bool:
        if v is None:
            return True
        try:
            return bool(pd.isna(v))
        except (TypeError, ValueError):
            return False

    mask = pd.Series(True, index=df.index)
    for cc in context_cols:
        cv = df.at[idx, cc]
        if _isnull(cv):
            mask &= df[cc].isna()
        else:
            mask &= df[cc] == cv
    mask.at[idx] = False  # 自行を除外

    # コンテキスト一致行に非集計値を持つ行が存在するか
    matching_vals = df.loc[mask, col]
    if not matching_vals.empty:
        return any(not _is_agg_label(str(v)) for v in matching_vals if not _isnull(v))

    # コンテキスト一致行が見つからない場合。
    # 2通りの状況がある:
    #
    # A) コンテキスト列に None/pd.NA が含まれる → fill_grouping_cols が未適用・不完全な可能性。
    #    集計行のグルーピング列が空白のまま残っているため、一致行が見つからなかった。
    #    → 列全体に非集計値があれば冗長と判定（保守的に除去）。
    #
    # B) コンテキスト列がすべて非 null（完全なコンテキスト）→ このコンテキストに
    #    サブレベルのデータが存在しない（例: 2015年は全国計のみで都道府県別データなし）。
    #    他の年のデータ（北海道, 青森など）を誤って「sibling」と見なすべきではない。
    #    → 冗長ではない = 除去しない。
    has_null_ctx = any(_isnull(df.at[idx, cc]) for cc in context_cols)
    if not has_null_ctx:
        return False
    return any(not _is_agg_label(str(v)) for v in df[col] if not _isnull(v))


def _to_jsonable(v: Any) -> Any:
    """numpy スカラー（int64/float64 等）を JSON 変換可能な Python 組み込み型に変換する。"""
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return v
    return v


def _agg_column_group_key(col_name: str) -> str:
    """列名から、集計列と内訳候補列を対応付けるためのグループキーを求める。

    単位表記（末尾の "[百万円]" 等）を除去した上で、アンダースコア区切りの
    最初のセグメント（多段ヘッダー統合時の上位グループ名）を返す。
    アンダースコアが無い単独名の列は空文字列（"接頭辞なしグループ"）を返す。
    例: "事業所数_合計" → "事業所数"、"現金給与総額 [百万円]" → ""
    """
    s = _re.sub(r"\s*\[[^\]]*\]\s*$", "", str(col_name)).strip()
    return s.split("_")[0] if "_" in s else ""


def _is_column_sum_verified(
    df: Any,
    agg_col: str,
    sibling_cols: List[str],
    match_ratio: float = 0.8,  # Wide_to_long検出等と揃えた一致率閾値
) -> bool:
    """agg_col の値が sibling_cols の行ごとの合計とみなせるかを検証する。

    合計元となりうる列（同じグループの内訳列）が2列未満の場合は検証不能と
    みなし False を返す（行の集計判定 _is_redundant_agg_row と同じ考え方で、
    内訳データが存在しない場合は集計列と誤認しない）。
    """
    import numbers

    import pandas as pd

    if len(sibling_cols) < 2:
        return False

    def _num(v: Any) -> Optional[float]:
        if isinstance(v, bool):
            return None
        # df.at[] は numpy.int64/float64 等をそのまま返し Python の int/float に
        # 自動変換されないため（Series の for ループでは自動変換される点と異なる）、
        # numbers.Number で判定する。
        if isinstance(v, numbers.Number) and not (isinstance(v, float) and pd.isna(v)):
            return float(v)
        return None

    checked = 0
    matched = 0
    for idx in df.index:
        agg_val = _num(df.at[idx, agg_col])
        if agg_val is None:
            continue
        sib_vals = [_num(df.at[idx, sc]) for sc in sibling_cols]
        if any(v is None for v in sib_vals):
            continue
        checked += 1
        tolerance = max(1e-6, abs(agg_val) * 0.01)
        if abs(agg_val - sum(sib_vals)) <= tolerance:
            matched += 1

    if checked == 0:
        return False
    return matched / checked >= match_ratio


def remove_aggregates(
    df: Any,  # pd.DataFrame
    protected_indices: Optional[Any] = None,
) -> Tuple[
    Any,
    List[Dict[str, Any]],
    List[str],
    List[int],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    集計行・集計列を除去した DataFrame と除去情報を返す。

    集計列: 列名がキーワードに一致し、かつ他の列（同じ多段ヘッダー由来グループの
            内訳列）の行ごとの合計と実際に一致することを検証できた場合のみ除去する
            （行の集計判定と同じ「冗長性を検証できた場合のみ除去」という考え方）。
            合計元となりうる内訳列が存在しない場合（例: 内訳の無い単独の指標）は、
            列名がキーワードに一致していても除去しない。
    集計行: dtype==object の列（ラベル列）に集計ラベルを持ち、かつその集計が「冗長」
            （同じ文脈で個別データが存在する）場合のみ除去する。
            個別データが存在しない場合（例: 全行が同一の集計ラベルのみの区分）は除去しない。

    protected_indices: 冗長と判定されても除去してはいけない行の df 上のインデックス
            集合（例:「うち」書き内訳分離（B-15）で内訳テーブルの親として参照
            済みの合計行。この行を削除すると内訳テーブルが参照する親が本体側から
            消え、対応関係が追えなくなるため保護する）。

    Returns:
        cleaned_df               — 集計行・集計列を除去した DataFrame（index リセット済み）
        removed_rows_info        — 除去した各行のラベル列値 [{col: val, ...}, ...]
        removed_col_names        — 除去した列名リスト
        removed_row_indices      — 除去した行の元 DataFrame 上の整数インデックスリスト
        agg_removed_row_metadata — 除去した集計行の監査用メタデータ。除去行 × 数値列ごとに
                                    1件、次の形式:
                                    {"key": トリガー列名,
                                     "context": {ラベル列名: 値, ...}（trigger 列自身を含む）,
                                     "sum_column": 数値列名, "reported_value": 除去された数値}
                                    context は trigger 列自身の値を含む全ラベル列値なので、
                                    集計行を単独で説明できる。個別行から reported_value を
                                    再現検証する際は、context から key の列だけを除いた
                                    条件で最終テーブルを絞り込む。
        agg_removed_col_metadata — 除去した集計列（列名自体がキーワード一致）の監査用
                                    メタデータ。除去列 × 元の行ごとに1件、次の形式:
                                    {"removed_column": 除去した列名,
                                     "context": {ラベル列名: 値, ...},
                                     "reported_value": 除去された値}
                                    列ごと削除されるとその列の値が完全に失われるため、
                                    後から参照できるよう全行分を記録する。
    """
    import pandas as pd

    protected_set = set(protected_indices) if protected_indices else set()

    # ── ラベル列の特定（文字列型の列）────────────────────────────
    """
    pandas 2.x では文字列列は dtype=object、pandas 3.x では dtype=StringDtype になる。
    'X'（秘匿値）などの混入で文字列型になっているが実質数値の列を除外する。
    int/float オブジェクトが過半数の列はデータ列として label_cols から除く。
    pd.NA（pandas 3.x nullable NA）も null として扱う。
    """

    def _is_text_dtype(series: Any) -> bool:
        """数値・bool・日時以外の列を文字列列と見なす（pandas 2.x/3.x 全バージョン対応）。
        pandas 2.x: dtype=object, pandas 3.0+: dtype=string / str など表記が変わるため
        deny-list 方式（数値・bool・日時を除外）で判定する。
        """
        import pandas.api.types as _pat

        return not (
            _pat.is_numeric_dtype(series)
            or _pat.is_bool_dtype(series)
            or _pat.is_datetime64_any_dtype(series)
        )

    def _is_null_scalar(v: Any) -> bool:
        """None / np.nan / pd.NA など全ての null を安全に判定する。"""
        if v is None:
            return True
        try:
            r = pd.isna(v)
            return bool(r)
        except (TypeError, ValueError):
            return False

    def _is_numeric_values_col(series: Any) -> bool:
        non_null = []
        n_num = 0
        for v in series:
            if _is_null_scalar(v):
                continue
            if isinstance(v, (int, float)):
                non_null.append(v)
                n_num += 1
            else:
                # 秘匿マーカー（'X', '***' 等）はnullとして扱い、数値列判定を妨げない
                if str(v).strip().lower() in _STAT_NA_MARKERS:
                    continue
                non_null.append(v)
        if not non_null:
            return False
        return n_num / len(non_null) >= 0.5

    label_cols: List[str] = [
        col
        for col in df.columns
        if _is_text_dtype(df[col]) and not _is_numeric_values_col(df[col])
    ]

    # ── 数値列 — ラベル列ではない列 ──────────────────────────────
    """
    メタデータの context には文字列型のラベル列を使う。

    数値列側にもユニーク率による絞り込みを追加すると、値の重複が多いだけの
    正当な集計対象列まで誤って除外してしまうことを実データ検証で確認したため、
    数値列は絞り込まずすべて集計対象候補として扱う。
    """
    all_value_cols: List[str] = [col for col in df.columns if col not in label_cols]

    # ── 集計列の検出（列名キーワード一致 ＋ 内訳列の合計として検証）──
    """
    列名がキーワードに一致するだけの候補（agg_col_candidates）のうち、
    同じ多段ヘッダー由来グループ（_agg_column_group_key が一致）に属する
    他の列が2列以上存在し、かつ行ごとの値がそれらの合計と一致する場合の
    みを実際の集計列とみなす。これにより、「現金給与総額」のように内訳列
    が存在しない単独指標が、列名だけを理由に誤って除去されることを防ぐ。
    """
    agg_col_candidates = [col for col in all_value_cols if _is_agg_label(str(col))]
    removed_cols: List[str] = []
    for cand in agg_col_candidates:
        group_key = _agg_column_group_key(cand)
        siblings = [
            c
            for c in all_value_cols
            if c != cand
            and c not in agg_col_candidates
            and _agg_column_group_key(c) == group_key
        ]
        if _is_column_sum_verified(df, cand, siblings):
            removed_cols.append(cand)

    value_cols: List[str] = [col for col in all_value_cols if col not in removed_cols]

    n_rows = len(df)

    # ── 冗長性チェック用コンテキスト列（カテゴリ・コード列のみ）────
    """
    数値データ列（売上額・給与額など）はほぼ全行が異なる値を持ち、
    コンテキストに含めると「完全一致行ゼロ」になり冗長性判定が誤る。
    ユニーク値の割合 < 50% の列（同じ値が繰り返し現れる列）のみ使用する。
    """
    ctx_cols: List[str] = [
        col for col in label_cols if _is_grouping_col(df[col], n_rows)
    ]

    # ── コード↔ラベルなど共変ペアを検出（ctx_cols ベース）──────────
    # ctx_cols のみを対象にすることで、高基数列による誤検出を防ぐ。
    covar = _detect_covariates(df, ctx_cols)

    # ── 集計行の検出（冗長性チェック付き）────────────────────────
    removed_row_indices: List[int] = []
    removed_rows_info: List[Dict[str, Any]] = []
    agg_removed_row_metadata: List[Dict[str, Any]] = []

    # 各ラベル列について「列全体に非集計値が存在するか」をキャッシュしておく
    col_has_nonag: Dict[str, bool] = {}
    for _lc in label_cols:
        col_has_nonag[_lc] = any(
            not _is_agg_label(str(v)) for v in df[_lc] if not _is_null_scalar(v)
        )

    # 完全一致する集計キーワードのセット（正規化・小文字）
    _exact_agg_set: frozenset = frozenset(kw.lower() for kw in AGG_KEYWORDS)

    for idx in df.index:
        if idx in protected_set:
            continue
        for col in label_cols:
            val = df.at[idx, col]
            if _is_null_scalar(val):
                continue
            val_norm = _normalize_label(str(val))
            if not val_norm:
                continue
            if not _is_agg_label(val_norm):
                continue
            # 集計ラベルを持つ列を発見。
            # その列に非集計値が全く存在しない場合は全行が集計のみ → 除去しない。
            # ただし continue で次の列を確認する（break では後続列の集計値を見逃す）。
            if not col_has_nonag.get(col, False):
                continue
            # 完全一致キーワード（「計」「合計」等の単独語）はコンテキストチェック不要で除去。
            # 部分一致（「一般計」「前年累計」等）は冗長性チェックを行う。
            val_lower = val_norm.lower()
            is_exact = val_lower in _exact_agg_set
            should_remove = False
            if is_exact:
                should_remove = True
            elif col not in ctx_cols:
                should_remove = True
            else:
                should_remove = _is_redundant_agg_row(df, idx, col, ctx_cols, covar)
            if should_remove:
                removed_row_indices.append(idx)
                row_info: Dict[str, Any] = {"__trigger_col__": col}
                for lc in label_cols:
                    v = df.at[idx, lc]
                    if v is not None and not (isinstance(v, float) and pd.isna(v)):
                        row_info[lc] = v
                removed_rows_info.append(row_info)

                """
                監査用メタデータ: この集計行が持っていた数値列ごとに1件記録する。

                context には trigger 列（key で示す列）自身の値も含め、全ラベル列値を
                記録する。これにより:

                  - context 単独で集計行を一意に説明できる（値を別フィールドに
                    分離する必要がない）。
                  - 複数のグルーピング列を持つ表では、同じラベル値を持つ集計行が
                    他のグルーピング列の組み合わせごとに複数存在し得るが、
                    context を見ればどの組み合わせに対応する集計値かを特定できる。

                個別行から reported_value を再現検証する際は、context から key の
                列だけを除いた条件で最終テーブルを絞り込む（trigger 列と共変する
                列（コードと名称のように、一方の値がもう一方を一意に決定する列の
                ペア）は最初から context に含めない。含めると集計行自身が持つ値が
                そのまま残り、個別行はその値を持たないため絞り込みが0件になって
                しまう。_is_redundant_agg_row と同じ考え方）。
                """
                trigger_covar = covar.get(col, set())
                context = {
                    lc: _to_jsonable(row_info[lc])
                    for lc in label_cols
                    if lc not in trigger_covar and lc in row_info
                }
                for vc in value_cols:
                    rv = df.at[idx, vc]
                    if _is_null_scalar(rv):
                        continue
                    agg_removed_row_metadata.append(
                        {
                            "key": str(col),
                            "context": context,
                            "sum_column": str(vc),
                            "reported_value": _to_jsonable(rv),
                        }
                    )
                break

    """
    集計列（列名自体がキーワードに一致する列）の監査用メタデータ。

    列ごと削除されるとその列が持っていた値は最終出力テーブルのどこにも
    残らないため、元の行ごとに値とラベル列コンテキストを記録しておく。
    """
    agg_removed_col_metadata: List[Dict[str, Any]] = []
    for rc in removed_cols:
        for idx in df.index:
            rv = df.at[idx, rc]
            if _is_null_scalar(rv):
                continue
            context = {
                lc: _to_jsonable(df.at[idx, lc])
                for lc in label_cols
                if not _is_null_scalar(df.at[idx, lc])
            }
            agg_removed_col_metadata.append(
                {
                    "removed_column": str(rc),
                    "context": context,
                    "reported_value": _to_jsonable(rv),
                }
            )

    # ── 変更がなければ None を返してスキップを示す ────────────────
    if not removed_row_indices and not removed_cols:
        return df, [], [], [], [], []

    cleaned = df.drop(index=removed_row_indices, columns=removed_cols, errors="ignore")
    cleaned = cleaned.reset_index(drop=True)

    return (
        cleaned,
        removed_rows_info,
        removed_cols,
        removed_row_indices,
        agg_removed_row_metadata,
        agg_removed_col_metadata,
    )


# ---------------------------------------------------------------------------
# グルーピング列の前方補完（視覚結合セル対応）
# ---------------------------------------------------------------------------


def fill_grouping_cols(df: Any) -> Tuple[Any, List[str]]:  # noqa: C901
    """
    視覚的セル結合（XML 上は未マージ）によるグルーピング列の None を前方補完する。

    Excel では「セルを結合」表示していても実際には先頭セルにのみ値があり、
    後続セルが空のケースがある。この関数はそのようなグルーピング列を検出し
    ffill（前方補完）を適用する。

    対象列の条件（両方を満たす場合のみ適用）:
      1. object 型かつ、最初の非 None 値の後ろに None が存在する（内部 None）
      2. ffill 後のユニーク値比率 < 50%（カテゴリ列であることを確認）

    Returns:
      (filled_df, filled_col_names)
    """
    import pandas as pd

    if df is None or df.empty:
        return df, []

    df = df.copy()
    filled_cols: List[str] = []

    for col in df.columns:
        series = df[col]

        # pandas 2.x: object, pandas 3.0+: string / str など表記が変わるため deny-list で判定
        import pandas.api.types as _pat

        if (
            _pat.is_numeric_dtype(series)
            or _pat.is_bool_dtype(series)
            or _pat.is_datetime64_any_dtype(series)
        ):
            continue

        # None/NaN を持たない列はスキップ
        null_mask = series.isna()
        if not null_mask.any():
            continue

        # 最初の非 None 値の位置を取得
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is None:
            continue  # 列全体が None → スキップ

        # 最初の非 None 値より後ろに None があるか（内部 None の検出）
        pos = df.index.get_loc(first_valid_idx)
        if isinstance(pos, (slice, type(None))):
            pos = 0  # 重複インデックス対策（通常は発生しない）
        if not null_mask.iloc[pos + 1 :].any():
            continue  # 末尾以降の None のみ → スキップ

        # ffill を適用し、集計ラベルの過伝播を防ぐ。
        # 例: 列_2 で「計」→ None → None と並ぶ場合、ffill で後続 None が「計」に
        # なるが、これらの None は「サブ分類なし」を意味するので戻す。
        filled = series.ffill()
        original_null = series.isna()
        if original_null.any():
            for _null_idx in df.index[original_null]:
                _fv = filled.at[_null_idx]
                _fv_null = (
                    _fv is None
                    or (isinstance(_fv, float) and pd.isna(_fv))
                    or (pd.NA is not None and _fv is pd.NA)
                )
                if not _fv_null and _is_agg_label(str(_fv)):
                    try:
                        filled.at[_null_idx] = (
                            pd.NA
                        )  # StringDtype では None より pd.NA が安全
                    except Exception:
                        filled.at[_null_idx] = None

        # ffill 後のユニーク比率チェック
        n_unique = filled.nunique(dropna=True)
        if n_unique / max(1, len(df)) >= 0.5:
            continue  # ffill 後もユニーク率が高い → カテゴリ列でない

        df[col] = filled
        filled_cols.append(str(col))

    return df, filled_cols


# ---------------------------------------------------------------------------
# 「うち」書き識別と別テーブル分離
# ---------------------------------------------------------------------------
#
# 「うち男性」のように直前の合計値の内訳を示す行を検出し、削除ではなく
# 親子関係を保ったまま別テーブル（内訳テーブル）へ分離する。
#
# remove_aggregates（「計」「合計」等の冗長な集計行を削除する）は、この機能が
# 親として参照する「合計」行そのものを削除対象とみなす場合があるため、
# 必ず remove_aggregates より前に実行した上で、親として使われた行の
# インデックスを remove_aggregates に保護対象として渡し、本体テーブルから
# 消えないようにする（詳細は normalize_tables 内のコメントを参照）。


_UCHI_OPEN_BRACKETS = "（("
_UCHI_CLOSE_BRACKETS = "）)"
_UCHI_TRAILING_PUNCT = " 　、：:・"


def _is_uchi_label(s: str) -> Optional[str]:
    """セル値が UCHI_PREFIXES のいずれかで始まるかを判定する。

    括弧（半角/全角丸括弧）で囲まれた表記も2パターン許容する:
      - 括弧内に接頭辞＋子ラベルの両方（例: "（うち男性）" → "男性"）
      - 接頭辞のみ括弧で囲み、子ラベルが括弧の外（例: "（再掲）女性" → "女性"）
    接頭辞直後の区切り文字（読点・コロン・中黒等）も除去してから子ラベルを返す。

    一致すれば子ラベルを返す。一致しない場合、または接頭辞を除くと
    空文字になる場合は None。
    """
    stripped = str(s).strip()
    if not stripped:
        return None

    # 1. 括弧なしの単純接頭辞（例: "うち男性"、"再掲・女性"）
    for prefix in UCHI_PREFIXES:
        if stripped.startswith(prefix):
            child = stripped[len(prefix):].strip(_UCHI_TRAILING_PUNCT)
            if child:
                return child

    # 2. 先頭が丸括弧の場合、対応する閉じ括弧までを接頭辞候補として調べる
    if stripped[0] in _UCHI_OPEN_BRACKETS:
        close_idx = next(
            (i for i, ch in enumerate(stripped) if ch in _UCHI_CLOSE_BRACKETS), None
        )
        if close_idx is not None:
            inner = stripped[1:close_idx].strip()
            after = stripped[close_idx + 1 :].strip(_UCHI_TRAILING_PUNCT)
            for prefix in UCHI_PREFIXES:
                if inner == prefix and after:
                    return after  # 例: "（再掲）女性"
                if inner.startswith(prefix):
                    child = inner[len(prefix):].strip(_UCHI_TRAILING_PUNCT)
                    if child:
                        return child  # 例: "（うち男性）"
    return None


def detect_uchi_breakdown(df: Any) -> Optional[Dict[str, Any]]:
    """
    「うち」等の接頭辞を持つ内訳行を検出し、親子関係を解決する。

    対象列: 文字列型の列のうち、_is_uchi_label に一致する行を1件以上持つ列。
    複数列が該当する場合は一致件数が最多の列を採用する。

    各「うち」行の親は次の順で解決する（同一コンテキスト＝他のラベル列の
    値が完全一致する範囲内で先行行を探索）:
      1. 当該列の値が _is_agg_label に一致する直近の先行行
      2. 上記が見つからない場合、直近の非「うち」行
      3. どちらも見つからない、またはコンテキストが切り替わって
         見つけられない場合はその行を分離対象から除外する（安全側に倒す）

    Returns:
      検出された場合: {"label_col", "parent_col_name", "child_col_name",
                       "rows"（{idx: (parent_value, child_label)}）, "match_count",
                       "parent_indices"（親として参照された df 上のインデックス集合。
                       remove_aggregates に保護対象として渡し、内訳テーブルが
                       参照する親行が本体側から消えないようにするために使う）}
      検出されなかった場合: None
    """
    import pandas as pd
    import pandas.api.types as _pat

    if df is None or df.empty:
        return None

    def _is_text_dtype(series: Any) -> bool:
        return not (
            _pat.is_numeric_dtype(series)
            or _pat.is_bool_dtype(series)
            or _pat.is_datetime64_any_dtype(series)
        )

    def _is_null_scalar(v: Any) -> bool:
        if v is None:
            return True
        try:
            return bool(pd.isna(v))
        except (TypeError, ValueError):
            return False

    label_cols = [c for c in df.columns if _is_text_dtype(df[c])]
    if not label_cols:
        return None

    best_col = None
    best_matches: Dict[Any, str] = {}
    for col in label_cols:
        matches: Dict[Any, str] = {}
        for idx, v in df[col].items():
            if _is_null_scalar(v):
                continue
            child = _is_uchi_label(str(v))
            if child:
                matches[idx] = child
        if matches and len(matches) > len(best_matches):
            best_col = col
            best_matches = matches

    if best_col is None:
        return None

    other_label_cols = [c for c in label_cols if c != best_col]
    idx_list = list(df.index)
    pos_of = {idx: p for p, idx in enumerate(idx_list)}

    def _same_context(idx_a: Any, idx_b: Any) -> bool:
        for oc in other_label_cols:
            a, b = df.at[idx_a, oc], df.at[idx_b, oc]
            a_null, b_null = _is_null_scalar(a), _is_null_scalar(b)
            if a_null and b_null:
                continue
            if a_null != b_null or a != b:
                return False
        return True

    resolved: Dict[Any, Tuple[Any, str]] = {}
    parent_indices: set = set()
    for idx, child_label in best_matches.items():
        pos = pos_of[idx]
        parent_value = None
        parent_idx = None

        # 1. 直近の「合計」ラベル行を優先して探す
        for p in range(pos - 1, -1, -1):
            pidx = idx_list[p]
            if not _same_context(idx, pidx):
                break
            pval = df.at[pidx, best_col]
            if _is_null_scalar(pval) or pidx in best_matches:
                continue
            if _is_agg_label(str(pval)):
                parent_value = pval
                parent_idx = pidx
                break

        # 2. 見つからなければ直近の非「うち」行にフォールバック
        if parent_value is None:
            for p in range(pos - 1, -1, -1):
                pidx = idx_list[p]
                if not _same_context(idx, pidx):
                    break
                pval = df.at[pidx, best_col]
                if _is_null_scalar(pval) or pidx in best_matches:
                    continue
                parent_value = pval
                parent_idx = pidx
                break

        if parent_value is not None:
            resolved[idx] = (parent_value, child_label)
            if parent_idx is not None:
                parent_indices.add(parent_idx)

    if not resolved:
        return None

    return {
        "label_col": str(best_col),
        "parent_col_name": f"親{best_col}",
        "child_col_name": f"子{best_col}",
        "rows": resolved,
        "match_count": len(resolved),
        "parent_indices": parent_indices,
    }


def apply_uchi_split(df: Any, info: Dict[str, Any]) -> Tuple[Any, Any, set]:
    """detect_uchi_breakdown の検出結果を使って、うち内訳行をメインテーブルから
    除去し、親子関係を保った内訳テーブルを生成する。

    Returns: (main_df, breakdown_df, protected_indices)
      protected_indices — main_df（reset_index 後）上で、内訳テーブルが親として
      参照している行の新インデックス集合。remove_aggregates にそのまま渡すと、
      これらの行が「冗長な合計行」として誤って削除されるのを防げる。
    """
    import pandas as pd

    label_col = info["label_col"]
    parent_col_name = info["parent_col_name"]
    child_col_name = info["child_col_name"]
    rows: Dict[Any, Tuple[Any, str]] = info["rows"]
    parent_indices = info.get("parent_indices", set())

    dropped = set(rows.keys())
    remaining_index_order = [i for i in df.index if i not in dropped]
    new_pos = {old: new for new, old in enumerate(remaining_index_order)}
    protected_indices = {new_pos[p] for p in parent_indices if p in new_pos}

    main_df = df.drop(index=list(dropped)).reset_index(drop=True)

    other_cols = [c for c in df.columns if c != label_col]
    breakdown_rows = []
    for idx, (parent_value, child_label) in rows.items():
        row = df.loc[idx]
        new_row = {parent_col_name: parent_value, child_col_name: child_label}
        for c in other_cols:
            new_row[c] = row[c]
        breakdown_rows.append(new_row)

    breakdown_df = pd.DataFrame(breakdown_rows)

    # 列順: label_col があった位置に親/子列を配置し、他の列は元の順序を維持
    cols = list(df.columns)
    label_pos = cols.index(label_col)
    ordered_cols = (
        cols[:label_pos] + [parent_col_name, child_col_name] + cols[label_pos + 1 :]
    )
    breakdown_df = breakdown_df[ordered_cols].reset_index(drop=True)

    return main_df, breakdown_df, protected_indices


# ---------------------------------------------------------------------------
# クロス集計形式（横持ち時系列）の検出と縦持ち変換
# ---------------------------------------------------------------------------

# タイトル・ファイル名から年を抽出する正規表現
_YEAR_CTX_RE = _re.compile(r"(19|20)\d{2}")


def _classify_col_time(col_name: str) -> Optional[str]:
    """列名が時系列パターンにマッチするか判定し、種別文字列を返す。"""
    s = str(col_name).strip()
    for pattern, kind in _TIME_PATTERNS:
        if pattern.match(s):
            return kind
    return None


def _extract_year_context(
    title: Optional[str], filename: Optional[str]
) -> Optional[int]:
    """タイトルまたはファイル名から西暦年（1900〜2099）を抽出する。"""
    for source in [title, filename]:
        if source:
            m = _YEAR_CTX_RE.search(str(source))
            if m:
                return int(m.group())
    return None


def detect_cross_table(
    df: Any,
    title: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    DataFrame がクロス集計形式（月・四半期・年度等を列に持つ横持ち）かを検出する。

    判定条件（両方必須）:
      - 列名の 80% 以上が時系列パターン（月・四半期・年度等）にマッチする
      - 時系列列が 2 列以上ある

    月のみ列（1月〜12月）の場合は title / filename から年を補完する。

    Returns:
      検出された場合は変換情報 dict、検出されなかった場合は None
    """
    if df is None or df.empty or len(df.columns) < 3:
        return None

    col_names = [str(c) for c in df.columns]
    time_types = [_classify_col_time(c) for c in col_names]

    # 最初の時系列列が現れる位置を特定
    first_time_idx = next((i for i, t in enumerate(time_types) if t is not None), None)
    if first_time_idx is None:
        return None

    # 先頭ラベル列を除いた範囲で時系列列の割合を判定（80% 以上）
    candidate_types = time_types[first_time_idx:]
    n_time = sum(1 for t in candidate_types if t is not None)
    if n_time < 2 or n_time / max(1, len(candidate_types)) < 0.8:
        return None

    label_cols = [col_names[i] for i, t in enumerate(time_types) if t is None]
    time_cols = [col_names[i] for i, t in enumerate(time_types) if t is not None]
    time_kind = next(t for t in time_types if t is not None)

    # 月のみ列の場合、title/filename から年を補完
    year_context: Optional[int] = None
    if time_kind == "month":
        year_context = _extract_year_context(title, filename)

    var_name = _VAR_NAME_MAP.get(time_kind, _VAR_NAME_FALLBACK)

    # 値列名の推定（タイトルキーワード優先、なければ "値"）
    value_name = "値"
    if title:
        for kw, name in _VALUE_KEYWORDS.items():
            if kw in title:
                value_name = name
                break

    return {
        "label_cols": label_cols,
        "time_cols": time_cols,
        "time_kind": time_kind,
        "var_name": var_name,
        "value_name": value_name,
        "year_context": year_context,
    }


def stack_cross_table(df: Any, stack_info: Dict[str, Any]) -> Any:
    """クロス集計形式（横持ち）を縦持ち（long format）に変換する。

    月のみ列で year_context が設定されている場合、年列を先頭ラベルの直後に挿入する。
    """
    label_cols = stack_info["label_cols"]
    time_cols = stack_info["time_cols"]
    var_name = stack_info["var_name"]
    value_name = stack_info["value_name"]
    year_context = stack_info.get("year_context")
    time_kind = stack_info["time_kind"]

    melted = df.melt(
        id_vars=label_cols,
        value_vars=time_cols,
        var_name=var_name,
        value_name=value_name,
    )

    # 月のみの場合、年列をラベル列の直後に挿入
    if year_context is not None and time_kind == "month":
        melted.insert(len(label_cols), "年", year_context)

    return melted.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Wide_to_long（多次元の同質軸×複数指標の複合列名）検出と変換
# ---------------------------------------------------------------------------
#
# detect_cross_table は「列名が軸トークンそのもの」（例: 2023年）である
# 横持ち表しか検出できない。ここでは「支店,2023年売上,2023年原価」のように
# 軸（時系列に限らない同質な繰り返し軸。支店・性別・年代等も対象）と指標名が
# 1つの列名に合成された複合表記の横持ち表を検出し、軸のみを縦持ちに変換する
# （指標は列として維持する）。
#
# 3段階（Tier）で分類器を切り替える。いずれも「列名→(axis_token, indicator,
# kind)」への分類だけが異なり、その後のグリッド検証（_build_grid_info）は
# 全Tier共通（語彙非依存）。
#   Tier1: 時系列語彙（_classify_col_time、閉じた正規表現語彙）による分類。
#          曖昧さがなく最も信頼できるため常に最優先・決定論的に試す。
#   Tier2: 区切り文字（AXIS_DELIMITERS）による分類。時系列以外の軸も、
#          "東京支社_売上" のように区切り文字があれば意味理解なしに
#          決定論的に分割できる（merge_header_rows がこの "_" 連結規約で
#          複合列名を生成しているため実利が大きい）。
#   Tier3: 区切り文字のない完全連結列名（例: "東京支店売上"）向け。頻度ベースで
#          （語彙もLLMも使わず）2列以上にまたがる反復部分文字列を発見できた
#          場合のみ、LLM にその候補群が本当に均質な1つのカテゴリ軸として
#          意味を持つかを確認・命名させる（LLMは分割点を発見しない）。
# Tier2/3 は語彙による裏付けがない分、Tier1より厳しい閾値でグリッドを検証する
# （_AXIS_SPLIT_* 定数、詳細は _build_grid_info を参照）。

_WIDE_TO_LONG_MATCH_RATIO = 0.8  # Tier1: 軸+指標に分解できる列が占めるべき最低割合
_WIDE_TO_LONG_COMPLETENESS = 0.6  # Tier1: 想定グリッド（軸数×指標数）に対する実列数の最低割合

_AXIS_SPLIT_MATCH_RATIO = 0.9  # Tier2/3: 語彙の裏付けがない分、一致率で厳しく補う
_AXIS_SPLIT_COMPLETENESS = 0.75  # Tier2/3: 同上、グリッド完全性も厳しく
_AXIS_MIN_TABLE_COLS = 6  # Tier2/3 を試す最低列数（Tier1の4列よりも厳しくし偶然の一致を抑制）
_AXIS_MIN_DOMINANT_CARDINALITY = 3  # Tier2/3: 軸・指標の少なくとも一方は3種類以上を要求
_AXIS_GENERIC_VAR_NAME = "区分"  # Tier2で軸の意味的名称が推定できない場合のフォールバック
_CONCAT_MIN_TOKEN_LEN = 2  # Tier3: 反復候補として採用する最小文字数（1文字は偶然一致が多すぎる）


def _split_time_indicator(col_name: str) -> Optional[Tuple[str, str, str]]:
    """列名を (time_token, indicator_name, time_kind) に分解する（Tier1）。

    列名のあらゆる分割点でプレフィックス／サフィックスが _classify_col_time
    （完全一致の時系列パターン判定）にマッチするかを試す。複数の分割点が
    マッチする場合は最も長く一致したトークンを採用する
    （例: "2024Q1売上" で "2024"(year_num) ではなく "2024Q1"(fiscal_quarter) を優先）。

    分解できない場合（時系列を含まない通常のラベル列名等）は None。
    """
    s = str(col_name).strip()
    candidates: List[Tuple[str, str, str]] = []
    for cut in range(1, len(s)):
        prefix, suffix = s[:cut], s[cut:]
        kind = _classify_col_time(prefix)
        if kind is not None:
            indicator = suffix.lstrip("_- 　")
            if indicator:
                candidates.append((prefix, indicator, kind))
        kind = _classify_col_time(suffix)
        if kind is not None:
            indicator = prefix.rstrip("_- 　")
            if indicator:
                candidates.append((suffix, indicator, kind))
    if not candidates:
        return None
    return max(candidates, key=lambda c: len(c[0]))


def _classify_columns_by_delimiter(
    candidate_cols: List[str],
) -> Optional[Dict[str, Tuple[str, str, str]]]:
    """列名を区切り文字で (axis_token, indicator, kind) に分解する（Tier2）。

    AXIS_DELIMITERS を優先順に試し、各区切り文字について「軸_指標」
    「指標_軸」の両方向を試す。区切り文字を複数含む列（分割位置が曖昧）は
    除外する。テーブル全体で最も一致率の高い (区切り文字, 向き) の組を
    採用する（列ごとに向きが混在しないようにするため）。一致率が同じ場合は
    AXIS_DELIMITERS の優先順、および「軸_指標」の向きを優先する
    （merge_header_rows の連結規約に合わせる）。

    分解できない場合は None。
    """
    if not candidate_cols:
        return None

    best_parsed: Optional[Dict[str, Tuple[str, str, str]]] = None
    best_ratio = 0.0

    for delim in AXIS_DELIMITERS:
        splittable = [c for c in candidate_cols if c.count(delim) == 1]
        if len(splittable) < 2:
            continue
        kind = f"delim:{delim}"
        for axis_first in (True, False):
            parsed: Dict[str, Tuple[str, str, str]] = {}
            for c in splittable:
                left, right = c.split(delim, 1)
                left, right = left.strip(), right.strip()
                if not left or not right:
                    continue
                parsed[c] = (left, right, kind) if axis_first else (right, left, kind)
            if not parsed:
                continue
            ratio = len(parsed) / len(candidate_cols)
            if ratio > best_ratio:
                best_ratio = ratio
                best_parsed = parsed

    return best_parsed


def _cluster_by_recurring_segment(
    candidate_cols: List[str], anchor: str
) -> Optional[Dict[str, Tuple[str, str]]]:
    """anchor="prefix" なら列名の先頭側、"suffix" なら末尾側で、2列以上に
    またがって繰り返し出現する部分文字列（語彙照合なしの純粋な頻度シグナル）を
    軸トークン候補として (axis_token, indicator) に分解する（Tier3の一部）。

    1回の呼び出し内では全列が同じ側（prefix または suffix）で分解される
    （列ごとに向きが混在しない）。見つからなければ None。
    """
    s_list = [str(c).strip() for c in candidate_cols]

    segment_counts: Dict[str, int] = {}
    for s in s_list:
        seen = set()
        for cut in range(_CONCAT_MIN_TOKEN_LEN, len(s) - _CONCAT_MIN_TOKEN_LEN + 1):
            seen.add(s[:cut] if anchor == "prefix" else s[cut:])
        for seg in seen:
            segment_counts[seg] = segment_counts.get(seg, 0) + 1

    recurring = {seg for seg, n in segment_counts.items() if n >= 2}
    if not recurring:
        return None

    parsed: Dict[str, Tuple[str, str]] = {}
    for c, s in zip(candidate_cols, s_list):
        best_seg: Optional[str] = None
        for seg in recurring:
            matches = s.startswith(seg) if anchor == "prefix" else s.endswith(seg)
            if matches and (best_seg is None or len(seg) > len(best_seg)):
                best_seg = seg
        if best_seg is None:
            continue
        if anchor == "prefix":
            remainder = s[len(best_seg):].strip("_- 　")
        else:
            remainder = s[: len(s) - len(best_seg)].strip("_- 　")
        if remainder:
            parsed[c] = (best_seg, remainder)

    return parsed if len(parsed) >= 2 else None


def _find_concatenated_axis_candidates(
    candidate_cols: List[str],
) -> Optional[Dict[str, Tuple[str, str, str]]]:
    """区切り文字のない完全連結列名から、2列以上にまたがって繰り返し出現する
    接頭辞・接尾辞を頻度ベースで検出する（Tier3の決定論的事前フィルタ。語彙も
    LLMも使わない純粋な構造シグナル）。見つからなければ None を返し、
    LLM呼び出しをスキップする（コスト制御）。

    prefix方向・suffix方向のどちらか一貫した側で、より多くの列を説明できる
    方を採用する。
    """
    if len(candidate_cols) < 2:
        return None

    prefix_parsed = _cluster_by_recurring_segment(candidate_cols, "prefix")
    suffix_parsed = _cluster_by_recurring_segment(candidate_cols, "suffix")

    best: Optional[Dict[str, Tuple[str, str]]] = None
    for cand in (prefix_parsed, suffix_parsed):
        if cand and (best is None or len(cand) > len(best)):
            best = cand
    if not best:
        return None

    return {c: (tok, ind, "concat") for c, (tok, ind) in best.items()}


def _build_grid_info(
    parsed: Dict[str, Tuple[str, str, str]],
    candidate_cols: List[str],
    col_names: List[str],
    match_ratio_threshold: float,
    completeness_threshold: float,
    min_dominant_cardinality: int = 2,
    reject_pure_digit_side: bool = False,
) -> Optional[Dict[str, Any]]:
    """分類済みの parsed（列名→(axis_token, indicator, kind)）からグリッド
    （軸数×指標数）としての妥当性を検証し、Wide_to_long の検出結果を組み立てる。
    Tier1〜3のいずれの分類器から呼ばれても共通のロジック（語彙非依存）。

    reject_pure_digit_side: True の場合、軸トークン・指標名のいずれかが
    「数字のみの文字列」だけで構成されるグリッドを却下する
    （_dedup_columns/_make_unique_columns が列名衝突解消で付与する "_1","_2"
    のような連番接尾辞を軸と誤認しないためのガード。時系列の年数字
    （例: "2020","2021"）は正当な軸トークンのため Tier1 では False にする）。
    """
    if not candidate_cols:
        return None
    if len(parsed) / len(candidate_cols) < match_ratio_threshold:
        return None

    axis_tokens = sorted({t for t, _, _ in parsed.values()})
    indicators: List[str] = []
    for _, ind, _ in parsed.values():
        if ind not in indicators:
            indicators.append(ind)

    if len(axis_tokens) < 2 or len(indicators) < 2:
        return None
    if max(len(axis_tokens), len(indicators)) < min_dominant_cardinality:
        return None
    if reject_pure_digit_side:
        if all(t.isdigit() for t in axis_tokens) or all(i.isdigit() for i in indicators):
            return None

    expected = len(axis_tokens) * len(indicators)
    if len(parsed) / expected < completeness_threshold:
        return None

    label_cols = [c for c in col_names if c not in parsed]
    axis_kind = next(iter(parsed.values()))[2]

    return {
        "label_cols": label_cols,
        "axis_kind": axis_kind,
        "axis_tokens": axis_tokens,
        "indicators": indicators,
        "parsed_cols": parsed,
    }


def detect_wide_to_long(
    df: Any,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    client: Any = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    DataFrame が「軸×複数指標」の複合列名を持つ横持ち表かを検出する
    （軸は時系列に限らない。支店・性別・年代等の同質な繰り返し軸も対象）。

    Tier1（時系列語彙）→Tier2（区切り文字）→Tier3（LLM確認、client指定時のみ）
    の順に試し、最初にグリッド検証（_build_grid_info）を通過した結果を返す。

    共通の判定条件（いずれのTierも必須）:
      - 列名が軸トークン単体（時系列なら _classify_col_time が非None）の列は
        対象外とする（detect_cross_table の担当領域と重複させない）
      - 検出された軸トークンが2種類以上、かつ指標名が2種類以上
        （指標が1種類のみなら detect_cross_table の担当。この条件により
        両検出は互いに排他になる）
      - 一致率・グリッド完全性が閾値以上（Tier2/3はTier1より厳しい閾値）

    Returns:
      検出された場合: {"label_cols", "axis_var_name", "axis_kind",
                       "axis_tokens", "indicators", "parsed_cols"}
      検出されなかった場合: None
    """
    if df is None or df.empty or len(df.columns) < 4:
        return None

    col_names = [str(c) for c in df.columns]
    candidate_cols = [c for c in col_names if _classify_col_time(c) is None]
    if not candidate_cols:
        return None

    # ── Tier1: 時系列語彙 ────────────────────────────────────────
    parsed: Dict[str, Tuple[str, str, str]] = {}
    for c in candidate_cols:
        result = _split_time_indicator(c)
        if result:
            parsed[c] = result
    grid = _build_grid_info(
        parsed, candidate_cols, col_names,
        _WIDE_TO_LONG_MATCH_RATIO, _WIDE_TO_LONG_COMPLETENESS,
    )
    if grid:
        grid["axis_var_name"] = _VAR_NAME_MAP.get(grid["axis_kind"], _VAR_NAME_FALLBACK)
        return grid

    if len(col_names) < _AXIS_MIN_TABLE_COLS:
        return None

    # ── Tier2: 区切り文字ベースの汎用軸分類（決定論的、LLM不使用） ──────
    delim_parsed = _classify_columns_by_delimiter(candidate_cols)
    if delim_parsed:
        grid = _build_grid_info(
            delim_parsed, candidate_cols, col_names,
            _AXIS_SPLIT_MATCH_RATIO, _AXIS_SPLIT_COMPLETENESS,
            _AXIS_MIN_DOMINANT_CARDINALITY, reject_pure_digit_side=True,
        )
        if grid:
            grid["axis_var_name"] = _AXIS_GENERIC_VAR_NAME
            return grid

    # ── Tier3: 区切り文字なし複合列名 + LLM確認（clientが渡された場合のみ）──
    if client is None:
        return None

    concat_parsed = _find_concatenated_axis_candidates(candidate_cols)
    if not concat_parsed:
        return None
    grid = _build_grid_info(
        concat_parsed, candidate_cols, col_names,
        _AXIS_SPLIT_MATCH_RATIO, _AXIS_SPLIT_COMPLETENESS,
        _AXIS_MIN_DOMINANT_CARDINALITY, reject_pure_digit_side=True,
    )
    if not grid:
        return None

    axis_result = detect_category_axis(
        grid["axis_tokens"], grid["indicators"], title, client, model
    )
    if not axis_result:
        return None
    grid["axis_var_name"] = axis_result["axis_name"]
    return grid


def stack_wide_to_long(df: Any, info: Dict[str, Any]) -> Any:
    """detect_wide_to_long の検出結果を使って横持ち→縦持ち変換する。

    軸トークンごとにサブフレームを作り（指標列は元の列出現順を維持し、
    該当する元列が存在しない組み合わせは NaN で埋めてグリッドの歯抜けを
    許容する）、縦に連結する。
    """
    import pandas as pd

    label_cols = info["label_cols"]
    axis_var_name = info["axis_var_name"]
    axis_tokens = info["axis_tokens"]
    indicators = info["indicators"]
    parsed_cols = info["parsed_cols"]

    frames = []
    for token in axis_tokens:
        sub = df[label_cols].copy()
        sub.insert(len(label_cols), axis_var_name, token)
        for ind in indicators:
            src_col = next(
                (c for c, (t, i, _k) in parsed_cols.items() if t == token and i == ind),
                None,
            )
            sub[ind] = df[src_col] if src_col is not None else None
        frames.append(sub)

    result = pd.concat(frames, ignore_index=True)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 単位混在の分離（指標マスタの生成）
# ---------------------------------------------------------------------------

# ラベル末尾の単位注記パターン: 「15歳以上人口(人)」「労働力率（％）」「指標[人]」など。
# 括弧種別（半角/全角丸括弧・角括弧）の表記揺れを許容する。
# 開き括弧と閉じ括弧の種類が一致しない組み合わせ（例: "指標(人]"）も緩く許容する
# （既存の _TRAILING_BRACKET_RE と同じ方針）。
_UNIT_SUFFIX_RE = _re.compile(
    r"^(?P<label>.+?)[\s　]*[（(\[]\s*(?P<unit>[^()（）\[\]]{1,15})\s*[)）\]]\s*$"
)

_UNIT_SPLIT_MATCH_RATIO = 0.6  # 単位付きセルが列内で占めるべき最低割合
_UNIT_SPLIT_MASTER_COL = "単位"


def detect_and_split_units(df: Any) -> Optional[Dict[str, Any]]:
    """
    「15歳以上人口(人)」のように指標名へ単位が埋め込まれた列を検出し、
    単位が異なる指標が同一テーブル内に混在している場合に指標マスタへ分離する。

    判定条件（対象列につき両方必須）:
      - 文字列セルのうち末尾に既知単位（UNIT_VOCAB）の括弧注記を持つものが
        列内の文字列セルの 60% 以上を占める
      - 抽出された単位が 2 種類以上存在する（単位が統一されている列は分離不要）

    複数列が条件を満たす場合は、単位付きセルの割合が最も高い列を採用する。

    Returns:
      検出された場合: {"label_col", "master_col", "cleaned_df", "master_df",
                       "mapping", "match_count"}
      検出されなかった場合: None
    """
    import pandas as pd

    if df is None or df.empty:
        return None

    best: Optional[Dict[str, Any]] = None
    best_ratio = 0.0

    for col in df.columns:
        if _classify_col_time(col) is not None:
            continue  # 時系列列（年・月等）は対象外

        series = df[col]
        non_null = [
            v for v in series if v is not None and not (isinstance(v, float) and pd.isna(v))
        ]
        str_vals = [str(v).strip() for v in non_null if isinstance(v, str)]
        if not str_vals or len(str_vals) / max(1, len(non_null)) < 0.8:
            continue  # 文字列主体でない列（数値列・時系列値列等）は対象外

        matches: Dict[str, Tuple[str, str]] = {}  # 元セル文字列 → (label, unit)
        matched_cell_count = 0  # 重複値を含む実セル数（比率計算用）
        for v in str_vals:
            m = _UNIT_SUFFIX_RE.match(v)
            if not m:
                continue
            label = m.group("label").strip()
            unit = m.group("unit").strip()
            if not label or unit.lower() not in UNIT_VOCAB:
                continue
            matches[v] = (label, unit)
            matched_cell_count += 1

        if not matches:
            continue

        """
        比率は実セル数（matched_cell_count）を分子に使う。
        matches は元セル文字列をキーとする辞書のため、指標列に同一ラベルが
        複数行（性別・年など他のグルーピング列との組み合わせ）で繰り返される
        典型的なケースでは重複が畳み込まれてしまい、len(matches) を使うと
        実際は全セル一致でも比率が大きく下がって誤検出漏れが起きる。
        """
        match_ratio = matched_cell_count / len(str_vals)
        distinct_units = {u for _, u in matches.values()}
        if match_ratio < _UNIT_SPLIT_MATCH_RATIO or len(distinct_units) < 2:
            continue  # 単位混在（2種類以上）でなければ分離不要

        if match_ratio > best_ratio:
            best_ratio = match_ratio
            best = {"col": col, "matches": matches, "matched_cell_count": matched_cell_count}

    if best is None:
        return None

    col = best["col"]
    matches = best["matches"]
    matched_cell_count = best["matched_cell_count"]

    mapping: Dict[str, str] = {}
    for label, unit in matches.values():
        mapping.setdefault(label, unit)  # 表記ゆれ等で単位が割れた場合は初出を優先

    cleaned_df = df.copy()
    cleaned_df[col] = cleaned_df[col].apply(
        lambda v: matches[str(v).strip()][0]
        if isinstance(v, str) and str(v).strip() in matches
        else v
    )

    master_df = pd.DataFrame(
        {str(col): list(mapping.keys()), _UNIT_SPLIT_MASTER_COL: list(mapping.values())}
    )

    return {
        "label_col": str(col),
        "master_col": _UNIT_SPLIT_MASTER_COL,
        "cleaned_df": cleaned_df,
        "master_df": master_df,
        "mapping": mapping,
        "match_count": matched_cell_count,
    }


# ---------------------------------------------------------------------------
# 統括関数（LLM処理＋決定論的処理を正しい順序で適用）
# ---------------------------------------------------------------------------


def _apply_agg_and_unit_split(
    t: Any, df: Any, protected_indices: Optional[Any] = None
) -> Any:
    """集計行・集計列の除去、単位混在の分離を1テーブルに適用し、結果の df を返す。

    normalize_tables() のメインループと、うち分離で新規生成された内訳テーブルの
    両方から呼ばれる共通処理（DetectedTable の該当フィールドを in-place 更新する）。

    protected_indices: 「うち」書き分離（B-15）が内訳テーブルの親として参照済みの
    合計行インデックス（apply_uchi_split が返すもの）。remove_aggregates に
    そのまま渡し、冗長行として誤って削除されないようにする。
    """
    (
        cleaned_df,
        agg_rows,
        agg_cols,
        agg_row_positions,
        agg_row_meta,
        agg_col_meta,
    ) = remove_aggregates(df, protected_indices=protected_indices)
    t.pre_agg_df = df if (agg_rows or agg_cols) else None
    t.agg_rows_removed = agg_rows
    t.agg_cols_removed = agg_cols
    t.agg_rows_removed_positions = agg_row_positions
    t.agg_removed_row_metadata = agg_row_meta
    t.agg_removed_col_metadata = agg_col_meta

    unit_split = detect_and_split_units(cleaned_df)
    if unit_split:
        t.pre_unit_split_df = cleaned_df
        cleaned_df = unit_split["cleaned_df"]
        t.unit_master_df = unit_split["master_df"]
        t.unit_split_info = {
            "label_col": unit_split["label_col"],
            "master_col": unit_split["master_col"],
            "mapping": unit_split["mapping"],
            "match_count": unit_split["match_count"],
        }

    return cleaned_df


def normalize_tables(tables: List[Any], filename: Optional[str] = None) -> None:
    """検出済みテーブル（DetectedTable）に Step3 の整形処理一式を適用する。

    各テーブルに対し、次の順序で処理する（各処理の出力が次の処理の入力になる）:
      1. Transpose検出・変換（LLM、step3_normalize_llm）— 他の処理はこの表が
         正しい向き（エンティティ＝行、属性＝列）であることを前提とするため最初に行う
      2. 多軸ヘッダー展開（決定論的分類＋必要時のみLLM） — 多段ヘッダーの
         列構造を確定させる処理のため、他の整形より前に行う（Transpose適用時は
         raw_header_rows が転置前の構造を指すため対象外）
      3. グルーピング列の前方補完
      4. 「うち」書きの内訳を別テーブルへ分離 — remove_aggregates は本来、
         親として参照される「合計」行も冗長行とみなして削除してしまうため、
         うち分離を必ず5.より前に行った上で、親として使われた行の
         インデックス（apply_uchi_split が返す protected_indices）を
         remove_aggregates に渡し、その行が本体テーブルから消えないよう
         保護する（内訳テーブル側は親の値を保持しているだけなので、本体
         からも合計行自体が消えると対応関係を追えなくなるため）。
         分離結果は独立した新規 DetectedTable として tables に追加し、以降の
         整形処理（5〜6）の対象にする（メインテーブルと同じ「実テーブル」として
         Step4のテーブル関係分析・Step6のテーブル選択にもそのまま乗る）。
      5. 集計行・集計列の除去
      6. 単位混在の分離（指標マスタ生成）
    全テーブルに対して上記が完了した後、テーブル間で独立な処理として:
      7. クロス集計形式（Wide_to_long含む）の検出と縦持ち変換

    DetectedTable の各フィールドを in-place で書き換え、tables 自体にも
    うち内訳テーブルを追加する（呼び出し元が持つ同一リストへの参照を
    直接変更するため、st.session_state.detected_tables 等にもそのまま反映される）。
    LLM クライアントは一度だけ生成し、全テーブルで使い回す。
    """
    from .models import DetectedTable

    llm_client, llm_model = make_transpose_client()
    new_tables: List[Any] = []

    for t in tables:
        if t.df is None or t.df.empty:
            continue

        df = t.df

        transpose_result = detect_transpose(df, llm_client, llm_model)
        if transpose_result:
            t.pre_transpose_df = df
            df = apply_transpose(df, transpose_result["entity_axis_name"])
            t.transpose_info = transpose_result

        # 多軸ヘッダー展開（B-09）: Transpose適用時は raw_header_rows が
        # 元の（転置前の）列構造を指すため対象外とする
        if not transpose_result and t.raw_header_rows:
            axis_info = detect_multi_axis_header(t.raw_header_rows)
            if axis_info is not None:
                if axis_info["candidate_idxs"]:
                    axis_values = [
                        axis_info["values"][i] for i in axis_info["candidate_idxs"]
                    ]
                    axis_result = detect_dimension_axes(
                        axis_values, t.title, llm_client, llm_model
                    )
                    if axis_result is not None:
                        t.pre_multi_axis_df = df
                        df = apply_multi_axis_header(df, axis_info, axis_result)
                        t.multi_axis_info = {
                            **axis_result,
                            "dropped_labels": [
                                axis_info["values"][i][0]
                                for i in axis_info["dropped_idxs"]
                                if axis_info["values"][i]
                            ],
                        }
                elif axis_info["dropped_idxs"]:
                    # LLM不要: 除外可能な行（単一値の注記行・時系列の冗長な
                    # 上位グルーピング行）を除いた素直な列名に置き換える。
                    # これにより後段のクロス集計/Wide_to_long検出が複合列名に
                    # 阻害されず正しく機能するようになる。
                    surviving_idxs = [
                        i
                        for i in range(len(t.raw_header_rows))
                        if i not in axis_info["dropped_idxs"]
                    ]
                    if surviving_idxs:
                        surviving_rows = [
                            axis_info["ffilled_rows"][i] for i in surviving_idxs
                        ]
                        new_cols = _dedup_columns(
                            merge_header_rows(
                                surviving_rows,
                                ["name"] * len(surviving_rows),
                                len(surviving_rows[0]),
                            )
                        )
                        df = df.copy()
                        df.columns = new_cols

        pre_fill_df_candidate = df
        df, filled_cols = fill_grouping_cols(df)
        t.filled_cols = filled_cols
        t.pre_fill_df = pre_fill_df_candidate if filled_cols else None

        uchi_info = detect_uchi_breakdown(df)
        uchi_protected_indices: set = set()
        if uchi_info:
            t.pre_uchi_split_df = df
            df, breakdown_df, uchi_protected_indices = apply_uchi_split(df, uchi_info)
            t.uchi_split_info = uchi_info
            t.uchi_breakdown_df = breakdown_df

            child_id = f"{t.table_id}_uchi_breakdown"
            child_title = f"{t.title or t.table_id} 内訳テーブル"
            new_tables.append(
                DetectedTable(
                    table_id=child_id,
                    sheet_name=t.sheet_name,
                    start_row=t.start_row,
                    end_row=t.end_row,
                    start_col=t.start_col,
                    end_col=t.end_col,
                    df=breakdown_df,
                    title=child_title,
                    is_step3_derived=True,
                )
            )

        t.df = _apply_agg_and_unit_split(t, df, protected_indices=uchi_protected_indices)

    # ── うち内訳テーブルにも集計除去・単位分離を適用する ────────────
    # Transpose・グルーピング列前方補完・うち検出自身は対象外（既に整形済みの
    # 派生テーブルであり、親の段階で確定した構造を再度崩す必要はないため）。
    for ct in new_tables:
        ct.df = _apply_agg_and_unit_split(ct, ct.df)

    tables.extend(new_tables)

    # ── クロス集計検出（テーブル間で独立、全テーブル走査後に一括適用）────
    # Wide_to_long（軸×複数指標の複合列名。軸は時系列に限らない）を先に試す。
    # 指標が1種類のみの場合は None を返す設計のため、detect_cross_table
    # （単一指標）とは互いに排他的に発火する。
    for t in tables:
        if t.df is None or t.df.empty:
            continue

        wtl_info = detect_wide_to_long(
            t.df, title=t.title, filename=filename, client=llm_client, model=llm_model
        )
        if wtl_info:
            t.pre_wide_to_long_df = t.df
            t.wide_to_long_info = wtl_info
            stacked = stack_wide_to_long(t.df, wtl_info)
            axis_var_name = wtl_info.get("axis_var_name", "")
            if axis_var_name and axis_var_name in stacked.columns:
                agg_mask = stacked[axis_var_name].astype(str).apply(_is_agg_label)
                if agg_mask.any():
                    stacked = stacked[~agg_mask].reset_index(drop=True)
            t.stacked_df = stacked
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
