"""
ステップ3 テーブル正規化モジュール（決定論的処理）。

処理概要: 検出された生 DataFrame を分析に適した形式に正規化する。
          多段ヘッダーの統合・集計行列の除去・グルーピング列の補完・
          単位混在の分離（指標マスタ生成）・クロス集計の縦持ち変換など、
          正規表現・語彙辞書のみで完結する（LLMを使わない）処理を扱う。
          LLMを使用する処理（Transpose検出等）は
          src/step3_normalize_llm.py を参照。
          normalize_tables() が両者を正しい順序で呼び出す統括関数。
入力    : DetectedTable.df（step2_detect が構築した生 DataFrame）
出力    : 正規化済み DataFrame、整形メタ情報
          （transpose_info, filled_cols, stack_info, agg_rows_removed,
          unit_split_info 等を DetectedTable に付与）
"""

import re as _re
from typing import Any, Dict, List, Optional, Tuple

from .keywords import (
    AGG_KEYWORDS,
    STAT_NA_MARKERS as _STAT_NA_MARKERS,
    TIME_PATTERNS as _TIME_PATTERNS,
    UNIT_VOCAB,
    VALUE_KEYWORDS as _VALUE_KEYWORDS,
    VAR_NAME_MAP as _VAR_NAME_MAP,
    VAR_NAME_FALLBACK as _VAR_NAME_FALLBACK,
)
from .step3_normalize_llm import apply_transpose, detect_transpose, make_transpose_client


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


def remove_aggregates(
    df: Any,  # pd.DataFrame
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

    集計列: 列名がキーワードに一致するもの。
    集計行: dtype==object の列（ラベル列）に集計ラベルを持ち、かつその集計が「冗長」
            （同じ文脈で個別データが存在する）場合のみ除去する。
            個別データが存在しない場合（例: 全行が同一の集計ラベルのみの区分）は除去しない。

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

    # ── 集計列の検出 ──────────────────────────────────────────────
    removed_cols: List[str] = [col for col in df.columns if _is_agg_label(str(col))]

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
        if col not in removed_cols
        and _is_text_dtype(df[col])
        and not _is_numeric_values_col(df[col])
    ]

    # ── 数値（集計対象）列 — ラベル列でも除去済み集計列でもない列 ──
    """
    メタデータの context には文字列型のラベル列を使う。

    数値列側にもユニーク率による絞り込みを追加すると、値の重複が多いだけの
    正当な集計対象列まで誤って除外してしまうことを実データ検証で確認したため、
    数値列は絞り込まずすべて集計対象候補として扱う。
    """
    value_cols: List[str] = [
        col for col in df.columns if col not in removed_cols and col not in label_cols
    ]

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
# Wide_to_long（時系列×複数指標の複合列名）検出と変換
# ---------------------------------------------------------------------------
#
# detect_cross_table は「列名が時系列トークンそのもの」（例: 2023年）である
# 横持ち表しか検出できない。ここでは「支店,2023売上,2023原価」のように
# 時系列と指標名が1つの列名に合成された複合表記の横持ち表を検出し、
# 時系列軸のみを縦持ちに変換する（指標は列として維持する）。
#
# 列名の意味（指標名が何を表すか）を理解する必要はなく、既存の
# _classify_col_time（完全一致の時系列パターン判定）をあらゆる分割点で
# 適用するだけで判定できるため、LLM を使わず決定論的に実装する。

_WIDE_TO_LONG_MATCH_RATIO = 0.8  # 時系列+指標に分解できる列が占めるべき最低割合
_WIDE_TO_LONG_COMPLETENESS = 0.6  # 想定グリッド（時系列数×指標数）に対する実列数の最低割合


def _split_time_indicator(col_name: str) -> Optional[Tuple[str, str, str]]:
    """列名を (time_token, indicator_name, time_kind) に分解する。

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


def detect_wide_to_long(
    df: Any,
    title: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    DataFrame が「時系列×複数指標」の複合列名を持つ横持ち表かを検出する。

    判定条件（すべて必須）:
      - 列名が時系列トークン単体（_classify_col_time が非Noneを返す）の列は
        対象外とする（detect_cross_table の担当領域と重複させない）
      - 残りの列のうち、時系列+指標に分解できる列が全体の
        _WIDE_TO_LONG_MATCH_RATIO（既定0.8）以上
      - 検出された時系列トークンが2種類以上、かつ指標名が2種類以上
        （指標が1種類のみなら detect_cross_table の担当。この条件により
        両検出は互いに排他になる）
      - 想定グリッド（時系列数×指標数）に対する実列数の割合が
        _WIDE_TO_LONG_COMPLETENESS（既定0.6）以上（歯抜けが多すぎる場合は
        誤検出とみなす）

    Returns:
      検出された場合: {"label_cols", "time_var_name", "time_kind",
                       "time_tokens", "indicators", "parsed_cols"}
      検出されなかった場合: None
    """
    if df is None or df.empty or len(df.columns) < 4:
        return None

    col_names = [str(c) for c in df.columns]

    parsed: Dict[str, Tuple[str, str, str]] = {}
    candidate_cols = [c for c in col_names if _classify_col_time(c) is None]
    if not candidate_cols:
        return None
    for c in candidate_cols:
        result = _split_time_indicator(c)
        if result:
            parsed[c] = result

    if len(parsed) / len(candidate_cols) < _WIDE_TO_LONG_MATCH_RATIO:
        return None

    time_tokens = sorted({t for t, _, _ in parsed.values()})
    indicators: List[str] = []
    for _, ind, _ in parsed.values():
        if ind not in indicators:
            indicators.append(ind)

    if len(time_tokens) < 2 or len(indicators) < 2:
        return None  # 指標が1種類のみなら detect_cross_table の担当

    expected = len(time_tokens) * len(indicators)
    if len(parsed) / expected < _WIDE_TO_LONG_COMPLETENESS:
        return None

    label_cols = [c for c in col_names if c not in parsed]
    time_kind = next(iter(parsed.values()))[2]
    time_var_name = _VAR_NAME_MAP.get(time_kind, _VAR_NAME_FALLBACK)

    return {
        "label_cols": label_cols,
        "time_var_name": time_var_name,
        "time_kind": time_kind,
        "time_tokens": time_tokens,
        "indicators": indicators,
        "parsed_cols": parsed,
    }


def stack_wide_to_long(df: Any, info: Dict[str, Any]) -> Any:
    """detect_wide_to_long の検出結果を使って横持ち→縦持ち変換する。

    時系列トークンごとにサブフレームを作り（指標列は元の列出現順を維持し、
    該当する元列が存在しない組み合わせは NaN で埋めてグリッドの歯抜けを
    許容する）、縦に連結する。
    """
    import pandas as pd

    label_cols = info["label_cols"]
    time_var_name = info["time_var_name"]
    time_tokens = info["time_tokens"]
    indicators = info["indicators"]
    parsed_cols = info["parsed_cols"]

    frames = []
    for token in time_tokens:
        sub = df[label_cols].copy()
        sub.insert(len(label_cols), time_var_name, token)
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


def normalize_tables(tables: List[Any], filename: Optional[str] = None) -> None:
    """検出済みテーブル（DetectedTable）に Step3 の整形処理一式を適用する。

    各テーブルに対し、次の順序で処理する（各処理の出力が次の処理の入力になる）:
      1. Transpose検出・変換（LLM、step3_normalize_llm）— 他の処理はこの表が
         正しい向き（エンティティ＝行、属性＝列）であることを前提とするため最初に行う
      2. グルーピング列の前方補完
      3. 集計行・集計列の除去
      4. 単位混在の分離（指標マスタ生成）
    全テーブルに対して上記が完了した後、テーブル間で独立な処理として:
      5. クロス集計形式の検出と縦持ち変換

    DetectedTable の各フィールドを in-place で書き換える（戻り値なし）。
    LLM クライアントは一度だけ生成し、全テーブルで使い回す。
    """
    llm_client, llm_model = make_transpose_client()

    for t in tables:
        if t.df is None or t.df.empty:
            continue

        df = t.df

        transpose_result = detect_transpose(df, llm_client, llm_model)
        if transpose_result:
            t.pre_transpose_df = df
            df = apply_transpose(df, transpose_result["entity_axis_name"])
            t.transpose_info = transpose_result

        pre_fill_df_candidate = df
        df, filled_cols = fill_grouping_cols(df)
        t.filled_cols = filled_cols
        t.pre_fill_df = pre_fill_df_candidate if filled_cols else None

        (
            cleaned_df,
            agg_rows,
            agg_cols,
            agg_row_positions,
            agg_row_meta,
            agg_col_meta,
        ) = remove_aggregates(df)
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

        t.df = cleaned_df

    # ── クロス集計検出（テーブル間で独立、全テーブル走査後に一括適用）────
    # Wide_to_long（時系列×複数指標の複合列名）を先に試す。指標が1種類のみの
    # 場合は None を返す設計のため、detect_cross_table（単一指標）とは
    # 互いに排他的に発火する。
    for t in tables:
        if t.df is None or t.df.empty:
            continue

        wtl_info = detect_wide_to_long(t.df, title=t.title, filename=filename)
        if wtl_info:
            t.pre_wide_to_long_df = t.df
            t.wide_to_long_info = wtl_info
            t.stacked_df = stack_wide_to_long(t.df, wtl_info)
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
