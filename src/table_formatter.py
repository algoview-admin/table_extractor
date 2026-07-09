"""
テーブル整形モジュール。

検出テーブルのデータをテーブル関係分析に適した形式に整形する。
現在実装されている整形機能:
  - 多段ヘッダーの統合: 名前行・単位行が複数行にわたる場合に 1 行へマージする
  - 集計行・集計列の除去: 計、合計、累計 等のラベルを持つ行・列を除去する

今後追加が想定される整形機能の例:
  - 欠損値・秘匿値マーカーの数値変換
  - データ型の推定と変換
  - 行ヘッダー列の正規化
"""

import re as _re
from typing import Any, Dict, List, Optional, Tuple


# --- 既知単位語彙（小文字比較用） ---
UNIT_VOCAB: frozenset = frozenset({
    # 日本語単位
    "人", "名", "千人", "万人",
    "円", "千円", "万円", "百万円", "億円", "兆円",
    "%", "％",
    "ha", "㎡", "m²", "km²", "m³",
    "m", "km", "cm", "mm",
    "t", "kg", "g", "トン",
    "kl", "l",
    "戸", "棟", "件", "社", "箇所", "か所", "ヶ所", "店", "台", "基",
    "千", "百万", "億", "万",
    "kw", "kwh", "mw",
    # 英語単位（小文字）
    "persons", "person",
    "yen", "mil. yen", "thou. yen", "billion yen", "million yen", "1,000 yen",
    "percent", "ratio", "rate", "index",
    "ha", "t", "ton", "tons", "kg", "g",
    "kl", "number", "numbers",
    "cases", "units", "households", "establishments", "workers",
    "mil.", "thou.", "million", "billion", "thousand",
    "kw", "kwh", "mw",
})


def is_unit_row(row: List[Any]) -> bool:
    """行が単位ラベルのみで構成されているかを判定する。

    以下の条件をすべて満たす場合に True を返す:
      - 全ての非空セルが 25 文字以内（単位は必ず短い）
      - 非空セルの 50% 以上が既知単位語彙に一致、
        または全セルが 15 文字以内かつ繰り返し率が高い（同一単位が複数列に並ぶ）
    """
    vals = [
        str(v).strip() for v in row
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
        role = "unit" if is_unit_row(row) else "name"
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
        return any("぀" <= c <= "鿿" or "豈" <= c <= "﫿" for c in s)

    cjk_cells = sum(1 for t in texts if _has_cjk(t))
    ascii_cells = sum(1 for t in texts if not _has_cjk(t) and any(c.isalpha() and c.isascii() for c in t))

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
                filtered = [p for p, l in zip(pairs, pair_langs) if l in (dominant, "other")]
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

# 集計を示すキーワード（小文字で比較）
AGG_KEYWORDS: frozenset = frozenset({
    "計", "合計", "小計", "集計", "累計", "総計", "総合計", "合計額",
    "total", "subtotal", "grand total", "sum", "cumulative", "aggregate",
    "year-to-date", "ytd",
})

# 全角スペース・ゼロ幅文字など ASCII strip で取れない空白類の正規化
_WS_RE = _re.compile(r'[\s　\xa0​‌‍﻿]+')

# 末尾の括弧注記パターン: 「（参考）」「[除く海外]」「(注1)」など
# 1つ以上の括弧グループ（各種括弧の対）が末尾に続く場合にマッチ
_TRAILING_BRACKET_RE = _re.compile(
    r'[\s　]*(?:[（(【〔「\[]\s*[^)）】〕」\]]*\s*[)）】〕」\]])+\s*$'
)


def _normalize_label(s: str) -> str:
    """ラベル比較用に Unicode 空白類・制御文字を除去して正規化する。"""
    return _WS_RE.sub('', s).strip()


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
    stripped = _TRAILING_BRACKET_RE.sub('', s).strip()
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
        for c2 in valid[i + 1:]:
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

    context_cols = [
        c for c in ctx_cols
        if c != col and c not in covar.get(col, set())
    ]

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
        return any(
            not _is_agg_label(str(v))
            for v in matching_vals
            if not _isnull(v)
        )

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
    return any(
        not _is_agg_label(str(v))
        for v in df[col]
        if not _isnull(v)
    )


def remove_aggregates(
    df: Any,  # pd.DataFrame
) -> Tuple[Any, List[Dict[str, Any]], List[str], List[int]]:
    """集計行・集計列を除去した DataFrame と除去情報を返す。

    集計列: 列名がキーワードに一致するもの。
    集計行: dtype==object の列（ラベル列）に集計ラベルを持ち、かつその集計が「冗長」
            （同じ文脈で個別データが存在する）場合のみ除去する。
            個別データが存在しない場合（例: 全行が '全国計' のみの年次）は除去しない。

    Returns:
        cleaned_df          — 集計行・集計列を除去した DataFrame（index リセット済み）
        removed_rows_info   — 除去した各行のラベル列値 [{col: val, ...}, ...]
        removed_col_names   — 除去した列名リスト
        removed_row_indices — 除去した行の元 DataFrame 上の整数インデックスリスト
    """
    import pandas as pd

    # ── 集計列の検出 ──────────────────────────────────────────────
    removed_cols: List[str] = [
        col for col in df.columns if _is_agg_label(str(col))
    ]

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

    # 日本の統計データでよく使われる秘匿・欠損マーカー（数値列判定時にnullと同等に扱う）
    _STAT_NA_MARKERS: frozenset = frozenset({
        'x', '***', '**', '*', '-', '…', '－', '–', '―', 'na', 'n/a', 'n.a.', '?',
    })

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
        col for col in df.columns
        if col not in removed_cols
        and _is_text_dtype(df[col])
        and not _is_numeric_values_col(df[col])
    ]

    # ── 冗長性チェック用コンテキスト列（カテゴリ・コード列のみ）────
    """
    数値データ列（売上額・給与額など）はほぼ全行が異なる値を持ち、
    コンテキストに含めると「完全一致行ゼロ」になり冗長性判定が誤る。
    ユニーク値の割合 < 50% の列（同じ値が繰り返し現れる列）のみ使用する。
    """
    n_rows = len(df)
    ctx_cols: List[str] = [
        col for col in label_cols
        if _is_grouping_col(df[col], n_rows)
    ]

    # ── コード↔ラベルなど共変ペアを検出（ctx_cols ベース）──────────
    # ctx_cols のみを対象にすることで、高基数列による誤検出を防ぐ。
    covar = _detect_covariates(df, ctx_cols)

    # ── 集計行の検出（冗長性チェック付き）────────────────────────
    removed_row_indices: List[int] = []
    removed_rows_info: List[Dict[str, Any]] = []

    # 各ラベル列について「列全体に非集計値が存在するか」をキャッシュしておく
    col_has_nonag: Dict[str, bool] = {}
    for _lc in label_cols:
        col_has_nonag[_lc] = any(
            not _is_agg_label(str(v))
            for v in df[_lc]
            if not _is_null_scalar(v)
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
                break

    # ── 変更がなければ None を返してスキップを示す ────────────────
    if not removed_row_indices and not removed_cols:
        return df, [], [], []

    cleaned = df.drop(index=removed_row_indices, columns=removed_cols, errors="ignore")
    cleaned = cleaned.reset_index(drop=True)

    return cleaned, removed_rows_info, removed_cols, removed_row_indices


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
        if _pat.is_numeric_dtype(series) or _pat.is_bool_dtype(series) or _pat.is_datetime64_any_dtype(series):
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
        if not null_mask.iloc[pos + 1:].any():
            continue  # 末尾以降の None のみ → スキップ

        # ffill を適用し、集計ラベルの過伝播を防ぐ。
        # 例: 列_2 で「計」→ None → None と並ぶ場合、ffill で後続 None が「計」に
        # なるが、これらの None は「サブ分類なし」を意味するので戻す。
        filled = series.ffill()
        original_null = series.isna()
        if original_null.any():
            for _null_idx in df.index[original_null]:
                _fv = filled.at[_null_idx]
                _fv_null = _fv is None or (isinstance(_fv, float) and pd.isna(_fv)) or (pd.NA is not None and _fv is pd.NA)
                if not _fv_null and _is_agg_label(str(_fv)):
                    try:
                        filled.at[_null_idx] = pd.NA  # StringDtype では None より pd.NA が安全
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

# 列名が時系列を表すパターンとその種別
_TIME_PATTERNS: List[Tuple[Any, str]] = [
    (_re.compile(r'^\d{4}年\d{1,2}月$'), 'year_month'),   # 2011年1月
    (_re.compile(r'^\d{1,2}月$'), 'month'),                 # 1月, 12月
    (_re.compile(r'^Q[1-4]$', _re.I), 'quarter'),           # Q1, Q2, Q3, Q4
    (_re.compile(r'^第[1-4一二三四]四半期$'), 'quarter_ja'), # 第1四半期
    (_re.compile(r'^\d{4}年度$'), 'fiscal_year'),            # 2020年度
    (_re.compile(r'^\d{4}年$'), 'year'),                    # 2020年
    (_re.compile(r'^(19|20)\d{2}$'), 'year_num'),           # 2020（4桁年数字）
]

# タイトル・ファイル名から年を抽出する正規表現
_YEAR_CTX_RE = _re.compile(r'(19|20)\d{2}')

# 値列名の推定キーワード
_VALUE_KEYWORDS: Dict[str, str] = {
    "売上": "売上", "予算": "予算", "実績": "実績",
    "件数": "件数", "人数": "人数", "金額": "金額",
    "数量": "数量", "利益": "利益", "費用": "費用",
    "コスト": "コスト",
}

# 種別→縦持ち後の時系列列名
_VAR_NAME_MAP: Dict[str, str] = {
    'year_month': '年月', 'month': '月', 'quarter': '四半期',
    'quarter_ja': '四半期', 'fiscal_year': '年度',
    'year': '年', 'year_num': '年',
}


def _classify_col_time(col_name: str) -> Optional[str]:
    """列名が時系列パターンにマッチするか判定し、種別文字列を返す。"""
    s = str(col_name).strip()
    for pattern, kind in _TIME_PATTERNS:
        if pattern.match(s):
            return kind
    return None


def _extract_year_context(title: Optional[str], filename: Optional[str]) -> Optional[int]:
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
    time_cols  = [col_names[i] for i, t in enumerate(time_types) if t is not None]
    time_kind  = next(t for t in time_types if t is not None)

    # 月のみ列の場合、title/filename から年を補完
    year_context: Optional[int] = None
    if time_kind == 'month':
        year_context = _extract_year_context(title, filename)

    var_name = _VAR_NAME_MAP.get(time_kind, '期間')

    # 値列名の推定（タイトルキーワード優先、なければ "値"）
    value_name = "値"
    if title:
        for kw, name in _VALUE_KEYWORDS.items():
            if kw in title:
                value_name = name
                break

    return {
        'label_cols': label_cols,
        'time_cols': time_cols,
        'time_kind': time_kind,
        'var_name': var_name,
        'value_name': value_name,
        'year_context': year_context,
    }


def stack_cross_table(df: Any, stack_info: Dict[str, Any]) -> Any:
    """クロス集計形式（横持ち）を縦持ち（long format）に変換する。

    月のみ列で year_context が設定されている場合、年列を先頭ラベルの直後に挿入する。
    """
    label_cols   = stack_info['label_cols']
    time_cols    = stack_info['time_cols']
    var_name     = stack_info['var_name']
    value_name   = stack_info['value_name']
    year_context = stack_info.get('year_context')
    time_kind    = stack_info['time_kind']

    melted = df.melt(
        id_vars=label_cols,
        value_vars=time_cols,
        var_name=var_name,
        value_name=value_name,
    )

    # 月のみの場合、年列をラベル列の直後に挿入
    if year_context is not None and time_kind == 'month':
        melted.insert(len(label_cols), '年', year_context)

    return melted.reset_index(drop=True)
