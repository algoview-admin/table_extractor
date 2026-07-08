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
    unique_ratio = len({v.lower() for v in vals}) / len(vals)
    return unique_ratio <= 0.4 and all(len(v) <= 15 for v in vals)


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
    _MAX_HEADER_ROWS = 8
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
    "total", "subtotal", "grand total", "sum", "cumulative",
})


def _is_agg_label(s: str) -> bool:
    """値が集計ラベルか判定する。完全一致 or キーワードで終わる場合に True。"""
    s = s.strip()
    if not s:
        return False
    sl = s.lower()
    for kw in AGG_KEYWORDS:
        kw_l = kw.lower()
        if sl == kw_l or sl.endswith(kw_l):
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
    mask = pd.Series(True, index=df.index)
    for cc in context_cols:
        cv = df.at[idx, cc]
        if cv is None or (isinstance(cv, float) and pd.isna(cv)):
            mask &= df[cc].isna()
        else:
            mask &= df[cc] == cv
    mask.at[idx] = False  # 自行を除外

    # 対象列に非集計値を持つ行が存在するか
    matching_vals = df.loc[mask, col]
    return any(
        not _is_agg_label(str(v))
        for v in matching_vals
        if v is not None and not (isinstance(v, float) and pd.isna(v))
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

    # ── ラベル列の特定（dtype == object の列）──────────────────────
    """
    'X'（秘匿値）などの混入で dtype==object になっているが実質数値の列を除外する。
    int/float オブジェクトが過半数の列はデータ列として label_cols から除く。
    """
    def _is_numeric_values_col(series: Any) -> bool:
        non_null = [v for v in series if v is not None and not (isinstance(v, float) and pd.isna(v))]
        if not non_null:
            return False
        n_num = sum(1 for v in non_null if isinstance(v, (int, float)) and not (isinstance(v, float) and pd.isna(v)))
        return n_num / len(non_null) >= 0.5

    label_cols: List[str] = [
        col for col in df.columns
        if col not in removed_cols
        and df[col].dtype == object
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

    for idx in df.index:
        for col in label_cols:
            val = df.at[idx, col]
            if val is None:
                continue
            if isinstance(val, float) and pd.isna(val):
                continue
            if not _is_agg_label(str(val)):
                continue
            # 集計ラベルを持つ列を発見。同じ文脈に個別データが存在する場合のみ除去。
            if col not in ctx_cols or _is_redundant_agg_row(df, idx, col, ctx_cols, covar):
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
