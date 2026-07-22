import html as _html
import json
from typing import Dict, List, Optional, Set

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from streamlit_ui.shared import _go_to, _inject_splitter_js, _splitter_marker
from src.models import DetectedTable
from src.step3_normalize_determ import _is_agg_label, normalize_tables

_TH_STYLE = (
    "position:sticky;top:0;z-index:2;"
    "background-color:var(--background-color,#0f1117);"
    "background-image:linear-gradient(rgba(66,153,225,0.20),rgba(66,153,225,0.20));"
    "color:var(--text-color,#fafafa);"
    "padding:6px 12px;"
    "text-align:left;"
    "border-bottom:2px solid rgba(66,153,225,0.5);"
    "white-space:nowrap;"
    "font-weight:600;"
    "font-size:13px;"
)
_TD_STYLE = (
    "padding:4px 12px;"
    "font-size:13px;"
    "border-bottom:1px solid rgba(255,255,255,0.06);"
)
_TD_CENTER_STYLE = (
    "padding:4px 10px;"
    "font-size:13px;"
    "text-align:center;"
    "white-space:nowrap;"
    "border-bottom:1px solid rgba(255,255,255,0.06);"
)


def _df_to_html(
    df: pd.DataFrame,
    max_height: Optional[int] = None,
    highlight_row_count: int = 0,
    highlight_row_indices: Optional[set] = None,
    amber_row_indices: Optional[set] = None,
    highlight_col_names: Optional[set] = None,
    unit_col_names: Optional[set] = None,
    green_col_names: Optional[set] = None,
) -> str:
    """DataFrameをモダンなスタイルのHTMLテーブルに変換する。
    max_height を指定すると縦スクロール可能なコンテナで包む。
    highlight_row_count > 0 の場合、先頭 N 行を赤色強調表示する。
    highlight_row_indices: 赤色強調する行の位置インデックス集合（本当に破棄される行）。
    amber_row_indices: 黄色強調する行の位置インデックス集合（破棄はされず、
      列名として採用される行。highlight_row_indices と重複する場合は赤を優先）。
    highlight_col_names: オレンジ色ヘッダーで示す除去列名集合。
    unit_col_names: 紫色ヘッダーで示す単位付加列名集合。
    green_col_names: 緑色ヘッダーで示す前方補完列名集合。"""
    col_names = list(df.columns)
    orange_pos: set = {
        j
        for j, c in enumerate(col_names)
        if highlight_col_names and str(c) in highlight_col_names
    }
    purple_pos: set = {
        j
        for j, c in enumerate(col_names)
        if unit_col_names and str(c) in unit_col_names
    }
    green_pos: set = {
        j
        for j, c in enumerate(col_names)
        if green_col_names and str(c) in green_col_names
    }

    def _th(j: int, c: str) -> str:
        label = _html.escape(str(c))
        if j in orange_pos:
            return (
                f"<th style='{_TH_STYLE}"
                f"background-image:linear-gradient(rgba(255,140,0,0.25),rgba(255,140,0,0.25));"
                f"color:rgba(200,100,0,0.9);border-bottom:2px solid rgba(255,140,0,0.5)'>"
                f"{label}</th>"
            )
        if j in purple_pos:
            return (
                f"<th style='{_TH_STYLE}"
                f"background-image:linear-gradient(rgba(124,58,237,0.35),rgba(124,58,237,0.35));"
                f"color:rgba(221,214,254,1.0);border-bottom:2px solid rgba(167,139,250,0.7)'>"
                f"{label}</th>"
            )
        if j in green_pos:
            return (
                f"<th style='{_TH_STYLE}"
                f"background-image:linear-gradient(rgba(16,185,129,0.25),rgba(16,185,129,0.25));"
                f"color:rgba(16,185,129,1.0);border-bottom:2px solid rgba(16,185,129,0.5)'>"
                f"{label}</th>"
            )
        return f"<th style='{_TH_STYLE}'>{label}</th>"

    headers = "".join(_th(j, c) for j, c in enumerate(col_names))
    rows_parts = []
    for i, (_, row) in enumerate(df.iterrows()):
        is_red = (i < highlight_row_count) or (
            highlight_row_indices is not None and i in highlight_row_indices
        )
        is_amber = not is_red and amber_row_indices is not None and i in amber_row_indices
        if is_red:
            cells = "".join(
                f"<td style='{_TD_STYLE}"
                f"{'border-left:3px solid rgba(220,50,50,0.55);' if j == 0 else ''}"
                f"color:rgba(220,50,50,0.9);'>"
                f"{_html.escape(str(v))}</td>"
                for j, v in enumerate(row)
            )
            rows_parts.append(
                f"<tr style='background:rgba(239,68,68,0.10);'>{cells}</tr>"
            )
        elif is_amber:
            cells = "".join(
                f"<td style='{_TD_STYLE}"
                f"{'border-left:3px solid rgba(217,119,6,0.55);' if j == 0 else ''}"
                f"color:rgba(217,119,6,0.95);'>"
                f"{_html.escape(str(v))}</td>"
                for j, v in enumerate(row)
            )
            rows_parts.append(
                f"<tr style='background:rgba(217,119,6,0.10);'>{cells}</tr>"
            )
        else:
            cells = "".join(
                f"<td style='{_TD_STYLE}'>{_html.escape(str(v))}</td>" for v in row
            )
            rows_parts.append(f"<tr>{cells}</tr>")
    rows = "".join(rows_parts)
    scroll_style = (
        f"overflow-x:auto;overflow-y:auto;max-height:{max_height}px"
        if max_height
        else "overflow-x:auto"
    )
    return (
        f"<div style='{scroll_style}'>"
        "<table style='border-collapse:separate;border-spacing:0;width:100%'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table></div>"
    )


def _render_merge_detail_body(t: "DetectedTable") -> None:
    """列名対応表 + before/after プレビュー（expander なし）。"""
    raw = t.raw_df
    # ①単純統合"自身"の結果と比較するため、後続処理（②軸展開・Transpose・
    # うち分離・集計除去）でそれぞれ上書きされる直前のスナップショットのうち、
    # パイプライン順（①→Transpose→②軸展開→うち分離→集計除去）で最も早く
    # 捕捉されたものを優先する。これを誤ると②軸展開後の最終結果が「①の
    # 整形後」として二重表示されてしまう。
    fmt = (
        t.pre_transpose_df
        if t.pre_transpose_df is not None
        else (
            t.pre_multi_axis_df
            if t.pre_multi_axis_df is not None
            else (
                t.pre_uchi_split_df
                if t.pre_uchi_split_df is not None
                else (t.pre_agg_df if t.pre_agg_df is not None else t.df)
            )
        )
    )
    n_residue = len(raw) - len(fmt)
    # header_merge_discarded_row_indices が分かれば、残留ヘッダー行のうち
    # 「本当に破棄される行」と「列名として採用される行」を色分けする。
    # 情報がない場合（軸展開が適用された等）は全て破棄扱いにフォールバックする。
    discarded_idx = t.header_merge_discarded_row_indices
    if discarded_idx is None:
        truly_discarded = set(range(n_residue))
        adopted = set()
    else:
        truly_discarded = {i for i in discarded_idx if i < n_residue}
        adopted = {i for i in range(n_residue) if i not in truly_discarded}

    if adopted:
        st.caption(
            f"ヘッダー行を統合しました（赤色 {len(truly_discarded)} 行は破棄、"
            f"黄色 {len(adopted)} 行は列名として採用）"
        )
    else:
        st.caption(
            f"ヘッダー行を統合し、残留ヘッダー {n_residue} 行をデータから除去しました"
        )

    before_cols = list(raw.columns)
    after_cols = list(fmt.columns)
    max_len = max(len(before_cols), len(after_cols))
    col_diff_df = pd.DataFrame(
        {
            "整形前（第1行のみ）": before_cols + [""] * (max_len - len(before_cols)),
            "整形後（多段マージ）": after_cols + [""] * (max_len - len(after_cols)),
        }
    ).assign(
        変化=lambda df: df.apply(
            lambda r: (
                "→" if r["整形前（第1行のみ）"] != r["整形後（多段マージ）"] else "="
            ),
            axis=1,
        )
    )[
        ["整形前（第1行のみ）", "変化", "整形後（多段マージ）"]
    ]

    st.markdown("**列名の変化**")
    rows_html = "".join(
        "<tr>"
        + f"<td style='{_TD_STYLE}'>{_html.escape(str(r['整形前（第1行のみ）']))}</td>"
        + f"<td style='{_TD_CENTER_STYLE}'>{_html.escape(str(r['変化']))}</td>"
        + f"<td style='{_TD_STYLE}'>{_html.escape(str(r['整形後（多段マージ）']))}</td>"
        + "</tr>"
        for _, r in col_diff_df.iterrows()
    )
    headers_html = (
        f"<th style='{_TH_STYLE}'>整形前（第1行のみ）</th>"
        f"<th style='{_TH_STYLE}text-align:center;'>変化</th>"
        f"<th style='{_TH_STYLE}'>整形後（多段マージ）</th>"
    )
    st.markdown(
        "<div style='overflow-x:auto'>"
        "<table style='border-collapse:collapse'>"
        f"<thead><tr>{headers_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )

    raw_col_set = {str(c) for c in raw.columns}
    unit_cols = {
        str(c) for c in fmt.columns if "[" in str(c) and str(c) not in raw_col_set
    }
    col_b, col_a = st.columns(2)
    with col_b:
        _before_title = f"**整形前**（全件 / 赤色 {len(truly_discarded)} 行が破棄対象"
        _before_title += f" / 黄色 {len(adopted)} 行が列名採用" if adopted else ""
        _before_title += "）"
        st.markdown(_before_title)
        st.markdown(
            _df_to_html(
                raw.astype(str),
                max_height=340,
                highlight_row_indices=truly_discarded,
                amber_row_indices=adopted,
            ),
            unsafe_allow_html=True,
        )
    with col_a:
        _after_hint = "紫列 = 単位付加" if unit_cols else ""
        st.markdown(f"**整形後**（全件{' / ' + _after_hint if _after_hint else ''}）")
        st.markdown(
            _df_to_html(
                fmt.astype(str), max_height=340, unit_col_names=unit_cols or None
            ),
            unsafe_allow_html=True,
        )


def _merge_detail_body_html(t: "DetectedTable") -> str:
    """列名対応表 + before/after プレビューをHTML文字列で返す（ネスト details 用）。"""
    raw = t.raw_df
    # ①単純統合"自身"の結果と比較するため、パイプライン順（①→Transpose→
    # ②軸展開→うち分離→集計除去）で最も早く捕捉されたスナップショットを
    # 優先する（②軸展開後の最終結果が「①の整形後」として二重表示されるのを防ぐ）。
    fmt = (
        t.pre_transpose_df
        if t.pre_transpose_df is not None
        else (
            t.pre_multi_axis_df
            if t.pre_multi_axis_df is not None
            else (
                t.pre_uchi_split_df
                if t.pre_uchi_split_df is not None
                else (t.pre_agg_df if t.pre_agg_df is not None else t.df)
            )
        )
    )
    n_residue = len(raw) - len(fmt)
    discarded_idx = t.header_merge_discarded_row_indices
    if discarded_idx is None:
        truly_discarded = set(range(n_residue))
        adopted = set()
    else:
        truly_discarded = {i for i in discarded_idx if i < n_residue}
        adopted = {i for i in range(n_residue) if i not in truly_discarded}

    before_cols = list(raw.columns)
    after_cols = list(fmt.columns)
    max_len = max(len(before_cols), len(after_cols))

    rows_html = ""
    for i in range(max_len):
        bc = _html.escape(before_cols[i]) if i < len(before_cols) else ""
        ac = _html.escape(after_cols[i]) if i < len(after_cols) else ""
        arrow = "→" if bc != ac else "="
        rows_html += (
            f"<tr>"
            f"<td style='{_TD_STYLE}'>{bc}</td>"
            f"<td style='{_TD_CENTER_STYLE}'>{arrow}</td>"
            f"<td style='{_TD_STYLE}'>{ac}</td>"
            f"</tr>"
        )
    headers_html = (
        f"<th style='{_TH_STYLE}'>整形前（第1行のみ）</th>"
        f"<th style='{_TH_STYLE}text-align:center;'>変化</th>"
        f"<th style='{_TH_STYLE}'>整形後（多段マージ）</th>"
    )

    PREVIEW_ROWS = 10
    raw_col_set = {str(c) for c in raw.columns}
    unit_cols = {
        str(c) for c in fmt.columns if "[" in str(c) and str(c) not in raw_col_set
    }
    before_tbl = _df_to_html(
        raw.head(PREVIEW_ROWS).astype(str),
        highlight_row_indices={i for i in truly_discarded if i < PREVIEW_ROWS},
        amber_row_indices={i for i in adopted if i < PREVIEW_ROWS},
    )
    after_tbl = _df_to_html(
        fmt.head(PREVIEW_ROWS).astype(str), unit_col_names=unit_cols or None
    )
    _after_hint = " / 紫列 = 単位付加" if unit_cols else ""
    _before_caption = f"整形前（赤色 {len(truly_discarded)} 行が破棄対象"
    _before_caption += f" / 黄色 {len(adopted)} 行が列名採用" if adopted else ""
    _before_caption += f" / 先頭 {PREVIEW_ROWS} 行）"
    _top_caption = (
        f"ヘッダー行を統合しました（赤色 {len(truly_discarded)} 行は破棄、"
        f"黄色 {len(adopted)} 行は列名として採用）"
        if adopted
        else f"ヘッダー行を統合し、残留ヘッダー {n_residue} 行をデータから除去しました"
    )

    return (
        f"<p style='font-size:0.83em;opacity:0.65;margin:0 0 0.6rem'>"
        f"{_html.escape(_top_caption)}</p>"
        f"<p style='font-weight:600;margin:0 0 0.3rem'>列名の変化</p>"
        f"<div style='overflow-x:auto'>"
        f"<table style='border-collapse:collapse'>"
        f"<thead><tr>{headers_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>"
        f"<div style='display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.8rem'>"
        f"<div style='flex:1;min-width:280px'>"
        f"<p style='font-weight:600;margin:0 0 0.3rem'>{_html.escape(_before_caption)}</p>"
        f"{before_tbl}</div>"
        f"<div style='flex:1;min-width:280px'>"
        f"<p style='font-weight:600;margin:0 0 0.3rem'>整形後（先頭 {PREVIEW_ROWS} 行{_after_hint}）</p>"
        f"{after_tbl}</div>"
        f"</div>"
    )


_MHD_CSS = """
<style>
/* ── Level-2: その他の同様処理 ── */
details.mhd-l2 {
    border: 1px solid rgba(127,255,212,0.4);
    border-radius: 8px;
    margin: 0.6rem 0 1.2rem;
    background: rgba(127,255,212,0.04);
    overflow: hidden;
}
details.mhd-l2 > summary {
    padding: 0.65rem 1rem;
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-weight: 700;
    font-size: 0.97rem;
    user-select: none;
    transition: background 0.15s;
    color: rgba(127,255,212,0.9);
}
details.mhd-l2 > summary:hover { background: rgba(127,255,212,0.1); }
details.mhd-l2 > summary::-webkit-details-marker { display: none; }
details.mhd-l2 > summary::before {
    content: "▶";
    font-size: 0.62em;
    opacity: 0.75;
    display: inline-block;
    transition: transform 0.18s ease;
    flex-shrink: 0;
}
details.mhd-l2[open] > summary::before { transform: rotate(90deg); }
details.mhd-l2 > .mhd-body {
    padding: 0.6rem 0.8rem;
    border-top: 1px solid rgba(127,255,212,0.2);
}
/* ── Level-3: 個別テーブル ── */
details.mhd-l3 {
    border: 1px solid rgba(127,255,212,0.22);
    border-radius: 6px;
    margin: 0.35rem 0;
    background: rgba(0,0,0,0.12);
    overflow: hidden;
}
details.mhd-l3 > summary {
    padding: 0.5rem 0.85rem;
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-weight: 500;
    font-size: 0.88rem;
    user-select: none;
    transition: background 0.15s;
}
details.mhd-l3 > summary:hover { background: rgba(127,255,212,0.07); }
details.mhd-l3 > summary::-webkit-details-marker { display: none; }
details.mhd-l3 > summary::before {
    content: "▶";
    font-size: 0.55em;
    opacity: 0.55;
    display: inline-block;
    transition: transform 0.18s ease;
    flex-shrink: 0;
}
details.mhd-l3[open] > summary::before { transform: rotate(90deg); }
details.mhd-l3 > .mhd-body {
    padding: 0.55rem 0.85rem;
    border-top: 1px solid rgba(127,255,212,0.15);
}
</style>
"""

# 無効カラムの検出と削除機能: st.checkbox の視認性向上。アプリ内で st.checkbox
# を使うのはこの機能のみ（他ウィジェットへの影響なし）。Streamlit の
# チェックボックス視覚要素には安定した data-testid が無いため、ラベル直下で
# 隠し input を包む <span> の直後の <div>（常にチェック視覚要素が入る唯一の
# 位置）を隣接セレクタ（span + div）で厳密に対象にする。以前 :not([data-
# testid="stWidgetLabel"]) で除外する方式にしていたが、ラベルのテキスト側
# 要素の実際の構造次第では意図せず一致してしまう場合があったため、
# テキスト側には一切触れない厳密な指定に変更した。チェック中は他処理の
# 「補完済み」「昇格で生まれた列」等と同じ緑（16,185,129）で塗りつぶし、
# 未チェック時も枠線をはっきり出して存在が分かるようにする。
_INVCOL_CHECKBOX_CSS = """
<style>
div[data-testid="stCheckbox"] label > span + div {
    width: 22px !important;
    height: 22px !important;
    min-width: 22px !important;
    max-width: 22px !important;
    flex-shrink: 0 !important;
    border: 2px solid rgba(255,255,255,0.65) !important;
    border-radius: 4px !important;
    background: rgba(255,255,255,0.06) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.15s, border-color 0.15s;
}
div[data-testid="stCheckbox"] label:has(input:checked) > span + div {
    background: rgba(16,185,129,0.9) !important;
    border-color: rgba(16,185,129,1) !important;
}
div[data-testid="stCheckbox"] label > span + div svg {
    stroke: #ffffff !important;
    stroke-width: 3 !important;
    width: 13px !important;
    height: 13px !important;
}
</style>
"""


def _make_details_html(
    label: str, body_html: str, open: bool = False, level: int = 2
) -> str:
    cls = f"mhd-l{level}"
    open_attr = " open" if open else ""
    return (
        f"<details class='{cls}'{open_attr}>"
        f"<summary>{label}</summary>"
        f"<div class='mhd-body'>{body_html}</div>"
        f"</details>"
    )


def _render_header_merge_detail(
    t: "DetectedTable",
    rest: "Optional[List[DetectedTable]]" = None,
) -> None:
    """代表テーブルの詳細 expander（Streamlit）。
    rest がある場合はネスト HTML details でその他を表示する。"""
    title_str = f"  🏷️ `{t.title}`" if t.title else ""
    with st.expander(
        f"**`{t.table_id}`**{title_str}  —  シート: {t.sheet_name}",
        expanded=True,
    ):
        _render_merge_detail_body(t)

        if rest:
            # レベル3: 各テーブルを <details> で包む
            inner_html = ""
            for r in rest:
                r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                lbl = (
                    f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                    f" — シート: {_html.escape(r.sheet_name)}"
                )
                inner_html += _make_details_html(
                    lbl, _merge_detail_body_html(r), open=False, level=3
                )
            # レベル2: 「その他の同様処理」<details>
            outer_html = _MHD_CSS + _make_details_html(
                f"その他の同様処理（{len(rest)} 件）",
                inner_html,
                open=False,
                level=2,
            )
            st.markdown(outer_html, unsafe_allow_html=True)


def _render_transpose_body(t: "DetectedTable") -> None:
    """Transpose（行列逆転）変換の詳細（Streamlit ウィジェット版）。"""
    before = t.pre_transpose_df
    after = t.post_transpose_df if t.post_transpose_df is not None else t.df
    info = t.transpose_info
    if not info or before is None or after is None:
        return

    entity_axis_name = str(info.get("entity_axis_name", ""))
    reasoning = str(info.get("reasoning", ""))

    st.markdown(
        "<div style='margin:4px 0 12px;line-height:2'>"
        f"検出されたエンティティ軸: "
        f"<span style='background:rgba(167,139,250,0.15);color:rgba(167,139,250,1);"
        f"border:1px solid rgba(167,139,250,0.4);border-radius:4px;"
        f"padding:2px 8px;font-size:12px;font-weight:600'>{_html.escape(entity_axis_name)}</span><br>"
        f"理由: {_html.escape(reasoning)}"
        "</div>",
        unsafe_allow_html=True,
    )

    entity_cols = {str(c) for c in before.columns[1:]}
    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown(
            f"**変換前**（{len(before.columns)} 列 × {len(before)} 行 / "
            f"オレンジ列 = 本来エンティティであるべき列群）"
        )
        st.markdown(
            _df_to_html(before, max_height=340, highlight_col_names=entity_cols),
            unsafe_allow_html=True,
        )
    with col_a:
        st.markdown(
            f"**変換後**（{len(after.columns)} 列 × {len(after)} 行 / "
            f"緑列 = 変換で生まれたエンティティ軸列）"
        )
        st.markdown(
            _df_to_html(after, max_height=340, green_col_names={entity_axis_name}),
            unsafe_allow_html=True,
        )


def _render_transpose_body_html(t: "DetectedTable") -> str:
    """Transpose（行列逆転）変換の詳細（HTML文字列版）。"""
    before = t.pre_transpose_df
    after = t.post_transpose_df if t.post_transpose_df is not None else t.df
    info = t.transpose_info
    if not info or before is None or after is None:
        return ""

    entity_axis_name = str(info.get("entity_axis_name", ""))
    reasoning = str(info.get("reasoning", ""))

    meta_html = (
        "<div style='margin:4px 0 12px;line-height:2'>"
        f"検出されたエンティティ軸: "
        f"<span style='background:rgba(167,139,250,0.15);color:rgba(167,139,250,1);"
        f"border:1px solid rgba(167,139,250,0.4);border-radius:4px;"
        f"padding:2px 8px;font-size:12px;font-weight:600'>{_html.escape(entity_axis_name)}</span><br>"
        f"理由: {_html.escape(reasoning)}"
        "</div>"
    )

    entity_cols = {str(c) for c in before.columns[1:]}
    pre_html = _df_to_html(before, max_height=340, highlight_col_names=entity_cols)
    post_html = _df_to_html(after, max_height=340, green_col_names={entity_axis_name})
    grid_html = (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px'>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換前（{len(before.columns)} 列 × {len(before)} 行 / オレンジ列 = 本来エンティティであるべき列群）</p>{pre_html}</div>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換後（{len(after.columns)} 列 × {len(after)} 行 / 緑列 = 変換で生まれたエンティティ軸列）</p>{post_html}</div>"
        "</div>"
    )
    return meta_html + grid_html


def _render_pivot_body(t: "DetectedTable") -> None:
    """Pivot検出と変換（(属性,値)ペアの横持ち昇格）の詳細（Streamlit ウィジェット版）。"""
    before = t.pre_pivot_df
    after = t.post_pivot_df if t.post_pivot_df is not None else t.df
    info = t.pivot_info
    if not info or before is None or after is None:
        return

    key_cols = info.get("key_cols", [])
    attr_col = info.get("attr_col", "")
    value_col = info.get("value_col", "")
    attributes = info.get("attributes", [])
    record_count = info.get("record_count", 0)

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    key_html = " ".join(_badge(c, "156,163,175") for c in key_cols) or "（なし）"
    attr_html = " ".join(_badge(c, "156,163,175") for c in [attr_col, value_col])
    new_col_html = " ".join(_badge(c, "52,211,153") for c in attributes)

    st.markdown(
        "<div style='margin:4px 0 12px;line-height:2'>"
        f"キー列: {key_html}<br>"
        f"属性列/値列: {attr_html}<br>"
        f"列見出しに昇格した属性: {new_col_html}（{len(attributes)} 種類）<br>"
        f"生成されたレコード数: <b>{record_count}</b>"
        "</div>",
        unsafe_allow_html=True,
    )

    new_col_set = set(attributes)
    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown(
            f"**変換前**（{len(before.columns)} 列 × {len(before)} 行 / "
            f"オレンジ列 = 属性列・値列）"
        )
        st.markdown(
            _df_to_html(
                before, max_height=340, highlight_col_names={attr_col, value_col}
            ),
            unsafe_allow_html=True,
        )
    with col_a:
        st.markdown(
            f"**変換後**（{len(after.columns)} 列 × {len(after)} 行 / "
            f"緑列 = 昇格で生まれた列）"
        )
        st.markdown(
            _df_to_html(after, max_height=340, green_col_names=new_col_set),
            unsafe_allow_html=True,
        )


def _render_pivot_body_html(t: "DetectedTable") -> str:
    """Pivot検出と変換（(属性,値)ペアの横持ち昇格）の詳細（HTML 文字列版）。"""
    before = t.pre_pivot_df
    after = t.post_pivot_df if t.post_pivot_df is not None else t.df
    info = t.pivot_info
    if not info or before is None or after is None:
        return ""

    key_cols = info.get("key_cols", [])
    attr_col = info.get("attr_col", "")
    value_col = info.get("value_col", "")
    attributes = info.get("attributes", [])
    record_count = info.get("record_count", 0)

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    key_html = " ".join(_badge(c, "156,163,175") for c in key_cols) or "（なし）"
    attr_html = " ".join(_badge(c, "156,163,175") for c in [attr_col, value_col])
    new_col_html = " ".join(_badge(c, "52,211,153") for c in attributes)
    meta_html = (
        "<div style='margin:4px 0 12px;line-height:2'>"
        f"キー列: {key_html}<br>"
        f"属性列/値列: {attr_html}<br>"
        f"列見出しに昇格した属性: {new_col_html}（{len(attributes)} 種類）<br>"
        f"生成されたレコード数: <b>{record_count}</b>"
        "</div>"
    )

    new_col_set = set(attributes)
    pre_html = _df_to_html(
        before, max_height=340, highlight_col_names={attr_col, value_col}
    )
    post_html = _df_to_html(after, max_height=340, green_col_names=new_col_set)
    grid_html = (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px'>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換前（{len(before.columns)} 列 × {len(before)} 行 / オレンジ列 = 属性列・値列）</p>{pre_html}</div>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換後（{len(after.columns)} 列 × {len(after)} 行 / 緑列 = 昇格で生まれた列）</p>{post_html}</div>"
        "</div>"
    )
    return meta_html + grid_html


def _render_invalid_col_body(t: "DetectedTable") -> None:
    """無効カラム（全欠損列・無名列）の検出結果と調整 UI（Streamlit ウィジェット版）。

    他の整形処理と同様に既定（全欠損列）を自動削除した状態を表示する
    （無名でもデータがある列は既定では削除しない）。列を失う操作のため、
    検出時点の全列 DataFrame（pre_invalid_col_df）を保持しており、
    チェックボックスで現在の削除対象列を選び直せる（復元・追加削除の
    両方に対応）。

    表示用の「現在」は pre_invalid_col_df + 現在の選択状態のみから計算する
    （この機能自身の結果だけを示し、後続のファイル外メタデータ生成機能等が
    追加した列を巻き込まない）。一方、実データ t.df の更新は候補列の増減
    のみを現在の t.df に対して行う（pre から丸ごと作り直すと、pre 取得後に
    後続ステップが追加した列が失われてしまうため）。"""
    candidates = t.invalid_col_candidates
    if not candidates:
        return

    st.markdown(_INVCOL_CHECKBOX_CSS, unsafe_allow_html=True)

    pre = t.pre_invalid_col_df
    removed_names = {c["name"] for c in t.invalid_cols_removed}
    display_after = (
        pre.drop(columns=[n for n in removed_names if n in pre.columns])
        if removed_names
        else pre.copy()
    )

    def _badge(c: Dict) -> str:
        active = c["name"] in removed_names
        color = "255,180,100" if active else "107,114,128"
        state = "削除中" if active else "保持中"
        return (
            f"<code style='background:rgba({color},0.15);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:1px 6px;margin:2px;display:inline-block'>"
            f"{_html.escape(c['name'])}（{_html.escape(c['reason'])} / {state}）</code>"
        )

    st.markdown(
        f"<p style='margin:4px 0 6px'>検出候補（{len(candidates)} 列）: "
        + " ".join(_badge(c) for c in candidates)
        + "</p>",
        unsafe_allow_html=True,
    )
    if removed_names:
        st.caption(
            f"既定で **{len(removed_names)}** 列を削除しました。"
            "不要な列はチェックを外すと復元できます"
            "（無名でもデータがある列は既定では保持。必要なら追加でチェックして削除できます）"
        )
    else:
        st.caption(
            "既定で削除される列はありませんでした"
            "（無名でもデータがある列は既定では保持。必要なら以下で選択して削除できます）"
        )

    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown(
            f"**検出時点**（{len(pre.columns)} 列 / オレンジ列 = 現在削除中の列）"
        )
        st.markdown(
            _df_to_html(pre.astype(str), max_height=300, highlight_col_names=removed_names),
            unsafe_allow_html=True,
        )
    with col_a:
        st.markdown(f"**現在**（{len(display_after.columns)} 列）")
        st.markdown(
            _df_to_html(display_after.astype(str), max_height=300), unsafe_allow_html=True
        )

    with st.form(key=f"invcol_form_{t.table_id}"):
        checks: Dict[str, bool] = {}
        for c in candidates:
            label = f"{c['name']}（{c['reason']} / 非空 {c['nonnull_count']} セル）を削除する"
            checks[c["name"]] = st.checkbox(
                label,
                value=c["name"] in removed_names,
                key=f"invcol_{t.table_id}_{c['position']}",
            )
        submitted = st.form_submit_button("選択を反映")

    if submitted:
        selected = {name for name, checked in checks.items() if checked}
        # 候補列だけを現在の t.df に対して増減させる（pre からの丸ごと再構築は
        # しない）。後続ステップ（ファイル外メタデータ生成等）が既に追加した
        # 列を、無効カラムの選び直しで失わないようにするため。
        to_drop = [
            c["name"] for c in candidates if c["name"] in selected and c["name"] in t.df.columns
        ]
        new_df = t.df.drop(columns=to_drop) if to_drop else t.df.copy()
        to_restore = [
            c for c in candidates if c["name"] not in selected and c["name"] not in new_df.columns
        ]
        for c in sorted(to_restore, key=lambda c: c["position"]):
            insert_pos = min(c["position"], len(new_df.columns))
            new_df.insert(insert_pos, c["name"], pre[c["name"]])
        t.df = new_df
        t.invalid_cols_removed = [
            {"name": c["name"], "reason": c["reason"]}
            for c in candidates
            if c["name"] in selected
        ]
        st.rerun()


def _render_fill_cols_body(t: "DetectedTable") -> None:
    """グルーピング列 ffill の詳細（Streamlit ウィジェット版）。"""
    pre = t.pre_fill_df
    post = t.post_fill_df if t.post_fill_df is not None else t.df
    cols = getattr(t, "filled_cols", [])

    badges = " ".join(
        f"<span style='background:rgba(16,185,129,0.2);color:rgba(16,185,129,1);border:1px solid rgba(16,185,129,0.4);"
        f"border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600'>{_html.escape(c)}</span>"
        for c in cols
    )
    st.markdown(
        f"<p style='margin:4px 0 10px'>空白補完した列: {badges}</p>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**補完前**（オレンジ列 = 空白補完の対象）")
        if pre is not None:
            st.markdown(
                _df_to_html(pre, max_height=340, highlight_col_names=set(cols)),
                unsafe_allow_html=True,
            )
    with col_b:
        st.markdown("**補完後**（緑列 = 補完済み）")
        if post is not None:
            st.markdown(
                _df_to_html(post, max_height=340, green_col_names=set(cols)),
                unsafe_allow_html=True,
            )


def _render_fill_cols_body_html(t: "DetectedTable") -> str:
    """グルーピング列 ffill の詳細（HTML 文字列版）。"""
    pre = t.pre_fill_df
    post = t.post_fill_df if t.post_fill_df is not None else t.df
    cols = getattr(t, "filled_cols", [])

    badges = " ".join(
        f"<span style='background:rgba(16,185,129,0.2);color:rgba(16,185,129,1);border:1px solid rgba(16,185,129,0.4);"
        f"border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600'>{_html.escape(c)}</span>"
        for c in cols
    )
    badge_html = f"<p style='margin:4px 0 10px'>空白補完した列: {badges}</p>"

    pre_html = (
        _df_to_html(pre, max_height=340, highlight_col_names=set(cols))
        if pre is not None
        else ""
    )
    post_html = (
        _df_to_html(post, max_height=340, green_col_names=set(cols))
        if post is not None
        else ""
    )

    return (
        badge_html
        + "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px'>"
        + f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>補完前（オレンジ列 = 空白補完の対象）</p>{pre_html}</div>"
        + f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>補完後（緑列 = 補完済み）</p>{post_html}</div>"
        + "</div>"
    )


def _render_external_meta_body(t: "DetectedTable") -> None:
    """ファイル外メタデータからの派生カラム生成機能の詳細（Streamlit ウィジェット版）。"""
    info = t.external_meta_info
    before = t.pre_external_meta_df
    after = t.df
    if not info or before is None or after is None:
        return

    columns = info.get("columns", [])
    reasoning = info.get("reasoning", "")
    filename = info.get("filename") or ""
    sheet_name = info.get("sheet_name") or ""

    def _badge(item: Dict) -> str:
        color = "56,189,248" if item.get("source") == "filename" else "251,191,36"
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px;display:inline-block'>"
            f"{_html.escape(str(item.get('column_name', '')))}="
            f"{_html.escape(str(item.get('value', '')))}</span>"
        )

    cols_html = " ".join(_badge(c) for c in columns) or "（なし）"

    st.markdown(
        "<div style='margin:4px 0 12px;line-height:2'>"
        f"抽出元ファイル名: <code>{_html.escape(filename)}</code><br>"
        f"抽出元シート名: <code>{_html.escape(sheet_name)}</code><br>"
        f"抽出された列: {cols_html}<br>"
        f"<span style='font-size:12px;opacity:0.75'>"
        f"<span style='color:rgba(56,189,248,1)'>■</span> ファイル名由来　"
        f"<span style='color:rgba(251,191,36,1)'>■</span> シート名由来</span><br>"
        f"理由: {_html.escape(reasoning)}"
        "</div>",
        unsafe_allow_html=True,
    )

    new_col_set = {str(c.get("column_name", "")) for c in columns}
    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown(f"**変換前**（{len(before.columns)} 列 × {len(before)} 行）")
        st.markdown(_df_to_html(before, max_height=340), unsafe_allow_html=True)
    with col_a:
        st.markdown(
            f"**変換後**（{len(after.columns)} 列 × {len(after)} 行 / 緑列 = 追加された派生列）"
        )
        st.markdown(
            _df_to_html(after, max_height=340, green_col_names=new_col_set),
            unsafe_allow_html=True,
        )


def _render_external_meta_body_html(t: "DetectedTable") -> str:
    """ファイル外メタデータからの派生カラム生成機能の詳細（HTML 文字列版）。"""
    info = t.external_meta_info
    before = t.pre_external_meta_df
    after = t.df
    if not info or before is None or after is None:
        return ""

    columns = info.get("columns", [])
    reasoning = info.get("reasoning", "")
    filename = info.get("filename") or ""
    sheet_name = info.get("sheet_name") or ""

    def _badge(item: Dict) -> str:
        color = "56,189,248" if item.get("source") == "filename" else "251,191,36"
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px;display:inline-block'>"
            f"{_html.escape(str(item.get('column_name', '')))}="
            f"{_html.escape(str(item.get('value', '')))}</span>"
        )

    cols_html = " ".join(_badge(c) for c in columns) or "（なし）"
    meta_html = (
        "<div style='margin:4px 0 12px;line-height:2'>"
        f"抽出元ファイル名: <code>{_html.escape(filename)}</code><br>"
        f"抽出元シート名: <code>{_html.escape(sheet_name)}</code><br>"
        f"抽出された列: {cols_html}<br>"
        f"理由: {_html.escape(reasoning)}"
        "</div>"
    )

    new_col_set = {str(c.get("column_name", "")) for c in columns}
    pre_html = _df_to_html(before, max_height=340)
    post_html = _df_to_html(after, max_height=340, green_col_names=new_col_set)
    grid_html = (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px'>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換前（{len(before.columns)} 列 × {len(before)} 行）</p>{pre_html}</div>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換後（{len(after.columns)} 列 × {len(after)} 行 / 緑列 = 追加された派生列）</p>{post_html}</div>"
        "</div>"
    )
    return meta_html + grid_html


def _render_stack_body(t: "DetectedTable") -> None:
    """クロス集計→縦持ち変換の詳細（Streamlit ウィジェット版）。"""
    info = t.stack_info
    wide = t.df
    long_df = t.stacked_df
    if not info or wide is None or long_df is None:
        return

    label_cols = info.get("label_cols", [])
    time_cols = info.get("time_cols", [])
    var_name = info.get("var_name", "期間")
    value_name = info.get("value_name", "値")
    year_ctx = info.get("year_context")

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    label_html = " ".join(_badge(c, "156,163,175") for c in label_cols) or "（なし）"
    shown_time = time_cols[:6]
    rest_count = len(time_cols) - len(shown_time)
    time_html = " ".join(_badge(c, "56,189,248") for c in shown_time)
    if rest_count > 0:
        time_html += (
            f" <span style='font-size:12px;opacity:0.7'>...他 {rest_count} 列</span>"
        )

    meta_lines = [
        f"ラベル列: {label_html}",
        f"時系列カラム: {time_html}（計 {len(time_cols)} 列）",
        f"縦持ち後の列構成: ラベル列 → <b>{_html.escape(var_name)}</b> → <b>{_html.escape(value_name)}</b>",
    ]
    if year_ctx:
        meta_lines.append(
            f"年コンテキスト（タイトル/ファイル名から補完）: <b>{year_ctx}年</b>"
        )

    st.markdown(
        "<div style='margin:4px 0 12px;line-height:2'>"
        + "<br>".join(meta_lines)
        + "</div>",
        unsafe_allow_html=True,
    )

    time_col_set = set(time_cols)
    new_col_set = {var_name, value_name}
    if year_ctx and info.get("time_kind") == "month":
        new_col_set.add("年")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"**変換前**（横持ち / {len(wide.columns)} 列 / オレンジ列 = 時系列カラム）"
        )
        st.markdown(
            _df_to_html(wide, max_height=340, highlight_col_names=time_col_set),
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"**変換後**（縦持ち / {len(long_df.columns)} 列 × {len(long_df)} 行 / 緑列 = 変換で生まれた列）"
        )
        st.markdown(
            _df_to_html(long_df, max_height=340, green_col_names=new_col_set),
            unsafe_allow_html=True,
        )


def _render_stack_body_html(t: "DetectedTable") -> str:
    """クロス集計→縦持ち変換の詳細（HTML 文字列版）。"""
    info = t.stack_info
    wide = t.df
    long_df = t.stacked_df
    if not info or wide is None or long_df is None:
        return ""

    label_cols = info.get("label_cols", [])
    time_cols = info.get("time_cols", [])
    var_name = info.get("var_name", "期間")
    value_name = info.get("value_name", "値")
    year_ctx = info.get("year_context")

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    label_html = " ".join(_badge(c, "156,163,175") for c in label_cols) or "（なし）"
    shown_time = time_cols[:6]
    rest_count = len(time_cols) - len(shown_time)
    time_html = " ".join(_badge(c, "56,189,248") for c in shown_time)
    if rest_count > 0:
        time_html += (
            f" <span style='font-size:12px;opacity:0.7'>...他 {rest_count} 列</span>"
        )

    year_line = f"<br>年コンテキスト: <b>{year_ctx}年</b>" if year_ctx else ""
    meta_html = (
        f"<div style='margin:4px 0 12px;line-height:2'>"
        f"ラベル列: {label_html}<br>"
        f"時系列カラム: {time_html}（計 {len(time_cols)} 列）<br>"
        f"縦持ち後の列構成: ラベル列 → <b>{_html.escape(var_name)}</b> → <b>{_html.escape(value_name)}</b>"
        f"{year_line}</div>"
    )

    time_col_set = set(time_cols)
    new_col_set = {var_name, value_name}
    if year_ctx and info.get("time_kind") == "month":
        new_col_set.add("年")

    pre_html = _df_to_html(wide, max_height=340, highlight_col_names=time_col_set)
    post_html = _df_to_html(long_df, max_height=340, green_col_names=new_col_set)
    grid_html = (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px'>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換前（横持ち / {len(wide.columns)} 列 / オレンジ列 = 時系列カラム）</p>{pre_html}</div>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換後（縦持ち / {len(long_df.columns)} 列 × {len(long_df)} 行 / 緑列 = 変換で生まれた列）</p>{post_html}</div>"
        "</div>"
    )
    return meta_html + grid_html


def _render_multi_axis_body(t: "DetectedTable") -> None:
    """多段ヘッダーの検出と解決機能（軸展開。独立した複数カテゴリ軸の交差を
    縦持ちに展開）の詳細（Streamlit ウィジェット版）。"""
    info = t.multi_axis_info
    wide = t.pre_multi_axis_df
    long_df = t.df
    if not info or wide is None or long_df is None:
        return

    axis_names = info.get("axis_names", [])
    value_name = info.get("value_name", "値")
    reasoning = info.get("reasoning", "")

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    axis_html = " ".join(_badge(c, "167,139,250") for c in axis_names) or "（なし）"

    st.markdown(
        "<div style='margin:4px 0 12px;line-height:2'>"
        f"検出された軸: {axis_html}<br>"
        f"値列: {_badge(value_name, '52,211,153')}<br>"
        f"理由: {_html.escape(reasoning)}"
        "</div>",
        unsafe_allow_html=True,
    )

    new_col_set = set(axis_names) | {value_name}

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**変換前**（多段ヘッダー / {len(wide.columns)} 列 × {len(wide)} 行）")
        st.markdown(_df_to_html(wide, max_height=340), unsafe_allow_html=True)
    with col_b:
        st.markdown(
            f"**変換後**（縦持ち / {len(long_df.columns)} 列 × {len(long_df)} 行 / 緑列 = 展開で生まれた列）"
        )
        st.markdown(
            _df_to_html(long_df, max_height=340, green_col_names=new_col_set),
            unsafe_allow_html=True,
        )


def _render_multi_axis_body_html(t: "DetectedTable") -> str:
    """多段ヘッダーの検出と解決機能（軸展開）の詳細（HTML 文字列版）。"""
    info = t.multi_axis_info
    wide = t.pre_multi_axis_df
    long_df = t.df
    if not info or wide is None or long_df is None:
        return ""

    axis_names = info.get("axis_names", [])
    value_name = info.get("value_name", "値")
    reasoning = info.get("reasoning", "")

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    axis_html = " ".join(_badge(c, "167,139,250") for c in axis_names) or "（なし）"
    meta_html = (
        f"<div style='margin:4px 0 12px;line-height:2'>"
        f"検出された軸: {axis_html}<br>"
        f"値列: {_badge(value_name, '52,211,153')}<br>"
        f"理由: {_html.escape(reasoning)}"
        f"</div>"
    )

    new_col_set = set(axis_names) | {value_name}
    pre_html = _df_to_html(wide, max_height=340)
    post_html = _df_to_html(long_df, max_height=340, green_col_names=new_col_set)
    grid_html = (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px'>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換前（多段ヘッダー / {len(wide.columns)} 列 × {len(wide)} 行）</p>{pre_html}</div>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換後（縦持ち / {len(long_df.columns)} 列 × {len(long_df)} 行 / 緑列 = 展開で生まれた列）</p>{post_html}</div>"
        "</div>"
    )
    return meta_html + grid_html


def _render_wide_to_long_body(t: "DetectedTable") -> None:
    """Wide_to_long（軸×複数指標の複合列名）変換の詳細（Streamlit ウィジェット版）。"""
    info = t.wide_to_long_info
    wide = t.pre_wide_to_long_df
    long_df = t.stacked_df
    if not info or wide is None or long_df is None:
        return

    label_cols = info.get("label_cols", [])
    axis_var_name = info.get("axis_var_name", "区分")
    indicators = info.get("indicators", [])
    parsed_cols = info.get("parsed_cols", {})

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    label_html = " ".join(_badge(c, "156,163,175") for c in label_cols) or "（なし）"
    indicator_html = " ".join(_badge(c, "251,191,36") for c in indicators) or "（なし）"
    axis_tokens = info.get("axis_tokens", [])
    shown_tokens = axis_tokens[:6]
    rest_count = len(axis_tokens) - len(shown_tokens)
    axis_html = " ".join(_badge(c, "56,189,248") for c in shown_tokens)
    if rest_count > 0:
        axis_html += (
            f" <span style='font-size:12px;opacity:0.7'>...他 {rest_count} 件</span>"
        )

    meta_lines = [
        f"ラベル列: {label_html}",
        f"検出された指標: {indicator_html}（計 {len(indicators)} 種類）",
        f"軸トークン（{_html.escape(axis_var_name)}）: {axis_html}（計 {len(axis_tokens)} 件）",
        f"縦持ち後の列構成: ラベル列 → <b>{_html.escape(axis_var_name)}</b> → 指標列（{len(indicators)} 列に分離）",
    ]

    st.markdown(
        "<div style='margin:4px 0 12px;line-height:2'>"
        + "<br>".join(meta_lines)
        + "</div>",
        unsafe_allow_html=True,
    )

    compound_col_set = set(parsed_cols.keys())
    new_col_set = {axis_var_name} | set(indicators)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"**変換前**（横持ち / {len(wide.columns)} 列 / オレンジ列 = 軸+指標の複合列）"
        )
        st.markdown(
            _df_to_html(wide, max_height=340, highlight_col_names=compound_col_set),
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"**変換後**（縦持ち / {len(long_df.columns)} 列 × {len(long_df)} 行 / 緑列 = 変換で生まれた列）"
        )
        st.markdown(
            _df_to_html(long_df, max_height=340, green_col_names=new_col_set),
            unsafe_allow_html=True,
        )


def _render_wide_to_long_body_html(t: "DetectedTable") -> str:
    """Wide_to_long（軸×複数指標の複合列名）変換の詳細（HTML 文字列版）。"""
    info = t.wide_to_long_info
    wide = t.pre_wide_to_long_df
    long_df = t.stacked_df
    if not info or wide is None or long_df is None:
        return ""

    label_cols = info.get("label_cols", [])
    axis_var_name = info.get("axis_var_name", "区分")
    indicators = info.get("indicators", [])
    parsed_cols = info.get("parsed_cols", {})

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    label_html = " ".join(_badge(c, "156,163,175") for c in label_cols) or "（なし）"
    indicator_html = " ".join(_badge(c, "251,191,36") for c in indicators) or "（なし）"
    axis_tokens = info.get("axis_tokens", [])
    shown_tokens = axis_tokens[:6]
    rest_count = len(axis_tokens) - len(shown_tokens)
    axis_html = " ".join(_badge(c, "56,189,248") for c in shown_tokens)
    if rest_count > 0:
        axis_html += (
            f" <span style='font-size:12px;opacity:0.7'>...他 {rest_count} 件</span>"
        )

    meta_html = (
        f"<div style='margin:4px 0 12px;line-height:2'>"
        f"ラベル列: {label_html}<br>"
        f"検出された指標: {indicator_html}（計 {len(indicators)} 種類）<br>"
        f"軸トークン（{_html.escape(axis_var_name)}）: {axis_html}（計 {len(axis_tokens)} 件）<br>"
        f"縦持ち後の列構成: ラベル列 → <b>{_html.escape(axis_var_name)}</b> → 指標列（{len(indicators)} 列に分離）"
        f"</div>"
    )

    compound_col_set = set(parsed_cols.keys())
    new_col_set = {axis_var_name} | set(indicators)

    pre_html = _df_to_html(wide, max_height=340, highlight_col_names=compound_col_set)
    post_html = _df_to_html(long_df, max_height=340, green_col_names=new_col_set)
    grid_html = (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px'>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換前（横持ち / {len(wide.columns)} 列 / オレンジ列 = 軸+指標の複合列）</p>{pre_html}</div>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換後（縦持ち / {len(long_df.columns)} 列 × {len(long_df)} 行 / 緑列 = 変換で生まれた列）</p>{post_html}</div>"
        "</div>"
    )
    return meta_html + grid_html


def _render_uchi_split_body(t: "DetectedTable") -> None:
    """「うち」書きの内訳を別テーブルへ分離した詳細（Streamlit ウィジェット版）。"""
    info = t.uchi_split_info
    before = t.pre_uchi_split_df
    # 変換後は「このうち分離処理」の結果のみを示す（後続の集計行除去まで
    # 進んだ t.df を使うと、無関係な他区分の合計行除去まで「うち分離が
    # 消した」ように誤表示されるため、集計行除去の直前スナップショットを使う）。
    after = t.pre_agg_df if t.pre_agg_df is not None else t.df
    breakdown = t.uchi_breakdown_df
    if not info or before is None or after is None or breakdown is None:
        return

    label_col = info.get("label_col", "")
    parent_col_name = info.get("parent_col_name", "")
    child_col_name = info.get("child_col_name", "")
    match_count = info.get("match_count", 0)
    removed_positions = set(info.get("rows", {}).keys())

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    meta_lines = [
        f"対象列: {_badge(label_col, '156,163,175')}",
        f"検出された内訳行: {match_count} 件",
        f"分離後の構成: メインテーブル（内訳行を除去）＋ "
        f"内訳テーブル（<b>{_html.escape(parent_col_name)}</b>, <b>{_html.escape(child_col_name)}</b>）",
    ]
    st.markdown(
        "<div style='margin:4px 0 12px;line-height:2'>"
        + "<br>".join(meta_lines)
        + "</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**変換前**（{len(before)} 行 / 赤色 {len(removed_positions)} 行 = 内訳行）")
        st.markdown(
            _df_to_html(before, max_height=340, highlight_row_indices=removed_positions),
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(f"**変換後**（{len(after)} 行 / 内訳行を除去済み）")
        st.markdown(
            _df_to_html(after, max_height=340),
            unsafe_allow_html=True,
        )

    st.markdown(f"**生成された内訳テーブル**（{len(breakdown)} 行）")
    st.markdown(
        _df_to_html(breakdown, max_height=240, green_col_names={parent_col_name, child_col_name}),
        unsafe_allow_html=True,
    )


def _render_uchi_split_body_html(t: "DetectedTable") -> str:
    """「うち」書きの内訳を別テーブルへ分離した詳細（HTML 文字列版）。"""
    info = t.uchi_split_info
    before = t.pre_uchi_split_df
    # 変換後は「このうち分離処理」の結果のみを示す（理由は Streamlit 版と同じ）。
    after = t.pre_agg_df if t.pre_agg_df is not None else t.df
    breakdown = t.uchi_breakdown_df
    if not info or before is None or after is None or breakdown is None:
        return ""

    label_col = info.get("label_col", "")
    parent_col_name = info.get("parent_col_name", "")
    child_col_name = info.get("child_col_name", "")
    match_count = info.get("match_count", 0)
    removed_positions = set(info.get("rows", {}).keys())

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    meta_html = (
        "<div style='margin:4px 0 12px;line-height:2'>"
        f"対象列: {_badge(label_col, '156,163,175')}<br>"
        f"検出された内訳行: {match_count} 件<br>"
        f"分離後の構成: メインテーブル（内訳行を除去）＋ "
        f"内訳テーブル（<b>{_html.escape(parent_col_name)}</b>, <b>{_html.escape(child_col_name)}</b>）"
        "</div>"
    )

    pre_html = _df_to_html(before, max_height=340, highlight_row_indices=removed_positions)
    post_html = _df_to_html(after, max_height=340)
    grid_html = (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px'>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換前（{len(before)} 行 / 赤色 {len(removed_positions)} 行 = 内訳行）</p>{pre_html}</div>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換後（{len(after)} 行 / 内訳行を除去済み）</p>{post_html}</div>"
        "</div>"
    )
    breakdown_html = _df_to_html(
        breakdown, max_height=240, green_col_names={parent_col_name, child_col_name}
    )
    breakdown_block = (
        f"<p style='margin:12px 0 6px;font-weight:600'>生成された内訳テーブル（{len(breakdown)} 行）</p>{breakdown_html}"
    )
    return meta_html + grid_html + breakdown_block


def _render_unit_split_body(t: "DetectedTable") -> None:
    """単位混在の分離（指標マスタ生成）の詳細（Streamlit ウィジェット版）。"""
    info = t.unit_split_info
    before = t.pre_unit_split_df
    after = t.post_unit_split_df if t.post_unit_split_df is not None else t.df
    master = t.unit_master_df
    if not info or before is None or after is None or master is None:
        return

    label_col = info.get("label_col", "")
    master_col = info.get("master_col", "単位")
    mapping = info.get("mapping", {})
    match_count = info.get("match_count", 0)
    distinct_units = sorted(set(mapping.values()))

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    unit_html = " ".join(_badge(u, "167,139,250") for u in distinct_units)
    meta_lines = [
        f"対象列: {_badge(label_col, '156,163,175')}",
        f"検出された単位: {unit_html}（{len(distinct_units)} 種類 / {match_count} セル）",
        f"分離後の構成: <b>{_html.escape(label_col)}</b>（単位除去済み）＋ "
        f"指標マスタ（<b>{_html.escape(label_col)}</b>, <b>{_html.escape(master_col)}</b>）",
    ]
    st.markdown(
        "<div style='margin:4px 0 12px;line-height:2'>"
        + "<br>".join(meta_lines)
        + "</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**変換前**（{len(before)} 行 / 紫列 = 単位混在列）")
        st.markdown(
            _df_to_html(before, max_height=340, unit_col_names={label_col}),
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(f"**変換後**（{len(after)} 行 / 緑列 = 単位を除去した列）")
        st.markdown(
            _df_to_html(after, max_height=340, green_col_names={label_col}),
            unsafe_allow_html=True,
        )

    st.markdown(f"**生成された指標マスタ**（{len(master)} 行）")
    st.markdown(
        _df_to_html(master, max_height=240, green_col_names={label_col, master_col}),
        unsafe_allow_html=True,
    )


def _render_unit_split_body_html(t: "DetectedTable") -> str:
    """単位混在の分離（指標マスタ生成）の詳細（HTML 文字列版）。"""
    info = t.unit_split_info
    before = t.pre_unit_split_df
    after = t.post_unit_split_df if t.post_unit_split_df is not None else t.df
    master = t.unit_master_df
    if not info or before is None or after is None or master is None:
        return ""

    label_col = info.get("label_col", "")
    master_col = info.get("master_col", "単位")
    mapping = info.get("mapping", {})
    match_count = info.get("match_count", 0)
    distinct_units = sorted(set(mapping.values()))

    def _badge(text: str, color: str) -> str:
        return (
            f"<span style='background:rgba({color},0.15);color:rgba({color},1);"
            f"border:1px solid rgba({color},0.4);border-radius:4px;"
            f"padding:2px 8px;font-size:12px;font-weight:600;margin:2px'>"
            f"{_html.escape(text)}</span>"
        )

    unit_html = " ".join(_badge(u, "167,139,250") for u in distinct_units)
    meta_html = (
        "<div style='margin:4px 0 12px;line-height:2'>"
        f"対象列: {_badge(label_col, '156,163,175')}<br>"
        f"検出された単位: {unit_html}（{len(distinct_units)} 種類 / {match_count} セル）<br>"
        f"分離後の構成: <b>{_html.escape(label_col)}</b>（単位除去済み）＋ "
        f"指標マスタ（<b>{_html.escape(label_col)}</b>, <b>{_html.escape(master_col)}</b>）"
        "</div>"
    )

    pre_html = _df_to_html(before, max_height=340, unit_col_names={label_col})
    post_html = _df_to_html(after, max_height=340, green_col_names={label_col})
    grid_html = (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px'>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換前（{len(before)} 行 / 紫列 = 単位混在列）</p>{pre_html}</div>"
        f"<div style='min-width:0'><p style='margin:0 0 6px;font-weight:600'>変換後（{len(after)} 行 / 緑列 = 単位を除去した列）</p>{post_html}</div>"
        "</div>"
    )
    master_html = _df_to_html(
        master, max_height=240, green_col_names={label_col, master_col}
    )
    master_block = (
        f"<p style='margin:12px 0 6px;font-weight:600'>生成された指標マスタ（{len(master)} 行）</p>{master_html}"
    )
    return meta_html + grid_html + master_block


_AGG_META_PREVIEW_N = 3  # 画面表示は重量化を避けるため代表件数のみに絞る（全件はエクスポート時に出力）


def _agg_meta_details_html(t: "DetectedTable") -> str:
    """集計除去メタデータ（監査用 JSON）のプレビューを折りたたみ HTML として返す。空なら空文字列。

    件数が多いテーブルでも画面が重くならないよう、行・列それぞれ代表
    _AGG_META_PREVIEW_N 件のみを表示する（全件はエクスポート時の JSON に出力される）。

    st.expander は入れ子にできないため（呼び出し元が既に expander 内にいる）、
    Streamlit ウィジェット・HTML 文字列の両方の描画箇所で <details> ベースの
    この HTML を共通利用する。
    """
    row_meta = getattr(t, "agg_removed_row_metadata", [])
    col_meta = getattr(t, "agg_removed_col_metadata", [])
    n_row, n_col = len(row_meta), len(col_meta)
    if not n_row and not n_col:
        return ""

    preview_json = json.dumps(
        {
            "aggregate_rows_removed": row_meta[:_AGG_META_PREVIEW_N],
            "aggregate_columns_removed": col_meta[:_AGG_META_PREVIEW_N],
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    note = (
        f"<p style='font-size:0.78em;opacity:0.7;margin:0 0 0.4em'>"
        f"代表{_AGG_META_PREVIEW_N}件のプレビューです（行 {n_row} 件中 / 列 {n_col} 件中）。"
        f"全件はエクスポート時に JSON として出力されます。</p>"
    )
    body = (
        note
        + f"<pre style='white-space:pre-wrap;font-size:0.78em;overflow-x:auto;margin:0'>"
        + f"{_html.escape(preview_json)}</pre>"
    )

    n_total = n_row + n_col
    return _make_details_html(
        f"📋 メタデータストア（除去した集計行・集計列: 計{n_total}件）",
        body,
        open=False,
        level=3,
    )


def _render_agg_removal_body(t: "DetectedTable") -> None:
    """集計除去の詳細（Streamlit ウィジェット版、expander なし）。"""
    pre = t.pre_agg_df
    post = t.post_agg_df if t.post_agg_df is not None else t.df

    removed_rows = t.agg_rows_removed
    removed_cols = t.agg_cols_removed

    parts = []
    if removed_rows:
        parts.append(f"集計行 **{len(removed_rows)}** 行")
    if removed_cols:
        parts.append(f"集計列 **{len(removed_cols)}** 列")
    st.caption("、".join(parts) + " を除去しました")

    # 除去した列
    if removed_cols:
        st.markdown("**除去した集計列**")
        st.markdown(
            " &nbsp;".join(
                f"<code style='background:rgba(255,180,100,0.15);"
                f"border:1px solid rgba(255,180,100,0.4);border-radius:4px;"
                f"padding:1px 6px'>{_html.escape(c)}</code>"
                for c in removed_cols
            ),
            unsafe_allow_html=True,
        )

    if removed_rows:
        n_removed = len(removed_rows)
        st.markdown(f"**除去した集計行**（{n_removed} 件）")
        rows_html = "".join(
            "<tr>"
            + "".join(
                (
                    (
                        f"<td style='{_TD_STYLE}'>"
                        f"<span style='background:rgba(255,140,0,0.22);border:1px solid rgba(255,140,0,0.45);"
                        f"border-radius:3px;padding:1px 5px;font-weight:600'>"
                        f"{_html.escape(str(v))}</span></td>"
                    )
                    if (
                        "__trigger_col__" in row_info
                        and k == row_info["__trigger_col__"]
                    )
                    or ("__trigger_col__" not in row_info and _is_agg_label(str(v)))
                    else (f"<td style='{_TD_STYLE}'>{_html.escape(str(v))}</td>")
                )
                for k, v in row_info.items()
                if k != "__trigger_col__"
            )
            + "</tr>"
            for row_info in removed_rows
        )
        headers_html = "".join(
            f"<th style='{_TH_STYLE}'>{_html.escape(c)}</th>"
            for c in removed_rows[0].keys()
            if c != "__trigger_col__"
        )
        row_max_h = 300 if n_removed > 10 else None
        scroll_style = (
            f"overflow-x:auto;overflow-y:auto;max-height:{row_max_h}px"
            if row_max_h
            else "overflow-x:auto"
        )
        st.markdown(
            f"<div style='{scroll_style}'>"
            "<table style='border-collapse:collapse'>"
            f"<thead><tr>{headers_html}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table></div>",
            unsafe_allow_html=True,
        )

    # メタデータストア（監査用 JSON）
    # 呼び出し元が既に st.expander 内にいるため、入れ子不可の st.expander ではなく
    # <details> ベースの HTML（他の「その他の同様処理」箇所と同じ方式）で表示する。
    meta_details_html = _agg_meta_details_html(t)
    if meta_details_html:
        st.markdown(_MHD_CSS + meta_details_html, unsafe_allow_html=True)

    # before / after プレビュー（代表テーブルは全件スクロール）
    st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
    removed_positions = set(getattr(t, "agg_rows_removed_positions", []))
    n_removed_rows = len(removed_rows)
    _before_hints = []
    if n_removed_rows:
        _before_hints.append(f"赤色 {n_removed_rows} 行が除去対象")
    if removed_cols:
        _before_hints.append("オレンジ列が除去対象")
    _before_label = "全件 / " + "・".join(_before_hints) if _before_hints else "全件"
    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown(f"**除去前**（{_before_label}）")
        st.markdown(
            _df_to_html(
                pre.astype(str),
                max_height=340,
                highlight_row_indices=removed_positions,
                highlight_col_names=set(removed_cols) if removed_cols else None,
            ),
            unsafe_allow_html=True,
        )
    with col_a:
        st.markdown("**除去後**（全件）")
        st.markdown(
            _df_to_html(post.astype(str), max_height=340), unsafe_allow_html=True
        )


def _render_agg_removal_body_html(t: "DetectedTable") -> str:
    """集計除去の詳細を HTML 文字列で返す（ネスト details 用）。"""
    pre = t.pre_agg_df
    post = t.post_agg_df if t.post_agg_df is not None else t.df
    removed_rows = t.agg_rows_removed
    removed_cols = t.agg_cols_removed

    parts = []
    if removed_rows:
        parts.append(f"集計行 {len(removed_rows)} 行")
    if removed_cols:
        parts.append(f"集計列 {len(removed_cols)} 列")
    caption = "、".join(parts) + " を除去しました"

    cols_html = ""
    if removed_cols:
        badges = " ".join(
            f"<code style='background:rgba(255,180,100,0.15);"
            f"border:1px solid rgba(255,180,100,0.4);border-radius:4px;"
            f"padding:1px 5px;font-size:0.82em'>{_html.escape(c)}</code>"
            for c in removed_cols
        )
        cols_html = (
            f"<p style='margin:0.4em 0 0.2em'><b>除去した集計列:</b> {badges}</p>"
        )

    rows_html_block = ""
    if removed_rows:
        n_removed = len(removed_rows)
        rows_html = "".join(
            "<tr>"
            + "".join(
                (
                    (
                        f"<td style='{_TD_STYLE}'>"
                        f"<span style='background:rgba(255,140,0,0.22);border:1px solid rgba(255,140,0,0.45);"
                        f"border-radius:3px;padding:1px 5px;font-weight:600'>"
                        f"{_html.escape(str(v))}</span></td>"
                    )
                    if (
                        "__trigger_col__" in row_info
                        and k == row_info["__trigger_col__"]
                    )
                    or ("__trigger_col__" not in row_info and _is_agg_label(str(v)))
                    else (f"<td style='{_TD_STYLE}'>{_html.escape(str(v))}</td>")
                )
                for k, v in row_info.items()
                if k != "__trigger_col__"
            )
            + "</tr>"
            for row_info in removed_rows
        )
        headers_html = "".join(
            f"<th style='{_TH_STYLE}'>{_html.escape(c)}</th>"
            for c in removed_rows[0].keys()
            if c != "__trigger_col__"
        )
        row_scroll = (
            f"overflow-x:auto;overflow-y:auto;max-height:340px"
            if n_removed > 10
            else "overflow-x:auto"
        )
        rows_html_block = (
            f"<p style='margin:0.6em 0 0.2em'><b>除去した集計行（{n_removed} 件）:</b></p>"
            f"<div style='{row_scroll}'>"
            "<table style='border-collapse:collapse'>"
            f"<thead><tr>{headers_html}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table></div>"
        )

    meta_html = _agg_meta_details_html(t)

    PREVIEW = 10
    removed_positions = set(getattr(t, "agg_rows_removed_positions", []))
    preview_removed = {p for p in removed_positions if p < PREVIEW}
    before_tbl = _df_to_html(
        pre.head(PREVIEW).astype(str),
        highlight_row_indices=preview_removed,
        highlight_col_names=set(removed_cols) if removed_cols else None,
    )
    after_tbl = _df_to_html(post.head(PREVIEW).astype(str))
    n_removed_label = len(removed_rows)
    _hints = []
    if n_removed_label:
        _hints.append(f"赤色 {n_removed_label} 行が除去対象")
    if removed_cols:
        _hints.append("オレンジ列が除去対象")
    _before_lbl = f"先頭 {PREVIEW} 行" + (" / " + "・".join(_hints) if _hints else "")
    preview_html = (
        f"<div style='display:flex;gap:1rem;flex-wrap:wrap;margin-top:1.2rem'>"
        f"<div style='flex:1;min-width:280px'>"
        f"<p style='font-weight:600;margin:0 0 0.3rem'>除去前（{_before_lbl}）</p>{before_tbl}</div>"
        f"<div style='flex:1;min-width:280px'>"
        f"<p style='font-weight:600;margin:0 0 0.3rem'>除去後（先頭 {PREVIEW} 行）</p>{after_tbl}</div>"
        f"</div>"
    )

    return (
        f"<p style='font-size:0.82em;opacity:0.7;margin:0 0 0.4em'>{_html.escape(caption)}</p>"
        f"{cols_html}{rows_html_block}{meta_html}{preview_html}"
    )


def step_format():
    st.header("🔧 ステップ 3 : テーブル整形")

    tables: List[DetectedTable] = st.session_state.detected_tables

    if not st.session_state.get("tables_normalized"):
        with st.spinner("テーブルを整形中です..."):
            try:
                normalize_tables(tables, st.session_state.filename)
                st.session_state.tables_normalized = True
            except Exception as e:
                st.error(f"❌ テーブル整形エラー: {e}")
                return

    if st.session_state.auto_processing:
        st.session_state.step = 4
        st.rerun()

    formatted = [t for t in tables if t.raw_df is not None]
    unformatted = [t for t in tables if t.raw_df is None]
    agg_removed = [t for t in tables if t.pre_agg_df is not None]
    fill_applied = [t for t in tables if getattr(t, "filled_cols", [])]

    stacked_all = [t for t in tables if getattr(t, "stacked_df", None) is not None]
    unit_split_applied = [t for t in tables if getattr(t, "unit_split_info", None)]
    transpose_applied = [t for t in tables if getattr(t, "transpose_info", None)]
    pivot_applied = [t for t in tables if getattr(t, "pivot_info", None)]
    multi_axis_applied = [t for t in tables if getattr(t, "multi_axis_info", None)]
    wide_to_long_applied = [t for t in tables if getattr(t, "wide_to_long_info", None)]
    uchi_split_applied = [t for t in tables if getattr(t, "uchi_split_info", None)]
    invalid_col_targets = [
        t for t in tables if getattr(t, "invalid_col_candidates", None)
    ]
    external_meta_applied = [
        t for t in tables if getattr(t, "external_meta_info", None)
    ]
    nothing_done = (
        not formatted
        and not agg_removed
        and not fill_applied
        and not stacked_all
        and not unit_split_applied
        and not transpose_applied
        and not pivot_applied
        and not multi_axis_applied
        and not uchi_split_applied
        and not invalid_col_targets
        and not external_meta_applied
    )
    if nothing_done:
        st.info("全テーブルに対して整形処理はありませんでした。")
    else:
        first_section = True

        # ── 多段ヘッダーの検出と解決機能 ──────────────────────────────
        if formatted:
            if not first_section:
                st.divider()
            first_section = False
            st.subheader(
                f"🔗 多段ヘッダーの検出と解決機能（対象：{len(formatted)}テーブル）"
            )
            st.success(
                f"**{len(formatted)}** テーブルで多段ヘッダーを統合しました"
                f"（整形なし: {len(unformatted)} テーブル）"
            )
            rest = formatted[1:] if len(formatted) > 1 else None
            _render_header_merge_detail(formatted[0], rest=rest)

        # ── 多段ヘッダーの検出と解決機能（軸展開） ─────────────────────
        if multi_axis_applied:
            if not first_section:
                st.divider()
            first_section = False
            st.subheader(
                f"🧩 多段ヘッダーの検出と解決機能（軸展開）（対象：{len(multi_axis_applied)}テーブル）"
            )
            st.success(
                f"**{len(multi_axis_applied)}** テーブルで多段ヘッダーが独立した"
                f"複数カテゴリ軸の交差であることを検出し、縦持ち形式に展開しました"
            )
            rep_m = multi_axis_applied[0]
            rep_m_title = f"  🏷️ `{rep_m.title}`" if rep_m.title else ""
            with st.expander(
                f"**`{rep_m.table_id}`**{rep_m_title}  —  シート: {rep_m.sheet_name}",
                expanded=True,
            ):
                _render_multi_axis_body(rep_m)

                rest_multi_axis = multi_axis_applied[1:]
                if rest_multi_axis:
                    inner_html = ""
                    for r in rest_multi_axis:
                        r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                        lbl = (
                            f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                            f" — シート: {_html.escape(r.sheet_name)}"
                        )
                        inner_html += _make_details_html(
                            lbl, _render_multi_axis_body_html(r), open=False, level=3
                        )
                    outer_html = _MHD_CSS + _make_details_html(
                        f"その他の同様処理（{len(rest_multi_axis)} 件）",
                        inner_html,
                        open=False,
                        level=2,
                    )
                    st.markdown(outer_html, unsafe_allow_html=True)

        # ── Pivot 検出と変換機能 ─────────────────────────────────
        if pivot_applied:
            if not first_section:
                st.divider()
            first_section = False
            total_attrs = sum(
                len((t.pivot_info or {}).get("attributes", [])) for t in pivot_applied
            )
            st.subheader(
                f"🔃 Pivot 検出と変換機能（対象：{len(pivot_applied)}テーブル）"
            )
            st.success(
                f"**{len(pivot_applied)}** テーブルで行が (属性名, 値) のペアで"
                f"繰り返されている構造を検出し、属性名を列見出しに昇格しました  "
                f"（生成列: 計 {total_attrs} 種類）"
            )
            rep_p = pivot_applied[0]
            rep_p_title = f"  🏷️ `{rep_p.title}`" if rep_p.title else ""
            with st.expander(
                f"**`{rep_p.table_id}`**{rep_p_title}  —  シート: {rep_p.sheet_name}",
                expanded=True,
            ):
                _render_pivot_body(rep_p)

                rest_pivot = pivot_applied[1:]
                if rest_pivot:
                    inner_html = ""
                    for r in rest_pivot:
                        r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                        lbl = (
                            f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                            f" — シート: {_html.escape(r.sheet_name)}"
                        )
                        inner_html += _make_details_html(
                            lbl, _render_pivot_body_html(r), open=False, level=3
                        )
                    outer_html = _MHD_CSS + _make_details_html(
                        f"その他の同様処理（{len(rest_pivot)} 件）",
                        inner_html,
                        open=False,
                        level=2,
                    )
                    st.markdown(outer_html, unsafe_allow_html=True)

        # ── Transpose検出と変換機能 ─────────────────────────────
        if transpose_applied:
            if not first_section:
                st.divider()
            first_section = False
            st.subheader(
                f"🔄 Transpose検出と変換機能（対象：{len(transpose_applied)}テーブル）"
            )
            st.success(
                f"**{len(transpose_applied)}** テーブルで行列が意味的に逆転した表を検出し、"
                f"正しい向き（エンティティ＝行、属性＝列）に変換しました"
            )
            rep_t = transpose_applied[0]
            rep_t_title = f"  🏷️ `{rep_t.title}`" if rep_t.title else ""
            with st.expander(
                f"**`{rep_t.table_id}`**{rep_t_title}  —  シート: {rep_t.sheet_name}",
                expanded=True,
            ):
                _render_transpose_body(rep_t)

                rest_transpose = transpose_applied[1:]
                if rest_transpose:
                    inner_html = ""
                    for r in rest_transpose:
                        r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                        lbl = (
                            f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                            f" — シート: {_html.escape(r.sheet_name)}"
                        )
                        inner_html += _make_details_html(
                            lbl, _render_transpose_body_html(r), open=False, level=3
                        )
                    outer_html = _MHD_CSS + _make_details_html(
                        f"その他の同様処理（{len(rest_transpose)} 件）",
                        inner_html,
                        open=False,
                        level=2,
                    )
                    st.markdown(outer_html, unsafe_allow_html=True)

        # ── グルーピング列の前方補完機能 ─────────────────────────
        if fill_applied:
            if not first_section:
                st.divider()
            first_section = False
            total_filled = sum(len(getattr(t, "filled_cols", [])) for t in fill_applied)
            st.subheader(
                f"↕️ グルーピング列の前方補完機能（対象：{len(fill_applied)}テーブル）"
            )
            st.success(
                f"**{len(fill_applied)}** テーブルのグルーピング列の空白を上の値で埋めました  "
                f"（列: {total_filled} 件）"
            )
            rep_f = fill_applied[0]
            rep_f_title = f"  🏷️ `{rep_f.title}`" if rep_f.title else ""
            with st.expander(
                f"**`{rep_f.table_id}`**{rep_f_title}  —  シート: {rep_f.sheet_name}",
                expanded=True,
            ):
                _render_fill_cols_body(rep_f)

                rest_fill = fill_applied[1:]
                if rest_fill:
                    inner_html = ""
                    for r in rest_fill:
                        r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                        lbl = (
                            f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                            f" — シート: {_html.escape(r.sheet_name)}"
                        )
                        inner_html += _make_details_html(
                            lbl, _render_fill_cols_body_html(r), open=False, level=3
                        )
                    outer_html = _MHD_CSS + _make_details_html(
                        f"その他の同様処理（{len(rest_fill)} 件）",
                        inner_html,
                        open=False,
                        level=2,
                    )
                    st.markdown(outer_html, unsafe_allow_html=True)

        # ── 「うち」書き識別と別テーブル分離機能 ────────────────
        if uchi_split_applied:
            if not first_section:
                st.divider()
            first_section = False
            total_uchi_rows = sum(
                (t.uchi_split_info or {}).get("match_count", 0) for t in uchi_split_applied
            )
            st.subheader(
                f"📤 「うち」書き識別と別テーブル分離機能（対象：{len(uchi_split_applied)}テーブル）"
            )
            st.success(
                f"**{len(uchi_split_applied)}** テーブルで「うち」書きの内訳行を検出し、"
                f"内訳テーブルへ分離しました  "
                f"（内訳行: 計 {total_uchi_rows} 件）"
            )
            rep_uc = uchi_split_applied[0]
            rep_uc_title = f"  🏷️ `{rep_uc.title}`" if rep_uc.title else ""
            with st.expander(
                f"**`{rep_uc.table_id}`**{rep_uc_title}  —  シート: {rep_uc.sheet_name}",
                expanded=True,
            ):
                _render_uchi_split_body(rep_uc)

                rest_uchi = uchi_split_applied[1:]
                if rest_uchi:
                    inner_html = ""
                    for r in rest_uchi:
                        r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                        lbl = (
                            f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                            f" — シート: {_html.escape(r.sheet_name)}"
                        )
                        inner_html += _make_details_html(
                            lbl, _render_uchi_split_body_html(r), open=False, level=3
                        )
                    outer_html = _MHD_CSS + _make_details_html(
                        f"その他の同様処理（{len(rest_uchi)} 件）",
                        inner_html,
                        open=False,
                        level=2,
                    )
                    st.markdown(outer_html, unsafe_allow_html=True)

        # ── 集計行の検出・削除・メタデータ保存機能 ──────────────────────
        if agg_removed:
            if not first_section:
                st.divider()
            total_rows = sum(len(t.agg_rows_removed) for t in agg_removed)
            total_cols = sum(len(t.agg_cols_removed) for t in agg_removed)
            st.subheader(
                f"🗑️ 集計行の検出・削除・メタデータ保存機能（対象：{len(agg_removed)}テーブル）"
            )
            st.success(
                f"**{len(agg_removed)}** テーブルで集計行・集計列を除去しました  "
                f"（行: {total_rows} 件、列: {total_cols} 件）"
            )

            # 代表テーブル（Streamlit expander）
            rep = agg_removed[0]
            rep_title = f"  🏷️ `{rep.title}`" if rep.title else ""
            with st.expander(
                f"**`{rep.table_id}`**{rep_title}  —  シート: {rep.sheet_name}",
                expanded=True,
            ):
                _render_agg_removal_body(rep)

                # その他を MHD_CSS + <details> でネスト表示
                rest_agg = agg_removed[1:]
                if rest_agg:
                    inner_html = ""
                    for r in rest_agg:
                        r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                        lbl = (
                            f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                            f" — シート: {_html.escape(r.sheet_name)}"
                        )
                        inner_html += _make_details_html(
                            lbl, _render_agg_removal_body_html(r), open=False, level=3
                        )
                    outer_html = _MHD_CSS + _make_details_html(
                        f"その他の同様処理（{len(rest_agg)} 件）",
                        inner_html,
                        open=False,
                        level=2,
                    )
                    st.markdown(outer_html, unsafe_allow_html=True)

        # ── 単位混在の分離（指標マスタ生成）機能 ────────────────────────
        if unit_split_applied:
            if not first_section:
                st.divider()
            first_section = False
            total_units = sum(
                len(set((t.unit_split_info or {}).get("mapping", {}).values()))
                for t in unit_split_applied
            )
            st.subheader(
                f"🏷️ 単位混在の分離（指標マスタ生成）機能（対象：{len(unit_split_applied)}テーブル）"
            )
            st.success(
                f"**{len(unit_split_applied)}** テーブルで単位混在の指標列を検出し、指標マスタへ分離しました  "
                f"（検出単位: 計 {total_units} 種類）"
            )
            rep_u = unit_split_applied[0]
            rep_u_title = f"  🏷️ `{rep_u.title}`" if rep_u.title else ""
            with st.expander(
                f"**`{rep_u.table_id}`**{rep_u_title}  —  シート: {rep_u.sheet_name}",
                expanded=True,
            ):
                _render_unit_split_body(rep_u)

                rest_unit = unit_split_applied[1:]
                if rest_unit:
                    inner_html = ""
                    for r in rest_unit:
                        r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                        lbl = (
                            f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                            f" — シート: {_html.escape(r.sheet_name)}"
                        )
                        inner_html += _make_details_html(
                            lbl, _render_unit_split_body_html(r), open=False, level=3
                        )
                    outer_html = _MHD_CSS + _make_details_html(
                        f"その他の同様処理（{len(rest_unit)} 件）",
                        inner_html,
                        open=False,
                        level=2,
                    )
                    st.markdown(outer_html, unsafe_allow_html=True)

        # ── 無効カラムの検出と削除機能 ────────────────────────────────
        # 他の整形処理と同様に既定（全欠損列）を自動適用済みの状態を表示する。
        # 代表テーブルの expander の中に「その他の同様処理」を入れ子の
        # expander として置く（他処理と同じ配置・折りたたみ挙動）。
        # 列ごとに削除・復元を選び直せるチェックボックス／フォームが必要な
        # ため他処理のような静的HTML <details> は使えないが、st.expander の
        # 入れ子は実際には問題なく動作する（内側の expander も既定で
        # 折りたたんでおく）。
        if invalid_col_targets:
            if not first_section:
                st.divider()
            first_section = False
            total_removed = sum(len(t.invalid_cols_removed) for t in invalid_col_targets)
            st.subheader(
                f"🧹 無効カラムの検出と削除機能（対象：{len(invalid_col_targets)}テーブル）"
            )
            st.success(
                f"**{len(invalid_col_targets)}** テーブルで全欠損列・無名列を検出し、"
                f"既定で削除しました（削除列: 計 {total_removed} 件。"
                f"不要な場合は各テーブルで復元・選び直しができます）"
            )

            rep_i = invalid_col_targets[0]
            rep_i_title = f"  🏷️ `{rep_i.title}`" if rep_i.title else ""
            with st.expander(
                f"**`{rep_i.table_id}`**{rep_i_title}  —  シート: {rep_i.sheet_name}",
                expanded=True,
            ):
                _render_invalid_col_body(rep_i)

                rest_i = invalid_col_targets[1:]
                if rest_i:
                    with st.expander(
                        f"その他の同様処理（{len(rest_i)} 件）", expanded=False
                    ):
                        for idx, r in enumerate(rest_i):
                            if idx > 0:
                                st.divider()
                            r_title = f"  🏷️ `{r.title}`" if r.title else ""
                            st.markdown(
                                f"**`{r.table_id}`**{r_title}  —  シート: {r.sheet_name}"
                            )
                            _render_invalid_col_body(r)

        # ── ファイル外メタデータからの派生カラム生成機能 ──────────────────
        if external_meta_applied:
            if not first_section:
                st.divider()
            first_section = False
            total_cols = sum(
                len((t.external_meta_info or {}).get("columns", []))
                for t in external_meta_applied
            )
            st.subheader(
                f"🏷️ ファイル外メタデータからの派生カラム生成機能（対象：{len(external_meta_applied)}テーブル）"
            )
            st.success(
                f"**{len(external_meta_applied)}** テーブルでファイル名・シート名から"
                f"データ本体にない付帯情報を抽出し、派生カラムとして追加しました  "
                f"（追加列: 計 {total_cols} 件）"
            )
            rep_e = external_meta_applied[0]
            rep_e_title = f"  🏷️ `{rep_e.title}`" if rep_e.title else ""
            with st.expander(
                f"**`{rep_e.table_id}`**{rep_e_title}  —  シート: {rep_e.sheet_name}",
                expanded=True,
            ):
                _render_external_meta_body(rep_e)

                rest_e = external_meta_applied[1:]
                if rest_e:
                    inner_html = ""
                    for r in rest_e:
                        r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                        lbl = (
                            f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                            f" — シート: {_html.escape(r.sheet_name)}"
                        )
                        inner_html += _make_details_html(
                            lbl, _render_external_meta_body_html(r), open=False, level=3
                        )
                    outer_html = _MHD_CSS + _make_details_html(
                        f"その他の同様処理（{len(rest_e)} 件）",
                        inner_html,
                        open=False,
                        level=2,
                    )
                    st.markdown(outer_html, unsafe_allow_html=True)

        # ── Wide_to_long検出と変換機能 ────────────────────────
        if wide_to_long_applied:
            if not first_section:
                st.divider()
            first_section = False
            total_indicators = sum(
                len((t.wide_to_long_info or {}).get("indicators", []))
                for t in wide_to_long_applied
            )
            st.subheader(
                f"🔀 Wide_to_long検出と変換機能（対象：{len(wide_to_long_applied)}テーブル）"
            )
            st.success(
                f"**{len(wide_to_long_applied)}** テーブルで軸×複数指標の複合列名を検出し、"
                f"軸を縦持ちに変換しました  "
                f"（検出指標: 計 {total_indicators} 種類）"
            )
            rep_w = wide_to_long_applied[0]
            rep_w_title = f"  🏷️ `{rep_w.title}`" if rep_w.title else ""
            with st.expander(
                f"**`{rep_w.table_id}`**{rep_w_title}  —  シート: {rep_w.sheet_name}",
                expanded=True,
            ):
                _render_wide_to_long_body(rep_w)

                rest_wtl = wide_to_long_applied[1:]
                if rest_wtl:
                    inner_html = ""
                    for r in rest_wtl:
                        r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                        lbl = (
                            f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                            f" — シート: {_html.escape(r.sheet_name)}"
                        )
                        inner_html += _make_details_html(
                            lbl, _render_wide_to_long_body_html(r), open=False, level=3
                        )
                    outer_html = _MHD_CSS + _make_details_html(
                        f"その他の同様処理（{len(rest_wtl)} 件）",
                        inner_html,
                        open=False,
                        level=2,
                    )
                    st.markdown(outer_html, unsafe_allow_html=True)

        # ── クロス集計形式の検出と縦持ち変換機能 ──────────────────────
        # Wide_to_long で処理済みのテーブルは対象外にする（互いに排他だが、
        # stacked_df は両方が書き込む共有フィールドのため二重計上を防ぐ）。
        stacked = [
            t
            for t in tables
            if getattr(t, "stacked_df", None) is not None
            and not getattr(t, "wide_to_long_info", None)
        ]
        if stacked:
            if not first_section:
                st.divider()
            st.subheader(
                f"📐 クロス集計形式の検出と縦持ち変換機能（対象：{len(stacked)}テーブル）"
            )
            total_time_cols = sum(
                len(getattr(t, "stack_info", {}).get("time_cols", [])) for t in stacked
            )
            st.success(
                f"**{len(stacked)}** テーブルで横持ち時系列カラムを検出し、縦持ちに変換しました  "
                f"（時系列カラム: 計 {total_time_cols} 列）"
            )
            rep_s = stacked[0]
            rep_s_title = f"  🏷️ `{rep_s.title}`" if rep_s.title else ""
            with st.expander(
                f"**`{rep_s.table_id}`**{rep_s_title}  —  シート: {rep_s.sheet_name}",
                expanded=True,
            ):
                _render_stack_body(rep_s)

                rest_stack = stacked[1:]
                if rest_stack:
                    inner_html = ""
                    for r in rest_stack:
                        r_title = f" 🏷️ {_html.escape(r.title)}" if r.title else ""
                        lbl = (
                            f"<code>{_html.escape(r.table_id)}</code>{r_title}"
                            f" — シート: {_html.escape(r.sheet_name)}"
                        )
                        inner_html += _make_details_html(
                            lbl, _render_stack_body_html(r), open=False, level=3
                        )
                    outer_html = _MHD_CSS + _make_details_html(
                        f"その他の同様処理（{len(rest_stack)} 件）",
                        inner_html,
                        open=False,
                        level=2,
                    )
                    st.markdown(outer_html, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 4])
    with c1:
        st.button("← 戻る", on_click=_go_to, args=(2,))
    with c2:
        st.button(
            "次へ：テーブル関係分析を開始 →",
            type="primary",
            use_container_width=True,
            on_click=_go_to,
            args=(4,),
        )
