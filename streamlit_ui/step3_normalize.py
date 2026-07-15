import html as _html
import json
from typing import Dict, List, Optional, Set

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from streamlit_ui.shared import _go_to, _inject_splitter_js, _splitter_marker
from src.models import DetectedTable
from src.step3_normalize import UNIT_VOCAB, _is_agg_label

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
    highlight_col_names: Optional[set] = None,
    unit_col_names: Optional[set] = None,
    green_col_names: Optional[set] = None,
) -> str:
    """DataFrameをモダンなスタイルのHTMLテーブルに変換する。
    max_height を指定すると縦スクロール可能なコンテナで包む。
    highlight_row_count > 0 の場合、先頭 N 行を赤色強調表示する。
    highlight_row_indices: 赤色強調する行の位置インデックス集合。
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
    # 集計行除去の前段階（ヘッダー統合直後）と比較することで、
    # 集計行を「残留ヘッダー除去対象」として誤表示しないようにする。
    # Transpose 適用テーブルは行列数がここで変わるため、pre_agg_df/df より先に
    # pre_transpose_df（ヘッダー統合直後・Transpose適用前）を優先する。
    fmt = (
        t.pre_transpose_df
        if t.pre_transpose_df is not None
        else (t.pre_agg_df if t.pre_agg_df is not None else t.df)
    )
    n_residue = len(raw) - len(fmt)
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
        st.markdown(f"**整形前**（全件 / 赤色 {n_residue} 行が除去対象）")
        st.markdown(
            _df_to_html(raw.astype(str), max_height=340, highlight_row_count=n_residue),
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
    # 集計行除去の前段階と比較し、集計行を残留ヘッダー除去対象として誤表示しない。
    # Transpose 適用テーブルは pre_transpose_df（ヘッダー統合直後・Transpose適用前）を優先する。
    fmt = (
        t.pre_transpose_df
        if t.pre_transpose_df is not None
        else (t.pre_agg_df if t.pre_agg_df is not None else t.df)
    )
    n_residue = len(raw) - len(fmt)

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
    hl = min(n_residue, PREVIEW_ROWS)
    raw_col_set = {str(c) for c in raw.columns}
    unit_cols = {
        str(c) for c in fmt.columns if "[" in str(c) and str(c) not in raw_col_set
    }
    before_tbl = _df_to_html(raw.head(PREVIEW_ROWS).astype(str), highlight_row_count=hl)
    after_tbl = _df_to_html(
        fmt.head(PREVIEW_ROWS).astype(str), unit_col_names=unit_cols or None
    )
    _after_hint = " / 紫列 = 単位付加" if unit_cols else ""

    return (
        f"<p style='font-size:0.83em;opacity:0.65;margin:0 0 0.6rem'>"
        f"ヘッダー行を統合し、残留ヘッダー {n_residue} 行をデータから除去しました</p>"
        f"<p style='font-weight:600;margin:0 0 0.3rem'>列名の変化</p>"
        f"<div style='overflow-x:auto'>"
        f"<table style='border-collapse:collapse'>"
        f"<thead><tr>{headers_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>"
        f"<div style='display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.8rem'>"
        f"<div style='flex:1;min-width:280px'>"
        f"<p style='font-weight:600;margin:0 0 0.3rem'>整形前（赤色 {n_residue} 行が除去対象 / 先頭 {PREVIEW_ROWS} 行）</p>"
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
    after = t.df
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
    after = t.df
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
        f"<div><p style='margin:0 0 6px;font-weight:600'>変換前（{len(before.columns)} 列 × {len(before)} 行 / オレンジ列 = 本来エンティティであるべき列群）</p>{pre_html}</div>"
        f"<div><p style='margin:0 0 6px;font-weight:600'>変換後（{len(after.columns)} 列 × {len(after)} 行 / 緑列 = 変換で生まれたエンティティ軸列）</p>{post_html}</div>"
        "</div>"
    )
    return meta_html + grid_html


def _render_fill_cols_body(t: "DetectedTable") -> None:
    """グルーピング列 ffill の詳細（Streamlit ウィジェット版）。"""
    pre = t.pre_fill_df
    post = t.df
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
    post = t.df
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
        + f"<div><p style='margin:0 0 6px;font-weight:600'>補完前（オレンジ列 = 空白補完の対象）</p>{pre_html}</div>"
        + f"<div><p style='margin:0 0 6px;font-weight:600'>補完後（緑列 = 補完済み）</p>{post_html}</div>"
        + "</div>"
    )


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
        f"<div><p style='margin:0 0 6px;font-weight:600'>変換前（横持ち / {len(wide.columns)} 列 / オレンジ列 = 時系列カラム）</p>{pre_html}</div>"
        f"<div><p style='margin:0 0 6px;font-weight:600'>変換後（縦持ち / {len(long_df.columns)} 列 × {len(long_df)} 行 / 緑列 = 変換で生まれた列）</p>{post_html}</div>"
        "</div>"
    )
    return meta_html + grid_html


def _render_unit_split_body(t: "DetectedTable") -> None:
    """単位混在の分離（指標マスタ生成）の詳細（Streamlit ウィジェット版）。"""
    info = t.unit_split_info
    before = t.pre_unit_split_df
    after = t.df
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
    after = t.df
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
        f"<div><p style='margin:0 0 6px;font-weight:600'>変換前（{len(before)} 行 / 紫列 = 単位混在列）</p>{pre_html}</div>"
        f"<div><p style='margin:0 0 6px;font-weight:600'>変換後（{len(after)} 行 / 緑列 = 単位を除去した列）</p>{post_html}</div>"
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
    post = t.df

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
    post = t.df
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
    nothing_done = (
        not formatted
        and not agg_removed
        and not fill_applied
        and not stacked_all
        and not unit_split_applied
        and not transpose_applied
    )
    if nothing_done:
        st.info("全テーブルに対して整形処理はありませんでした。")
    else:
        first_section = True

        # ── ① 多段ヘッダーの検出と解決機能 ──────────────────────────────
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

        # ── ①.5 Transpose検出と変換機能 ─────────────────
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

        # ── ② グルーピング列の前方補完機能 ─────────────────
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

        # ── ③ 集計行の検出・削除・メタデータ保存機能 ──────────────────────
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

        # ── ④ 単位混在の分離（指標マスタ生成）機能 ────────────────────────
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

        # ── ⑤ クロス集計形式の検出と縦持ち変換機能 ──────────────────────────
        stacked = [t for t in tables if getattr(t, "stacked_df", None) is not None]
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
