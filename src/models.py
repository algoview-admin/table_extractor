from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class SheetGrid:
    """step1_upload が構築した生グリッド。step2_detect に渡す受け渡しデータ。"""
    sheet_name: str
    grid: List[List[Any]]
    max_row: int
    max_col: int


@dataclass
class DetectedTable:
    table_id: str
    sheet_name: str
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    df: Optional[pd.DataFrame] = field(default=None, repr=False)
    title: Optional[str] = None  # テーブルの直上で検出されたセクションタイトル行
    notes: List[str] = field(default_factory=list)  # テーブル末尾に続く注釈・脚注行
    raw_df: Optional[pd.DataFrame] = field(default=None, repr=False)  # 整形前 DataFrame（多段ヘッダー整形時のみ）
    pre_agg_df: Optional[pd.DataFrame] = field(default=None, repr=False)  # 集計除去前 DataFrame
    agg_rows_removed: List[dict] = field(default_factory=list)          # 除去した行のラベル値 [{col: val, ...}, ...]
    agg_cols_removed: List[str] = field(default_factory=list)           # 除去した列名
    agg_rows_removed_positions: List[int] = field(default_factory=list) # 除去した行の元 DataFrame 上の整数インデックス
    agg_removed_row_metadata: List[Dict[str, Any]] = field(default_factory=list)  # 除去行の監査用メタデータ [{key, value, context, sum_column, reported_value}, ...]
    agg_removed_col_metadata: List[Dict[str, Any]] = field(default_factory=list)  # 除去列の監査用メタデータ [{removed_column, context, reported_value}, ...]
    filled_cols: List[str] = field(default_factory=list)                # ffill を適用したグルーピング列名
    pre_fill_df: Optional[pd.DataFrame] = field(default=None, repr=False)  # ffill 前 DataFrame（ffill 適用時のみ）
    stack_info: Optional[Dict[str, Any]] = field(default=None)          # クロス集計検出情報
    stacked_df: Optional[pd.DataFrame] = field(default=None, repr=False)   # 縦持ち変換後 DataFrame
    pre_unit_split_df: Optional[pd.DataFrame] = field(default=None, repr=False)  # 単位分離前 DataFrame（単位混在検出時のみ）
    unit_split_info: Optional[Dict[str, Any]] = field(default=None)     # 単位分離情報 {label_col, master_col, mapping, match_count}
    unit_master_df: Optional[pd.DataFrame] = field(default=None, repr=False)     # 生成された指標マスタ DataFrame（指標列, 単位）
    pre_transpose_df: Optional[pd.DataFrame] = field(default=None, repr=False)  # Transpose前 DataFrame（行列逆転検出時のみ）
    transpose_info: Optional[Dict[str, Any]] = field(default=None)  # Transpose検出情報 {entity_axis_name, reasoning}
    pre_wide_to_long_df: Optional[pd.DataFrame] = field(default=None, repr=False)  # Wide_to_long前 DataFrame（複合列名検出時のみ）
    wide_to_long_info: Optional[Dict[str, Any]] = field(default=None)  # Wide_to_long検出情報 {label_cols, time_var_name, time_kind, time_tokens, indicators, parsed_cols}
    pre_uchi_split_df: Optional[pd.DataFrame] = field(default=None, repr=False)  # うち分離前 DataFrame（内訳検出時のみ）
    uchi_split_info: Optional[Dict[str, Any]] = field(default=None)  # うち分離情報 {label_col, parent_col_name, child_col_name, rows, match_count}
    uchi_breakdown_df: Optional[pd.DataFrame] = field(default=None, repr=False)  # 生成された内訳テーブル DataFrame（親子列＋値列）
    is_step3_derived: bool = False  # Step3整形処理中に新規生成されたテーブルか（Step2の検出結果表示から除外するためのフラグ）
    raw_header_rows: Optional[List[List[Any]]] = field(default=None, repr=False)  # 多段ヘッダーの検出と解決機能（Step3）が列名統合に使う生ヘッダー行（結合前、2行以上の場合のみ。df は暫定的に先頭行のみの列名）
    raw_header_roles: Optional[List[str]] = field(default=None)  # raw_header_rows の各行の役割（"name"/"unit"）。単純統合のペアリングと、unit行を軸候補と誤認しないための判定の両方に使う
    pre_multi_axis_df: Optional[pd.DataFrame] = field(default=None, repr=False)  # 多段ヘッダーの検出と解決機能（軸展開）適用前 DataFrame（展開適用時のみ）
    multi_axis_info: Optional[Dict[str, Any]] = field(default=None)  # 多段ヘッダーの検出と解決機能（軸展開）情報 {axis_names, value_name, dropped_labels, reasoning}
    multi_axis_candidates_declined: bool = False  # 軸候補は見つかったがLLMが妥当でないと判定した（またはLLM呼び出しが失敗した）か。Wide_to_long検出がTier2の閾値を緩和する判断材料に使う

    @property
    def effective_df(self) -> Optional[pd.DataFrame]:
        """分析・表示に使用する最終 DataFrame。縦持ち変換済みの場合はそちらを優先する。"""
        if self.stacked_df is not None:
            return self.stacked_df
        return self.df

    @property
    def row_count(self) -> int:
        df = self.effective_df
        return len(df) if df is not None else 0

    @property
    def col_count(self) -> int:
        df = self.effective_df
        return len(df.columns) if df is not None else 0

    def to_summary_dict(self, max_sample_rows: int = 3) -> Dict[str, Any]:
        df = self.effective_df
        if df is None or df.empty:
            return {
                "table_id": self.table_id,
                "sheet_name": self.sheet_name,
                "title": self.title,
                "row_count": 0,
                "col_count": 0,
                "columns": [],
                "sample_data": [],
            }

        sample = df.head(max_sample_rows).copy()
        for col in sample.columns:
            sample[col] = sample[col].astype(str)

        columns = [str(c) for c in df.columns]

        return {
            "table_id": self.table_id,
            "sheet_name": self.sheet_name,
            "title": self.title,
            "position": f"行{self.start_row}〜{self.end_row}, 列{self.start_col}〜{self.end_col}",
            "row_count": self.row_count,
            "col_count": self.col_count,
            "columns": columns[:20],
            "columns_truncated": len(df.columns) > 20,
            "sample_data": sample.to_dict(orient="records"),
            "notes": self.notes,
        }


@dataclass
class SheetClassification:
    sheet_name: str
    is_data_sheet: bool
    description: str


@dataclass
class TableAnalysisResult:
    table_id: str
    display_name: str
    description: str
    granularity_level: str  # "detail"（明細）, "summary"（集計）, "master"（マスタ）, "reference"（参照）, "unknown"（不明）
    is_master_table: bool
    parent_table_ids: List[str]
    child_table_ids: List[str]
    similar_table_ids: List[str]
    is_minimum_granularity_candidate: bool
    recommended_for_extraction: bool
    has_external_info: bool
    external_info_description: Optional[str]
    reasoning: str
    integration_group_id: Optional[str] = None


@dataclass
class IntegrationRecommendation:
    recommendation_id: str
    group_name: str
    description: str
    table_ids: List[str]
    new_column_name: str          # 後方互換のために保持（= new_column_names[0]）
    new_column_values: Dict[str, str]  # 後方互換のために保持（最初の軸のみ）
    reasoning: str
    parent_table_id: Optional[str] = None
    parent_label_column: Optional[str] = None
    user_decision: Optional[bool] = None
    # 多軸対応: 設定されている場合、上記の単一軸フィールドより優先される
    new_column_names: List[str] = field(default_factory=list)
    new_column_multi_values: Dict[str, List[str]] = field(default_factory=dict)
    # 軸ごとの親情報: インデックスは new_column_names のインデックスに対応
    axis_parent_table_ids: List[Optional[str]] = field(default_factory=list)
    axis_parent_label_columns: List[Optional[str]] = field(default_factory=list)


@dataclass
class MasterTableInfo:
    table_id: str
    key_column: str
    referenced_by: List[str]
    description: str


@dataclass
class DerivedLatentTable:
    """明示的には存在しないが、親の集計テーブルと検出済みの構成要素から
    数値の差し引きにより算出可能なテーブル。
    関連テーブルに付属する集計注記によって示唆される。"""

    proposal_id: str
    derived_name: str              # 欠落している構成要素の推定名
    df: pd.DataFrame               # 算出済みデータ（親 − 検出済み子の合計）
    parent_table_id: str           # 集計／親テーブルの table ID
    parent_title: str              # 集計テーブルの表示タイトル
    detected_child_ids: List[str]  # 使用した検出済み構成要素の ID
    note_text: str                 # 元の注釈テキスト
    derivation_formula: str        # 計算内容を示す人間可読な数式
    source_display_order: List[str]  # [parent_id, child1_id, child2_id, ...]
    reasoning: str
    user_decision: Optional[bool] = None  # True=含める, False=除外, None=未決定


@dataclass
class AIAnalysisResult:
    sheet_classifications: List[SheetClassification]
    table_analyses: List[TableAnalysisResult]
    integration_recommendations: List[IntegrationRecommendation]
    master_tables: List[MasterTableInfo]
    summary: str
    raw_response: Dict[str, Any]
