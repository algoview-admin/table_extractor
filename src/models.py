from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import pandas as pd


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

    @property
    def row_count(self) -> int:
        return len(self.df) if self.df is not None else 0

    @property
    def col_count(self) -> int:
        return len(self.df.columns) if self.df is not None else 0

    def to_summary_dict(self, max_sample_rows: int = 3) -> Dict[str, Any]:
        if self.df is None or self.df.empty:
            return {
                "table_id": self.table_id,
                "sheet_name": self.sheet_name,
                "title": self.title,
                "row_count": 0,
                "col_count": 0,
                "columns": [],
                "sample_data": [],
            }

        sample = self.df.head(max_sample_rows).copy()
        for col in sample.columns:
            sample[col] = sample[col].astype(str)

        columns = [str(c) for c in self.df.columns]

        return {
            "table_id": self.table_id,
            "sheet_name": self.sheet_name,
            "title": self.title,
            "position": f"行{self.start_row}〜{self.end_row}, 列{self.start_col}〜{self.end_col}",
            "row_count": self.row_count,
            "col_count": self.col_count,
            "columns": columns[:20],
            "columns_truncated": len(self.df.columns) > 20,
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
