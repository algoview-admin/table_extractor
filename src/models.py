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
    title: Optional[str] = None  # Section title row detected immediately above the table

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
    granularity_level: str  # "detail", "summary", "master", "reference", "unknown"
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
    new_column_name: str          # kept for backward compat (= new_column_names[0])
    new_column_values: Dict[str, str]  # kept for backward compat (first axis only)
    reasoning: str
    parent_table_id: Optional[str] = None
    parent_label_column: Optional[str] = None
    user_decision: Optional[bool] = None
    # Multi-axis support: if populated these take precedence over the single-axis fields above
    new_column_names: List[str] = field(default_factory=list)
    new_column_multi_values: Dict[str, List[str]] = field(default_factory=dict)
    # Per-axis parent info: index corresponds to new_column_names index
    axis_parent_table_ids: List[Optional[str]] = field(default_factory=list)
    axis_parent_label_columns: List[Optional[str]] = field(default_factory=list)


@dataclass
class MasterTableInfo:
    table_id: str
    key_column: str
    referenced_by: List[str]
    description: str


@dataclass
class AIAnalysisResult:
    sheet_classifications: List[SheetClassification]
    table_analyses: List[TableAnalysisResult]
    integration_recommendations: List[IntegrationRecommendation]
    master_tables: List[MasterTableInfo]
    summary: str
    raw_response: Dict[str, Any]
