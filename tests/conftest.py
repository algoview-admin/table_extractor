"""前処理機能テストスイート共通フィクスチャ。

tests/fixtures/*.xlsx, *.csv を実際の読み込み経路（src.step1_upload →
src.step2_detect）に通して List[DetectedTable] を返すヘルパーを提供する。
"""

import os
import sys
from typing import Dict, List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.step1_upload import load_csv, load_excel
from src.step2_detect import detect_tables
from src.step3_normalize_llm import make_transpose_client
from src.models import DetectedTable

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def load_fixture_tables(filename: str) -> List[DetectedTable]:
    """フィクスチャファイルを実際の読み込み経路で読み込み、検出済みテーブル一覧を返す。"""
    path = os.path.join(FIXTURES_DIR, filename)
    with open(path, "rb") as f:
        content = f.read()
    if filename.lower().endswith(".csv"):
        sheet_grids, _ = load_csv(content)
    else:
        sheet_grids, _ = load_excel(content, filename)
    tables, _ = detect_tables(sheet_grids)
    return tables


def tables_by_sheet(filename: str) -> Dict[str, DetectedTable]:
    """シート名 → 最初に検出されたテーブル の辞書を返す（1シート1テーブル前提の
    フィクスチャで使う簡易ヘルパー）。"""
    result: Dict[str, DetectedTable] = {}
    for t in load_fixture_tables(filename):
        result.setdefault(t.sheet_name, t)
    return result


def _has_llm_credentials() -> bool:
    # make_transpose_client() と同じ自動選択ロジック（OPENAI_API_TYPE未設定時は
    # .envに実際に設定されている方を見る）に合わせて判定する。
    api_type = os.getenv("OPENAI_API_TYPE", "").strip().lower()
    if not api_type:
        api_type = "azure" if os.getenv("AZURE_OPENAI_API_KEY") else "openai"
    if api_type == "azure":
        return bool(os.getenv("AZURE_OPENAI_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def llm_client():
    """実際のLLM APIクライアントを返す（src.step3_normalize_llm.make_transpose_client）。

    どのAPI（OpenAI / Azure OpenAI）・どのモデルを使うかは、アプリ本体と
    全く同じ環境変数で設定する（OPENAI_API_TYPE, OPENAI_API_KEY, OPENAI_MODEL、
    または AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT）。認証情報が設定されていない場合はテストをskipする。
    """
    if not _has_llm_credentials():
        pytest.skip("LLM APIの認証情報が設定されていないためスキップ（詳細はtests/README.md参照）")
    client, model = make_transpose_client()
    return client, model
