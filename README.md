# Table Extractor

ExcelファイルからLLMを活用してテーブルを自動検出・統合・エクスポートする Streamlit アプリ。

---

## 機能概要（6ステップ）

| Step | 機能 | 概要 |
|------|------|------|
| 1 | ファイル選択 | Excel / CSV アップロード、またはプロジェクト復元 |
| 2 | テーブル検出 | シートを走査し罫線・データ密度から表領域を自動検出 |
| 3 | テーブル関係分析 | LLMが各テーブルの粒度・親子関係・統合可能性を分析 |
| 4 | 新規テーブル案生成 | LLMの統合推奨をユーザーが承認・却下し、統合テーブルを生成 |
| 5 | テーブル選択 | エクスポート対象テーブルを選択 |
| 6 | エクスポート | CSV / ZIP でダウンロード |

---

## ディレクトリ構成

```
table_extractor/
├── app.py                   # Streamlit メインアプリ
├── src/
│   ├── excel_parser.py      # テーブル検出ロジック
│   ├── relation_analyzer.py # テーブル関係分析・統合推奨生成
│   ├── aggregation_detector.py # 数値合計関係の事前検証
│   └── models.py            # データクラス定義
├── requirements.txt
└── env.example
```

---

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`env.example` をコピーして `.env` を作成し、API キーを設定します。

```bash
cp env.example .env
```

**OpenAI を使う場合**

```env
OPENAI_API_TYPE=openai
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-5.4
```

**Azure OpenAI を使う場合**

```env
OPENAI_API_TYPE=azure
AZURE_OPENAI_API_KEY=your-azure-openai-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2025-04-01-preview
```

---

## 起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開きます。

---

## 補足

- テーブルが 50 件を超える場合、テーブル関係分析は 10 件ずつのサブバッチに分割して並列処理されます（429 エラー対策済み）
- プロジェクト保存（`.tep` ファイル）により、途中状態を復元して再開できます。
