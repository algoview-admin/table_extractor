# Table Extractor

ExcelファイルからLLMを活用してテーブルを自動検出・整形・統合・エクスポートする Streamlit アプリ。

---

## 機能概要（7ステップ）

| Step | 機能 | 概要 |
|------|------|------|
| 1 | ファイル選択 | Excel / CSV アップロード、またはプロジェクト（`.tep`）復元 |
| 2 | テーブル検出 | シートを走査し、行分類ステートマシンで表領域を自動検出 |
| 3 | テーブル整形 | ブロンズレイヤー各種前処理 |
| 4 | テーブル関係分析 | LLMが各テーブルの粒度・親子関係・統合可能性を分析（統合推奨も同一呼び出しで生成） |
| 5 | 新規テーブル案生成 | 統合テーブル案・注記からの潜在テーブル案・親子差分による導出テーブル案を提示 |
| 6 | テーブル選択 | エクスポート対象テーブルを粒度別（統合・最小粒度・マスタ等）に選択 |
| 7 | エクスポート | CSV / ZIP でダウンロード（集計除去の監査メタデータがあれば `_metadata.json` も同梱） |

---

## ディレクトリ構成

```
table_extractor/
├── app.py                          # Streamlit メインアプリ（ページ設定・CSS・ステップルーティング）
├── src/
│   ├── step1_upload.py             # Step1: ファイル読込
│   ├── step2_detect.py             # Step2: テーブル検出
│   ├── step3_normalize_determ.py   # Step3: テーブル整形（決定論的処理）
│   ├── step3_normalize_llm.py      # Step3: テーブル整形（LLM処理）
│   ├── step4_analyze.py            # Step4: LLMによるテーブル関係分析
│   ├── step5_suggest.py            # Step5: 新規テーブル提案の集約
│   ├── step6_select.py             # Step6: エクスポート対象の選択・グルーピング
│   ├── step7_export.py             # Step7: CSV / ZIP エクスポート
│   ├── keywords.py                 # dictionaries/*.yaml のローダー
│   ├── models.py                   # データクラス定義
│   └── dictionaries/               # 単位・集計語・時間パターン等のYAML辞書
├── streamlit_ui/
│   ├── shared.py                    # 共通UI（ヘッダー・ステップ遷移・プロジェクト保存/復元）
│   └── step1〜7_*.py                # 各ステップの画面表示ロジック（src/ の処理結果を描画）
├── docs/                            # 各ステップ・全体の処理フロー図（HTML）
│   └── essential/                   # Step3整形処理の業務向けフローチャート
├── input_data/                      # 動作確認用サンプルExcelファイル
├── project/                         # 保存されたプロジェクト（.tep）ファイル
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
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

---

## 起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開きます。

---

## 補足

- テーブルが 50 件を超える場合、テーブル関係分析（Step4）は 10 件ずつのサブバッチに分割して並列処理されます（429 エラー時は指数バックオフでリトライ）
- プロジェクト保存（`.tep` ファイル、`project/` 配下）により、途中状態を復元して Step1 から再開できます
