# 前処理機能テストスイート：設計方針と横展開ガイド

`table_extractor` の Step1（パース）〜Step4（決定論的事前検証）に実装された
検出・変換機能を、テストデータ＋pytestで検証するスイート。今後、新機能を
追加した際に同じ方法論で拡張できるよう、方針と手順を残す。

## 1. 設計方針：症状ベースではなく実装項目ベース

テストデータを作るとき、最初にありがちな失敗は「ヘッダーのずれ」「結合セル」
「不要な行」のような**データ品質の症状**を軸にテストデータを組み立てること。
これは一見網羅的に見えるが、実際には次の問題がある。

- ある症状がどの実装関数を叩くのか（あるいはどれも叩かないのか）が曖昧
- pytestで「この関数が正しく発火したか」を精密にassertしにくい
- 実装済みの機能（例: 階層圧縮カラムの展開、括弧書き注釈の分離）が
  たまたま作ったデータに含まれず、カバレッジの抜けに気づけない

**正しい進め方**: まずソースコードから「検出・変換を行う関数」を1つずつ
すべて洗い出し、関数名ベースのチェックリストを作る。その後、各関数が
発火する最小条件（正規表現・閾値・語彙）を実装コードから読み取り、
その条件を満たす最小〜中規模のテストデータを1つずつ設計する。

### チェックリストの作り方

対象ファイルごとに `grep -n "^def "` でトップレベル関数を洗い出し、
`detect_*` / `apply_*` / `stack_*` / `expand_*` のような検出・変換関数を
パイプライン順に並べる（`normalize_tables()` のdocstringが実行順の正本）。

```
grep -n "^def " src/step1_upload.py src/step2_detect.py \
  src/step3_normalize_determ.py src/step3_normalize_llm.py src/step4_analyze.py
```

## 2. データ配置の原則

- **1データ1関心事**を基本にする。無関係な機能を同じ行・同じ列に混在させない。
  混在させると「どちらの機能の発火でこの結果になったか」の切り分けが困難になる
  （本スイート開発時、「うち」書きと集計行除去を同じ行に混在させたところ、
  集計行の階層整合性検証が「うち」内訳行まで合計対象に含めてしまい、
  意図と異なる結果になった。関心事ごとにシート・テーブルを分けて解決した）。
- ただし完全に1機能1ファイルにすると数が爆発するため、**互いに干渉しない
  機能同士は同じファイル内の別シートにまとめてよい**（本スイートは17ファイル・
  合計40項目程度に収めている）。
- `DetectedTable` の多くのフィールド（`pre_*_df` / `post_*_df` / `*_detection` /
  `*_integrity_check`）は、まさにこの「機能ごとの中間状態を後から検証する」
  ために存在する。既存のこの資産を最大限使い、DataFrameの最終結果だけでなく
  スナップショットも積極的にassertする。

## 3. LLM併用機能のテスト方針

`src/step3_normalize_llm.py` の関数はすべて `_call_transpose_api` 経由で
`client.chat.completions.create(...)` を呼ぶ。テストは `tests/conftest.py` の
`llm_client` fixtureを使い、**実際のLLM API**を呼び出す
（`src.step3_normalize_llm.make_transpose_client()` をそのまま利用するため、
アプリ本体と全く同じクライアント生成ロジックで検証する）。

```python
def test_example(llm_client):
    client, model = llm_client
    result = llm.some_detect_function(..., client=client, model=model)
```

### 3.1 どのAPIを使うか設定する

`make_transpose_client()` が読む環境変数で設定する（アプリ本体の設定と共通）。
`OPENAI_API_TYPE` を明示的に設定しない場合、`.env` に実際に設定されている方
（`AZURE_OPENAI_API_KEY` があればAzure、無ければOpenAI）を自動選択する。

| 変数 | 用途 | 既定値 |
|---|---|---|
| `OPENAI_API_TYPE` | `openai` または `azure`（省略時は自動選択） | 自動選択 |
| `OPENAI_API_KEY` | OpenAI使用時のAPIキー | なし |
| `OPENAI_MODEL` | OpenAI使用時のモデル名 | `gpt-5.4` |
| `AZURE_OPENAI_API_KEY` | Azure使用時のAPIキー | なし |
| `AZURE_OPENAI_ENDPOINT` | Azureエンドポイント | なし |
| `AZURE_OPENAI_API_VERSION` | Azure APIバージョン | `2024-08-01-preview` |
| `AZURE_OPENAI_DEPLOYMENT` | Azureデプロイ名（モデル相当） | `gpt-5.4` |

```
# OpenAIを使う場合
OPENAI_API_KEY=sk-... pytest tests/test_step3_normalize_llm.py -v

# Azure OpenAIを使う場合
OPENAI_API_TYPE=azure AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=... \
  pytest tests/test_step3_normalize_llm.py -v
```

### 3.2 認証情報が無い場合の挙動

`llm_client` fixtureは、対応する環境変数（`OPENAI_API_KEY` または
`AZURE_OPENAI_API_KEY`）が設定されていない場合、そのテストを自動的に
skipする（`pytest.skip`）。認証情報を用意していない環境で `pytest` を
実行しても、この部分だけskip表示になりエラーにはならない。

### 3.3 実APIを呼ぶことに関する注意

D群のテストはLLM呼び出しのたびに実際の課金が発生する。CIや通常の開発中は
`OPENAI_API_KEY` 等を設定しない環境で実行し、これらのテストをskipさせておき、
D群の変更を検証したいときだけ認証情報を設定して実行する運用を推奨する。

実APIの応答は決定論的でないため、テストのassertは「結果がNoneでない」
「必要なキーが揃っている」「型が正しい」といった構造的な妥当性の確認に
留めている。具体的な命名文言・reasoning文言はテストコードにハードコード
しない（実行のたびに変わりうるため）。

## 4. 拡張手順（新機能を追加したとき）

1. 本ファイルの第1節と同じ要領で、追加した関数を実装項目チェックリストに
   追記する。
2. `tests/fixtures/generate_fixtures.py` に、その関数の発火条件を満たす
   最小データを1つ追加する関数（`gen_xxx_test()`）を書く。追加後は必ず
   単体で `detect_tables()` に通し、意図通りに検出されるか確認する。
3. 対応する `tests/test_stepN_*.py` にテスト関数を追加する。LLM併用なら
   `llm_client` fixture を使う（第3節参照）。
4. `pytest tests/ -v` で全体が通ることを確認する。

## 5. 実行方法

```
pip install -r requirements.txt
python tests/fixtures/generate_fixtures.py   # フィクスチャの(再)生成
pytest tests/ -v
```

`tests/fixtures/*.xlsx` / `*.csv` はGit管理下に置く（`input_data/` とは異なり
gitignore対象にしない。CI・他の開発者の環境でも同じデータが必要なため）。

## 6. カバレッジ確認

`pytest.ini` に `--cov=src --cov-report=term-missing` を設定済みのため、
`pytest` を実行するだけでカバレッジが表示される（`src/` 配下が対象。
`streamlit_ui/` はUI層でありこのスイートの対象外のため計測しない）。

### 6.1 基本の実行方法

```
pytest
```

出力の末尾に、モジュールごとのカバレッジ表（`Stmts`=総行数、`Miss`=未実行
行数、`Cover`=カバー率、`Missing`=未実行の行番号）が表示される。

```
Name                            Stmts   Miss  Cover   Missing
-------------------------------------------------------------
src/step1_upload.py               185     88    52%   32-35, 40-43, ...
src/step2_detect.py               462    111    76%   66, 101, 119-126, ...
...
TOTAL                            3611   1502    58%
```

`Missing` 列の行番号（例: `32-35`）が、テストで一度も実行されていない
コード行。該当行をエディタで開き、その分岐・関数に対するテストが
不足していないか確認する材料にする。

### 6.2 特定ファイル・特定テストだけを対象にする

```
pytest tests/test_step3_normalize_determ.py          # このファイルのテストだけ実行
pytest tests/test_step3_normalize_determ.py -k pivot  # 名前に"pivot"を含むテストだけ実行
```

`--cov=src` は `pytest.ini` の既定値のままなので、一部のテストだけ実行しても
カバレッジ表には `src/` 配下全体が引き続き表示される（＝そのテストだけを
実行した場合にどこがカバーされるかを確認できる）。

### 6.3 HTMLレポート（行単位で色分け表示）

```
pytest --cov-report=html
```

生成された `htmlcov/index.html` をブラウザで開く。

行番号を1つずつ`Missing`欄で確認するより、`htmlcov/index.html` を開いて
ファイル別に緑（カバー済み）／赤（未カバー）で色分けされたソースコードを
見る方が分かりやすい。ファイル一覧のCover%をクリックすると、その
ファイルのソースコードが行ごとに色分けされて表示される。
`htmlcov/` は`.gitignore`済みで生成物のためコミットしない。

### 6.4 特定モジュールだけを表示する

`--cov=<パス>` はコマンドラインで指定しても `pytest.ini` の `--cov=src` を
**上書きせず追加**される（両方の対象が測定され、結果は変わらない）。
1つのファイルの結果だけに絞って見たい場合は、`pytest` 実行後に
`coverage report` を `--include` 付きで別途呼ぶ（`pytest`が裏で使っている
`coverage.py` の集計データ`.coverage`をそのまま再利用するため、再実行は不要）。

```
pytest -q                                              # 通常通り実行（.coverageが生成される）
coverage report --include="src/step2_detect.py"        # その1ファイルだけの結果を表示
```

特定の機能を修正した直後に、その1ファイルのカバレッジ変化だけを
素早く確認したいときに使う。

### 6.5 カバレッジ結果をリセットしたいとき

```
coverage erase   # .coverage（前回実行分の集計データ）を削除
```

通常は毎回の`pytest`実行で自動的に上書きされるため不要だが、
複数回に分けて実行した結果を合算してしまった場合等にリセットする。

`src/step5_suggest.py` / `step6_select.py` / `step7_export.py` はスコープ外
（第1節参照）のため 0% のまま表示される。これは意図した状態であり、
バグではない。
