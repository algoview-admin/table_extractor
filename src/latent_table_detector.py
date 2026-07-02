"""
テーブル外の注記・注釈からの潜在テーブル検出。

検出済みテーブルに隣接する注記・注釈は、ファイル内の他の場所に存在するが
まだ取得されていないテーブルを示唆することが多い — パースされなかったセクションにある場合や、
単一セッションの表示範囲外にある場合などが該当する。

このモジュールは、末尾の注記から「エンティティ参照」（潜在的なテーブル名）を抽出し、
検出済みテーブルリストと照合して、以下の条件を満たすすべてのケースに対して
LatentTableProposal を返す：
  - 参照エンティティの少なくとも1件が検出済みテーブルに一致する（関連性の確認）
  - 参照エンティティの少なくとも1件が一致しない（潜在テーブル）

注記タイプの対応（ルールベース、API 不要）：
  1. 集計       : "A、B、Cの合計"  →  C が欠損している可能性あり
  2. 列挙       : "以下4種: A・B・C・D"  →  D が欠損している可能性あり
  3. 除外       : "AとBを除いた値"  →  A / B が別テーブルの可能性あり
  4. 参照       : "「A」および「B」を参照"  →  A / B がテーブルの可能性あり
  5. 並列リスト : 一部の項目がテーブルに一致する、「、」区切りのリスト

上記パターンに当てはまらない複雑な自然言語の注記については、
呼び出し元が LLM に渡すことも可能（ここでは実装しない；余分なレイテンシを避けるため
純粋なルールベースを維持）。
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .models import DetectedTable, DerivedLatentTable


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LatentTableProposal:
    """注記内で参照されているが、まだ検出されていない提案テーブル。"""

    proposal_id: str
    source_table_id: str        # この提案をトリガーした末尾注記を持つテーブル
    source_title: str           # ソーステーブルの人間が読めるタイトル
    note_text: str              # 注記の完全な原文テキスト
    note_type: str              # 推定注記タイプ（aggregation / enumeration / reference / general）
    all_referenced: List[str]   # 注記から抽出されたすべてのエンティティ名
    detected_table_ids: List[str]   # 検出済みテーブルに一致したエンティティの table_id
    detected_names: List[str]       # 検出済みテーブルに一致した参照名
    missing_names: List[str]        # 検出済みテーブルに一致しなかった参照名
    reasoning: str


@dataclass
class LatentTableGroup:
    """複数シートにわたる類似した LatentTableProposal（同じ missing_names）のグループ。
    対応する DerivedLatentTable オブジェクトとまとめて管理する。"""

    group_key: str           # "|".join(sorted(missing_names)) — セッション状態のキーとして使用
    missing_names: List[str]
    detected_names: List[str]  # 代表メンバーから取得
    note_type: str
    note_text: str             # 代表メンバーから取得
    members: List              # (LatentTableProposal, Optional[DerivedLatentTable]) のリスト

    @property
    def has_derived(self) -> bool:
        return any(dlt is not None for _, dlt in self.members)


# ---------------------------------------------------------------------------
# Note-text normalisation & entity extraction
# ---------------------------------------------------------------------------

_NOTE_PREFIX_RE = re.compile(r"^[※＊\*注）注\)（注\(注意＜<]+\s*")
_SPLIT_RE = re.compile(r"[、，,・／/＋+及びおよびとや]+")

# 集計・合計の関係を示すキーワード
_AGG_KEYWORDS = ("合計", "合算", "総計", "小計", "集計", "の計", "sum")
# 分割前に除去する集計系の末尾表現
_AGG_SUFFIX_RE = re.compile(
    r"[のをにおける]*(合計|合算|総計|小計|集計|計)\s*$", re.IGNORECASE
)
# 除外の関係を示すキーワード
_EXCL_KEYWORDS = ("除く", "除いた", "除外", "を除")
# 参照・クロスリファレンスの関係を示すキーワード
_REF_KEYWORDS = ("参照", "参考", "を見る", "については", "に記載")


def _detect_note_type(text: str) -> str:
    """注記の意味タイプを分類する（表示・フィルタリング用）。"""
    if any(k in text for k in _AGG_KEYWORDS):
        return "aggregation"
    if any(k in text for k in _EXCL_KEYWORDS):
        return "exclusion"
    if any(k in text for k in _REF_KEYWORDS):
        return "reference"
    return "general"


def _extract_entities(note_text: str) -> List[str]:
    """注記から候補エンティティ名を抽出する。

    以下のパターンを処理する：
      - 「引用符付き名称」
      - （括弧内の名称）
      - 集計系の末尾を除去した後の区切り文字による列挙
    """
    text = _NOTE_PREFIX_RE.sub("", note_text).strip()

    candidates: List[str] = []

    # 1. 「日本語引用符」の抽出
    candidates.extend(re.findall(r"「([^」]{2,40})」", text))

    # 2. （括弧内）の抽出
    candidates.extend(re.findall(r"[（(]([^）)]{2,40})[）)]", text))

    # 3. 区切り文字による列挙の抽出
    #    集計の末尾を除去してから分割（例："AとBの合計" → "AとB"）
    clean = _AGG_SUFFIX_RE.sub("", text).strip()
    # 除外・参照系の末尾も除去
    clean = re.sub(r"[をにの](除く|除いた|除外|参照|参考|含む|合わせた|まとめた).*$", "", clean).strip()
    parts = _SPLIT_RE.split(clean)
    candidates.extend(p.strip() for p in parts if 2 <= len(p.strip()) <= 60)

    # 順序を保ちながら重複を除去し、短すぎるノイズトークンをフィルタリング
    _NOISE_RE = re.compile(r"^(その|この|以下|上記|なお|ただし|また|※|注|合計|合算|小計|総計|集計)$")
    seen = set()
    result = []
    for c in candidates:
        c = c.strip()
        if c and c not in seen and not _NOISE_RE.match(c) and len(c) >= 2:
            seen.add(c)
            result.append(c)

    return result


# ---------------------------------------------------------------------------
# Table matching
# ---------------------------------------------------------------------------


def _exact_match(name: str, tables: List[DetectedTable]) -> Optional[str]:
    """完全一致または部分文字列一致のみ。table_id または None を返す。"""
    for t in tables:
        title = t.title or ""
        if name == title or name == t.table_id:
            return t.table_id
    for t in tables:
        title = t.title or ""
        if title and (name in title or title in name):
            return t.table_id
    return None


def _fuzzy_match(name: str, tables: List[DetectedTable]) -> Optional[str]:
    """（事前フィルタリング済みの）テーブルリストに対してファジーマッチを行う。table_id または None を返す。

    閾値を意図的に高く設定（0.92）することで、シリーズ末尾の違いのみで
    区別される準同形名称（例："X-3" と "X合計" や "X-D" は長い共通接頭辞を
    持つが本質的に異なる）間の誤検知を防ぐ。
    """
    best_ratio = 0.0
    best_id: Optional[str] = None
    for t in tables:
        for candidate in filter(None, [t.title, t.table_id]):
            ratio = difflib.SequenceMatcher(None, name, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = t.table_id
    return best_id if best_ratio >= 0.92 else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_latent_tables(tables: List[DetectedTable]) -> List[LatentTableProposal]:
    """検出済みテーブルの末尾注記とタイトルをスキャンして潜在テーブルの提案を生成する。

    以下の条件を満たす場合に提案が生成される：注記（または集計系タイトル）が N ≥ 2 件の
    エンティティ名を含み、かつ：
      - 少なくとも1件のエンティティが検出済みテーブルに一致する（注記の関連性を確認）
      - 少なくとも1件のエンティティが検出済みテーブルに一致しない（潜在/欠損テーブル）

    2段階スキャン：
      1. 末尾注記 (t.notes)  — excel_parser 修正後のメインパス
      2. テーブルタイトル (t.title)  — 旧プロジェクトファイルでタイトルに
                                        誤分類された注記のフォールバック；
                                        タイトルが列挙・集計の形式の場合のみ実行

    マッチングは類似名シリーズエントリ（例：C-1 / C-2 / C-3）間の
    クロスコンタミネーションを避けるため2パス戦略を使用：
      パス1 — 全エンティティに対して完全一致・部分文字列一致を行い、
               すでに使用された table_id の「予約済み」セットを構築する。
      パス2 — パス1で一致しなかったエンティティのみファジーマッチを行い、
               パス1で予約されていないテーブルのみを対象とする。
    """
    proposals: List[LatentTableProposal] = []
    # 重複提案を避けるため、(source_table_id, frozenset(entities)) を追跡
    _seen: set = set()

    # 集計・列挙パターン — excel_parser._AGG_ENUM_RE と同じ正規表現
    _title_note_re = re.compile(
        r"[・、,＋+].+[のをにおける]*(合計|内訳|合算|総計|小計|集計|含む|合わせた|除く|除外)"
    )

    def _scan_text(t: DetectedTable, text: str, is_title: bool = False) -> None:
        """単一の注記・タイトルテキストをテーブル全件リストと照合して処理する。"""
        entities = _extract_entities(text)
        if len(entities) < 2:
            return

        key = (t.table_id, frozenset(entities))
        if key in _seen:
            return
        _seen.add(key)

        # ── パス1：完全一致・部分文字列一致 ──────────────────────────
        # 同一シートのテーブルを優先して照合する。複数シートに同名の
        # 同名の構成テーブルが複数シートにまたがって存在する場合、
        # 各シートの注記が他シートのテーブルを借用するのではなく、
        # 自シートの構成テーブルに一致するようにする。
        same_sheet = [t2 for t2 in tables if t2.sheet_name == t.sheet_name]
        other_sheet = [t2 for t2 in tables if t2.sheet_name != t.sheet_name]

        exact_map: dict = {}
        reserved_ids: set = {t.table_id}
        for entity in entities:
            mid = _exact_match(entity, same_sheet) or _exact_match(entity, other_sheet)
            if mid and mid != t.table_id:
                exact_map[entity] = mid
                reserved_ids.add(mid)

        # ── パス2：予約済み以外のテーブルのみファジーマッチ ────────────────────
        available = [t2 for t2 in tables if t2.table_id not in reserved_ids]

        # 完全一致済みテーブルの名称接頭辞（先頭10文字）を事前計算する。
        # 1件以上が完全一致済みの場合、同じ長い接頭辞を持つファジー候補は
        # 同シリーズの兄弟テーブルである可能性が高い（例："TypeC合計" は
        # "TypeC-1" と接頭辞を共有）。
        # このようなファジーマッチを受け入れると欠損シリーズメンバーを誤って
        # 消費してしまうため、拒否して missing に分類する。
        exact_title_prefixes: List[str] = []
        if exact_map:
            id_to_table = {tb.table_id: tb for tb in tables}
            for eid in exact_map.values():
                tb = id_to_table.get(eid)
                title = (tb.title or "") if tb else ""
                if len(title) >= 10:
                    exact_title_prefixes.append(title[:10])

        def _prefix_conflict(candidate_table: DetectedTable) -> bool:
            """候補テーブルが完全一致済みテーブルと10文字の接頭辞を共有する場合 True を返す。"""
            if not exact_title_prefixes:
                return False
            c_title = candidate_table.title or ""
            if len(c_title) < 10:
                return False
            c_pre = c_title[:10]
            return c_pre in exact_title_prefixes

        fuzzy_map: dict = {}
        for entity in entities:
            if entity not in exact_map:
                mid = _fuzzy_match(entity, available)
                if mid:
                    # 完全一致済みテーブルと長い接頭辞を共有する候補を拒否
                    # （シリーズ兄弟ガード）。
                    cand = next((tb for tb in available if tb.table_id == mid), None)
                    if cand is not None and _prefix_conflict(cand):
                        mid = None
                if mid:
                    fuzzy_map[entity] = mid

        # ── 各エンティティを分類 ────────────────────────────────────────
        detected_ids: List[str] = []
        detected_names: List[str] = []
        missing_names: List[str] = []

        for entity in entities:
            mid = exact_map.get(entity) or fuzzy_map.get(entity)
            if mid:
                detected_ids.append(mid)
                detected_names.append(entity)
            else:
                missing_names.append(entity)

        if not detected_names or not missing_names:
            return

        source_title = t.title or t.table_id
        note_type = _detect_note_type(text)
        note_short = text[:100] + ("…" if len(text) > 100 else "")

        type_label = {
            "aggregation": "集計注記",
            "exclusion":   "除外注記",
            "reference":   "参照注記",
            "general":     "注記",
        }.get(note_type, "注記")

        origin = "テーブル表題" if is_title else type_label

        reasoning = (
            f"テーブル「{source_title}」の{origin}「{note_short}」に "
            f"{len(entities)} 件の名称が列挙されています。"
            f"うち {len(detected_names)} 件（{', '.join(detected_names)}）は検出済みですが、"
            f"{len(missing_names)} 件（{', '.join(missing_names)}）は未検出です。"
            f"これらのテーブルが実際に存在する可能性があります。"
        )

        proposals.append(
            LatentTableProposal(
                proposal_id=f"LP_{len(proposals) + 1}",
                source_table_id=t.table_id,
                source_title=source_title,
                note_text=text,
                note_type=note_type,
                all_referenced=entities,
                detected_table_ids=detected_ids,
                detected_names=detected_names,
                missing_names=missing_names,
                reasoning=reasoning,
            )
        )

    for t in tables:
        # メイン：末尾注記
        for note in getattr(t, "notes", None) or []:
            _scan_text(t, note, is_title=False)

        # フォールバック：集計系タイトル（旧パーサーで注記がタイトルに誤分類されたケースを捕捉）
        title = getattr(t, "title", None) or ""
        if title and _title_note_re.search(title):
            _scan_text(t, title, is_title=True)

    return proposals


# ---------------------------------------------------------------------------
# Derived latent table generation (numeric subtraction)
# ---------------------------------------------------------------------------


def derive_latent_tables(
    tables: List[DetectedTable],
) -> List[DerivedLatentTable]:
    """
    注記によって関連付けられたテーブル群の中で、少なくとも1件の構成要素が
    欠損している各潜在テーブル提案に対して、数値の引き算によって欠損テーブルの
    データを導出しようとする：

        欠損 ≈ 集計テーブル − sum(検出済み構成要素)

    集計テーブル（親）は、絶対値の数値合計が最大の候補として識別される。
    複数の構成要素が欠損している場合、計算された残差はそれらの合算を表す。

    形状の柔軟性：全候補に共通する数値列のみを使用するため、数値以外の列や
    追加列が異なるテーブルでも参加できる。行数は一致する必要がある。

    DerivedLatentTable インスタンスのリストを返す（導出できない場合は空リスト）。
    """
    latent_proposals = find_latent_tables(tables)
    id_to_table = {t.table_id: t for t in tables}
    derived: List[DerivedLatentTable] = []

    for lp in latent_proposals:
        try:
            _try_derive_one(lp, id_to_table, derived)
        except Exception:
            pass  # 予期しないエラーが発生した提案はスキップ

    return derived


def _try_derive_one(
    lp: "LatentTableProposal",
    id_to_table: dict,
    derived: List[DerivedLatentTable],
) -> None:
    """LatentTableProposal から1件の潜在テーブルの導出を試みる。
    `derived` にインプレースで追記する；回復不能なエラーは raise する（呼び出し元でキャッチ）。"""

    # 導出には少なくとも1件の欠損構成要素が必要
    if not lp.missing_names:
        return

    # 注記タイプは問わない — 集計の文言がなくても数学的には成立し、
    # 一部の注記は異なる表現を使う（例："C-3を含む"）。

    # 関係するすべてのテーブルを収集：注記元テーブル + 検出済み参照テーブル
    candidate_ids: List[str] = [lp.source_table_id] + list(lp.detected_table_ids)
    candidate_ids_unique = list(dict.fromkeys(candidate_ids))  # 順序を保持しつつ重複除去
    candidates = [id_to_table.get(cid) for cid in candidate_ids_unique]
    if any(c is None or c.df is None or c.df.empty for c in candidates):
        return

    # 全候補に共通する数値列を検索
    common_num_cols: Optional[List] = None
    for c in candidates:
        nc = list(c.df.select_dtypes(include=[np.number]).columns)
        if not nc:
            return  # この候補に数値データがない
        if common_num_cols is None:
            common_num_cols = nc
        else:
            # 両方に存在する列のみ残す（最初の候補の順序を維持）
            common_num_cols = [col for col in common_num_cols if col in nc]

    if not common_num_cols:
        return  # 共通の数値列がない

    # 要素ごとの引き算のために行数が一致する必要がある
    row_counts = [len(c.df) for c in candidates]
    if len(set(row_counts)) != 1:
        return

    # 共通列の数値配列を抽出
    arrays: List[Tuple["DetectedTable", np.ndarray]] = []
    for c in candidates:
        arr = np.nan_to_num(
            c.df[common_num_cols].values.astype(float), nan=0.0
        )
        arrays.append((c, arr))

    # 絶対値の数値合計が最大の候補を親（PARENT）として識別
    totals = [float(np.nansum(np.abs(arr))) for _, arr in arrays]
    if max(totals) < 1e-9:
        return  # 全てゼロ — 導出するものがない

    parent_idx = int(np.argmax(totals))
    parent_table, parent_arr = arrays[parent_idx]
    children: List[Tuple["DetectedTable", np.ndarray]] = [
        (c, arr) for i, (c, arr) in enumerate(arrays) if i != parent_idx
    ]
    if not children:
        return

    # サニティチェック：親の合計は子の合計の80%以上でなければならない。
    child_sum_total = sum(float(np.nansum(np.abs(arr))) for _, arr in children)
    if child_sum_total > 0 and totals[parent_idx] < child_sum_total * 0.8:
        return  # 親が子より小さい — 親として誤ったテーブルが識別されている

    # 計算：欠損 = 親 − sum(検出済み子テーブル)
    c_sum = np.zeros_like(parent_arr, dtype=float)
    for _, arr in children:
        c_sum += arr
    derived_arr = parent_arr - c_sum

    # 導出 DataFrame を構築：親の全構造を使い、数値列を置換
    derived_df = parent_table.df.copy()
    for ci, col in enumerate(common_num_cols):
        derived_df[col] = derived_arr[:, ci]

    # 人間が読める名称と数式
    parent_label = parent_table.title or parent_table.table_id
    child_labels = [c.title or c.table_id for c, _ in children]
    missing_name = (
        lp.missing_names[0]
        if len(lp.missing_names) == 1
        else "（" + " + ".join(lp.missing_names) + "）合算"
    )
    formula = (
        f"{missing_name}  ≈  {parent_label}"
        f"  −  ( {' + '.join(child_labels)} )"
    )

    derived.append(
        DerivedLatentTable(
            proposal_id=f"DLT_{len(derived) + 1}",
            derived_name=missing_name,
            df=derived_df,
            parent_table_id=parent_table.table_id,
            parent_title=parent_label,
            detected_child_ids=[c.table_id for c, _ in children],
            note_text=lp.note_text,
            derivation_formula=formula,
            source_display_order=[parent_table.table_id] + [c.table_id for c, _ in children],
            reasoning=(
                f"注記「{lp.note_text[:100]}{'…' if len(lp.note_text) > 100 else ''}」に"
                f"よれば「{missing_name}」は「{parent_label}」の構成要素として記載されているが"
                f"未検出。検出済み構成要素（{', '.join(child_labels)}）を集計テーブルから"
                f"差し引くことで推定データを算出した。"
            ),
        )
    )


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------


def group_latent_proposals(
    proposals: List[LatentTableProposal],
    derived_latent: List[DerivedLatentTable],
) -> List[LatentTableGroup]:
    """LatentTableProposal を missing_names セットでグループ化し、対応する DLT を付与する。

    同じ frozenset の missing_names を持つ提案は「類似」とみなす
    （例：複数シートにわたって現れる同一の潜在テーブル）。
    各グループには、同じソーステーブルから計算された DerivedLatentTable が
    割り当てられる（parent_table_id == proposal.source_table_id）。
    """
    dlt_by_source: dict = {dlt.parent_table_id: dlt for dlt in derived_latent}
    groups: dict = {}
    for lp in proposals:
        key = "|".join(sorted(lp.missing_names))
        if key not in groups:
            groups[key] = LatentTableGroup(
                group_key=key,
                missing_names=list(lp.missing_names),
                detected_names=list(lp.detected_names),
                note_type=lp.note_type,
                note_text=lp.note_text,
                members=[],
            )
        groups[key].members.append((lp, dlt_by_source.get(lp.source_table_id)))
    return list(groups.values())
