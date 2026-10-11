# e-shikaku（仮）

E資格の範囲を、数式の要点と「数式・図・数値が連動して動く図」でまとめたノートです。Zensical（Material for MkDocs の後継）で作り、GitHub Pages で公開しています。MkDocs ＋ Material でもビルドできます。

## 方針

- 試験直前に見直せる**凝縮ノート**の形を保つ（文章は短く、式が中心）
- 引用していた他サイトの図は使わず、自作の図・動く図に置き換える
- 同じテーマは1ページにまとめ、記号を統一する

## 構成

```
docs/
  index.md            ホーム
  dl/                 深層学習
  css/note.css        スタイル（動く図の色も含む）
  js/katex-init.js    数式の描画（KaTeX）
  js/widgets/         動く図
    core.js           共通部品（スライダー、切り替え、数式の行と図の連動）
    gate.js           計算グラフの1ノード
    neuron.js         1つのニューロンの順伝播・逆伝播
    vanishing.js      勾配消失
note_qiita/ note_exam/ 移行元の資料。Git 管理外
```

## 動く図の追加のしかた

1. `docs/js/widgets/名前.js` で `ENote.register("名前", (el) => { ... })` を書く
2. `mkdocs.yml` の `extra_javascript` に追加する
3. ページに `<div class="ew" data-widget="名前"></div>` を置く

図（SVG）の要素に `data-k="キー"` を付け、`ENote.formulaRows` の各行に `hl: ["キー"]` を渡すと、その行にマウスを乗せたときに対応する要素が光ります。

## 動かし方

初回だけ、このフォルダで仮想環境を作ります。

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

| 何をする | 方法 |
|---|---|
| ローカルで見る | `serve.bat`（`http://127.0.0.1:8001/e-shikaku/`） |
| ビルドだけする | `build.bat` |
| **公開する** | **`deploy.bat`**（ビルド確認 → コミット → push。push すると GitHub Actions がビルドして公開します） |

公開の仕組みは `.github/workflows/pages.yml` です（GitHub の Settings → Pages → Source を「GitHub Actions」にしておきます）。MkDocs でビルドする予備の手順は、`requirements-mkdocs.txt` にあります。
