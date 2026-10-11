# Transformer

<p class="kw">キーワード：Self-Attention、Scaled Dot-Product Attention、Source Target Attention、Masked Attention、Multi-Head Attention、Positional Encoding、Position-wise Feed-Forward Network、Add & Norm（残差接続・層正規化）、BERT、GPT</p>

!!! abstract "要点"
    - **Attention だけ**で系列を処理するモデル。再帰を使わないので**全位置を並列に計算**でき、任意の2位置が**1ステップ**で結ばれる（長距離依存を学びやすい）。
    - 中心は $\mathrm{softmax}\!\left(\mathbf{Q}\mathbf{K}^{\top}/\sqrt{d_k}\right)\mathbf{V}$。$\sqrt{d_k}$ で割るのは、softmax の**飽和（勾配消失）**を防ぐため。
    - 語順の情報は Attention にないので**位置エンコーディング**で足す。デコーダは**マスク**で未来を隠す。
    - エンコーダだけ＝**BERT**（双方向・穴埋め）、デコーダだけ＝**GPT**（左から右・次語予測）。

---

## 全体の構造 {#architecture}

- **エンコーダ**：「Multi-Head Self-Attention → FFN」を $N$ 層（原論文は 6 層）。各サブ層に**残差接続＋層正規化**。
- **デコーダ**：「Masked Multi-Head Self-Attention → Source-Target Attention → FFN」を $N$ 層。最後に線形層とソフトマックスで次の語の確率を出す。
- 入力は、トークンの**埋め込み**（原論文では $\sqrt{d_{\mathrm{model}}}$ 倍）に**位置エンコーディング**を足したもの。埋め込み行列は、エンコーダ・デコーダ・出力側の線形層で**重みを共有**する。

| 記号 | 意味 | 原論文の値 |
|:--|:--|:--|
| $n$, $m$ | クエリの数（デコーダ側の長さ）、キー・バリューの数 | — |
| $d_{\mathrm{model}}$ | 各層の入出力の次元 | 512 |
| $d_k$, $d_v$ | キー（クエリ）・バリューの次元 | 64 |
| $h$ | ヘッドの数 | 8 |
| $d_{\mathrm{ff}}$ | FFN の中間次元 | 2048 |

---

## Scaled Dot-Product Attention {#scaled-dot-product}

$$
\begin{aligned}
&\mathbf{Q}\in\mathbb{R}^{n\times d_k},\quad \mathbf{K}\in\mathbb{R}^{m\times d_k},\quad \mathbf{V}\in\mathbb{R}^{m\times d_v} \\[2mm]
&\mathbf{S}=\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\in\mathbb{R}^{n\times m},\qquad
\mathbf{A}=\mathrm{softmax}(\mathbf{S})\ \ (\text{行ごと}),\qquad
\mathbf{Z}=\mathbf{A}\mathbf{V}\in\mathbb{R}^{n\times d_v}
\end{aligned}
$$

- **クエリ** $\mathbf{Q}$：何を探すか。**キー** $\mathbf{K}$：何があるか（照合される側）。**バリュー** $\mathbf{V}$：実際に取り出す中身。
- $S_{ij}=\mathbf{q}_i\cdot\mathbf{k}_j/\sqrt{d_k}$ が類似度。softmax で重み $A_{ij}$（各行の和が 1）にして、バリューの**重み付き平均**を出力する。
- **$\sqrt{d_k}$ で割る理由**：$\mathbf{q},\mathbf{k}$ の各成分が独立で平均 0・分散 1 なら、$\mathbf{q}\cdot\mathbf{k}$ の**分散は $d_k$**（標準偏差 $\sqrt{d_k}$）。$d_k$ が大きいと内積の絶対値が大きくなり、softmax が one-hot に近づく。飽和した softmax の勾配 $\partial A_i/\partial S_j=A_i(\delta_{ij}-A_j)$ はほぼ 0 になり、学習が進まない。$\sqrt{d_k}$ で割ると分散が 1 に戻る。

<div class="ew" data-widget="transformer-attn"></div>

!!! tip "動かしてみる"
    - 「割らない」にして $d_k$ を 64 に上げてみてください。最大の重みの平均が 1 に近づき、重みが one-hot になります（標準偏差は $\sqrt{d_k}$ 程度）。
    - 「$\sqrt{d_k}$ で割る」に戻すと、$d_k$ を変えても標準偏差が約 1 のままで、重みがなだらかなままです。
    - 「マスクあり」にすると、$j>i$ の列が $-\infty$（重み 0）になり、$q_1$ は $k_1$ だけを見ます。各行は $j\le i$ の範囲だけで合計 1 になります。

---

## 3 種類の Attention {#attention-types}

| 種類 | $\mathbf{Q}$ | $\mathbf{K},\mathbf{V}$ | 使う場所 |
|:--|:--|:--|:--|
| **Self-Attention** | 同じ系列 $\mathbf{X}$ | 同じ系列 $\mathbf{X}$ | エンコーダ、デコーダの第1サブ層 |
| **Source-Target Attention**（Cross-Attention） | デコーダ側 | エンコーダの最終出力 | デコーダの第2サブ層 |
| **Masked Attention** | 同じ系列 | 同じ系列（未来を遮蔽） | デコーダの第1サブ層、GPT |

$$
\begin{aligned}
&\text{Self}\quad &&\mathbf{Q}=\mathbf{X}\mathbf{W}^Q,\ \ \mathbf{K}=\mathbf{X}\mathbf{W}^K,\ \ \mathbf{V}=\mathbf{X}\mathbf{W}^V,\qquad \mathbf{X}\in\mathbb{R}^{n\times d_{\mathrm{model}}} \\[2mm]
&\text{Source-Target}\quad &&\mathbf{Q}=\mathbf{X}_{\mathrm{dec}}\mathbf{W}^Q,\ \ \mathbf{K}=\mathbf{X}_{\mathrm{enc}}\mathbf{W}^K,\ \ \mathbf{V}=\mathbf{X}_{\mathrm{enc}}\mathbf{W}^V,\qquad \mathbf{X}_{\mathrm{enc}}\in\mathbb{R}^{m\times d_{\mathrm{model}}}
\end{aligned}
$$

- Source-Target Attention は、[系列変換](seq2seq.md)の Attention と同じ役目（翻訳のとき原文のどこを見るか）。Self-Attention は同一系列内の語どうしの関係（照応・係り受けなど）を捉える。
- **マスク**は、softmax の前のスコアに加える。

$$
\mathbf{Z}=\mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}+\mathbf{M}\right)\mathbf{V},\qquad
M_{ij}=\begin{cases}0&(j\le i)\\-\infty&(j>i)\end{cases}
$$

- $e^{-\infty}=0$ なので、位置 $i$ は**自分より後ろ**を見られない。学習時に正解系列を丸ごと入れても、推論時（左から順に生成）と同じ条件になる（**因果性**、自己回帰）。実装では $-\infty$ の代わりに $-10^{9}$ などの大きな負数を使う。
- 対角成分は 0 のため、どの行にも見える要素が 1 つ以上あり、softmax が定義できる。
- この $\mathbf{M}$ は出力次元 $M$ とは別物（マスク行列）。
- パディング（長さをそろえるための詰め物）の位置も、同じ方法で遮蔽する。

---

## Multi-Head Attention {#multi-head}

$$
\begin{aligned}
&\mathrm{MHA}(\mathbf{Q},\mathbf{K},\mathbf{V})=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,\mathbf{W}^O \\[2mm]
&\mathrm{head}_i=\mathrm{Attention}\bigl(\mathbf{Q}\mathbf{W}_i^Q,\ \mathbf{K}\mathbf{W}_i^K,\ \mathbf{V}\mathbf{W}_i^V\bigr) \\[2mm]
&\mathbf{W}_i^Q,\mathbf{W}_i^K\in\mathbb{R}^{d_{\mathrm{model}}\times d_k},\quad
\mathbf{W}_i^V\in\mathbb{R}^{d_{\mathrm{model}}\times d_v},\quad
\mathbf{W}^O\in\mathbb{R}^{h d_v\times d_{\mathrm{model}}}
\end{aligned}
$$

- 各ヘッドが**別々の射影**で別の部分空間を見るので、異なる種類の関係（隣の語、遠い照応など）を同時に捉えられる。
- 原論文は $d_k=d_v=d_{\mathrm{model}}/h=64$（$h=8$）。1 ヘッドの次元を $1/h$ にするので、**計算量は 1 ヘッドで $d_{\mathrm{model}}$ 次元を使う場合とほぼ同じ**。
- 連結した $h d_v=d_{\mathrm{model}}$ 次元を $\mathbf{W}^O$ で $d_{\mathrm{model}}$ 次元に戻す。入出力の形が同じなので、層を積み重ねられる。

---

## 位置エンコーディング {#positional-encoding}

- Self-Attention は、入力の語順を入れ替えると出力も同じ順に入れ替わるだけ（**置換同変**）で、**語順の情報を持たない**。そこで埋め込みに位置ベクトルを**足す**。

$$
\mathrm{PE}_{(pos,\,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right),\qquad
\mathrm{PE}_{(pos,\,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)
$$

- $pos$：トークンの位置、$i=0,\dots,d_{\mathrm{model}}/2-1$：次元の組の番号。次元の組 $i$ ごとに角周波数 $\omega_i=10000^{-2i/d_{\mathrm{model}}}$ の $\sin,\cos$ が入り、**波長は $2\pi$ から約 $2\pi\cdot10000$ まで等比的に**長くなる。小さい $i$ が細かい位置、大きい $i$ が大まかな位置を表す（時計の秒針・分針・時針のようなもの）。
- 値は常に $[-1,1]$ に収まり、**学習した系列より長い入力にも式で値を出せる**（学習が不要）。
- **$\sin$ と $\cos$ を対にする理由**：オフセット $k$ だけ離れた位置は、元の位置ベクトルの**回転（線形変換）**で表せる。

$$
\begin{pmatrix}\sin\omega(pos+k)\\ \cos\omega(pos+k)\end{pmatrix}
=\begin{pmatrix}\cos\omega k&\sin\omega k\\ -\sin\omega k&\cos\omega k\end{pmatrix}
\begin{pmatrix}\sin\omega\,pos\\ \cos\omega\,pos\end{pmatrix}
$$

- 回転行列は $pos$ に依存しないので、モデルが**相対位置**を線形に取り出しやすい。
- 学習で獲得する**位置埋め込み**（BERT・GPT で採用）でも、性能はほぼ同じと原論文は報告している。

<div class="ew" data-widget="transformer-pe"></div>

!!! tip "動かしてみる"
    - $i$ を 0 にすると波長が約 6.3 で、位置が 1 つ動くだけで値が大きく変わります（細かい位置の手がかり）。
    - $i$ を 31 にすると波長は約 47000 で、128 位置の範囲ではほとんど変わりません（粗い位置の手がかり）。
    - $pos$ を動かすと、点が $\sin$（実線）と $\cos$（破線）の上を動き、式の右に値が出ます。同じ $pos$ でも $i$ ごとに違う値が入るため、$d_{\mathrm{model}}$ 次元全体で位置が区別されます。

---

## その他の部品 {#components}

### Position-wise Feed-Forward Network

$$
\mathrm{FFN}(\mathbf{x})=\max(0,\ \mathbf{x}\mathbf{W}_1+\mathbf{b}_1)\,\mathbf{W}_2+\mathbf{b}_2,\qquad \mathbf{W}_1\in\mathbb{R}^{d_{\mathrm{model}}\times d_{\mathrm{ff}}},\ \mathbf{W}_2\in\mathbb{R}^{d_{\mathrm{ff}}\times d_{\mathrm{model}}}
$$

- 各位置に**独立に**（重みは位置間で共有して）適用する 2 層の全結合。他の語との相互作用は Attention の側が担当し、FFN は各位置の特徴を非線形に変換する。$d_{\mathrm{ff}}$ は $d_{\mathrm{model}}$ の約 4 倍（2048）。BERT・GPT では ReLU の代わりに [GELU](activation.md) を使う。
- 1 層あたりの重みの数（バイアスを除く）は、Attention が $4d_{\mathrm{model}}^2$、FFN が $2d_{\mathrm{model}}d_{\mathrm{ff}}$。原論文の値では $4\cdot512^2+2\cdot512\cdot2048\approx3.1\times10^6$。

### Add & Norm（残差接続と層正規化）

$$
\text{Post-LN（原論文）：}\ \ \mathbf{y}=\mathrm{LayerNorm}\bigl(\mathbf{x}+\mathrm{Sublayer}(\mathbf{x})\bigr),\qquad
\text{Pre-LN：}\ \ \mathbf{y}=\mathbf{x}+\mathrm{Sublayer}\bigl(\mathrm{LayerNorm}(\mathbf{x})\bigr)
$$

- 残差接続で勾配が直接流れ（勾配消失の緩和）、[層正規化](normalization.md#layernorm)でバッチサイズに依存せず各トークンの分布をそろえる。
- 深く積むときは、**Pre-LN** のほうが学習が安定し、ウォームアップを短くできると報告されている。

### 学習の設定（原論文）

- 最適化は Adam（$\beta_1=0.9,\ \beta_2=0.98,\ \epsilon=10^{-9}$）。学習率は**ウォームアップ付き**で、最初の $T_w$ ステップ（4000）は線形に上げ、その後は $1/\sqrt{\text{step}}$ で下げる。

$$
\eta=d_{\mathrm{model}}^{-0.5}\cdot\min\bigl(\text{step}^{-0.5},\ \text{step}\cdot T_w^{-1.5}\bigr)
$$

- 正則化はドロップアウト（0.1。サブ層の出力と、埋め込み＋位置エンコーディングの和に適用）と、[ラベル平滑化](ensemble-hyperparam.md#label-smoothing)（0.1）。

---

## なぜ Self-Attention か {#why-self-attention}

| 層の種類 | 1層の計算量 | 逐次処理の回数 | 任意の2位置を結ぶ最長経路 |
|:--|:--|:--|:--|
| Self-Attention | $O(n^2 d)$ | $O(1)$ | $O(1)$ |
| Recurrent（RNN） | $O(n d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional（幅 $k$） | $O(k n d^2)$ | $O(1)$ | $O(\log_k n)$ |

- $n$：系列長、$d$：表現の次元。Self-Attention は、$n<d$ なら再帰層より軽く、**全位置を並列化**でき、**どの2位置も 1 ステップ**で結べる（長距離依存を学びやすい）。重み $\mathbf{A}$ を可視化すれば解釈もしやすい。
- 弱点は、スコア行列が $n\times n$ のため、**系列長の 2 乗**で計算量とメモリが増えること（長い系列では効率化した Attention が使われる。→ [軽量化・高速化](../app/efficiency.md)）。

---

## BERT と GPT {#bert-gpt}

| | BERT | GPT |
|:--|:--|:--|
| 使う部分 | Transformer の**エンコーダ** | Transformer の**デコーダ**（相手のエンコーダがないので Source-Target Attention は使わない） |
| Attention | Self-Attention（**双方向**） | Masked Attention（**左から右**だけ） |
| 事前学習 | 穴埋め（MLM）＋隣接文判定（NSP） | 次の語の予測（言語モデル） |
| 得意 | 文の理解・分類・抽出 | 文の生成 |

### BERT {#bert}

- 事前学習した**深い双方向**表現に、**出力層を足してファインチューニング**するだけで多くのタスクに使える。ELMo は前向きと後ろ向きの LSTM を別々に学習して結合する「浅い双方向」、GPT は左から右だけ。
- **入力**：先頭に `[CLS]`、文の境目に `[SEP]`。**トークン埋め込み＋セグメント埋め込み（1文目か2文目か）＋位置埋め込み**の和。
- **MLM**（Masked Language Model、単語マスク問題）：入力の 15 % のトークンを選び、元の語を当てる。選んだトークンのうち 80 % は `[MASK]` に、10 % はランダムな語に、10 % はそのままにする（ファインチューニング時に `[MASK]` が現れない食い違いを和らげる）。双方向で文脈を使うので、普通の左から右の言語モデルとは違い、「未来が見えてしまう」問題がマスクで解かれる。
- **NSP**（Next Sentence Prediction、隣接文問題）：2 文が連続しているかの 2 クラス分類。`[CLS]` の出力を使い、文どうしの関係を学ぶ。
- **サイズ**：BASE は $L=12,\ H=768,\ A=12$（約 1.1 億パラメータ）、LARGE は $L=24,\ H=1024,\ A=16$（約 3.4 億）。$L$ は層の数、$H$ は隠れ層の大きさ、$A$ はヘッドの数。
- **評価**：GLUE（複数の文理解タスクの集まり）、SQuAD（質問に対する答えの範囲を文章から抜き出す。v2.0 は「答えなし」もある）、SWAG（続く文を 4 択で選ぶ常識推論）。
- **アブレーション**：NSP を外すと質問応答や含意関係のタスクが悪化。MLM の双方向性（左から右だけにする）を外すと SQuAD などが大きく悪化。モデルを大きくすると、**小さな下流タスクでも**精度が上がる。
- ファインチューニングの代わりに、BERT の出力を固定した**特徴量**として使う方法でも有効。

### GPT {#gpt}

$$
\begin{aligned}
&L_1(\mathcal{U})=\sum_i\log P\bigl(u_i\mid u_{i-k},\dots,u_{i-1};\Theta\bigr) \\[2mm]
&\mathbf{h}_0=\mathbf{U}\mathbf{W}_e+\mathbf{W}_p,\qquad \mathbf{h}_l=\mathrm{TransformerBlock}(\mathbf{h}_{l-1}),\qquad P(u)=\mathrm{softmax}\bigl(\mathbf{h}_L\mathbf{W}_e^{\top}\bigr) \\[2mm]
&L_3(\mathcal{C})=L_2(\mathcal{C})+\lambda L_1(\mathcal{C})
\end{aligned}
$$

- $L_1$：事前学習（直前 $k$ 語から次の語を予測する対数尤度の和。最大化する）。$\mathbf{U}$：トークンの one-hot の並び、$\mathbf{W}_e$：トークン埋め込み、$\mathbf{W}_p$：**学習する**位置埋め込み（$\sin/\cos$ ではない）。出力の確率は、$\mathbf{W}_e$ を転置して再利用する（重み共有）。
- ファインチューニングでは、ラベル付きデータ $\mathcal{C}$ の分類損失 $L_2$ に、**言語モデルの損失 $L_1$ を補助として足す**（$L_3$）と、汎化が良くなり収束も速い。タスクごとに入力の並べ方（`[SEP]` 区切りなど）を変えるだけで、モデルの構造は変えない。GELU を使う。
- **GPT-2**：構造は GPT-1 とほぼ同じで、より大きなモデル・コーパスで学習し、**ファインチューニングなし**（zero-shot）でも多くのタスクが解けることを示した。
- **GPT-3**（約 1750 億パラメータ）：パラメータを更新せず、**プロンプトにタスクの説明と数個の例を並べるだけ**で解く（**in-context learning**）。例の数で Zero-shot（説明のみ）、One-shot（例 1 つ）、Few-shot（例数個）と呼ぶ。規模を大きくするほど、バッチサイズを大きく、学習率を小さくする。
- Zero-shot / One-shot は、一般の機械学習では「学習時に存在しなかったクラスを扱う」「少数の教師ありデータで学ぶ」意味。GPT では**プロンプトの例の数**を指す点が違う。

### 発展 {#variants}

- エンコーダとデコーダの両方を使う構成（T5 など）、画像を小片に区切って系列として入力する **Vision Transformer**（→ [画像認識](../app/vision.md)）、言語処理への応用（→ [自然言語処理](../app/nlp.md)）がある。

---

## 試験の着眼点

- $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top}/\sqrt{d_k})\mathbf{V}$。$\sqrt{d_k}$ は**内積の分散が $d_k$ になる**ので、softmax の飽和と勾配消失を防ぐために割る。
- $\mathbf{Q},\mathbf{K},\mathbf{V}$ の出どころで3種類を区別する。Self＝全部同じ系列、Source-Target＝$\mathbf{Q}$ はデコーダ・$\mathbf{K},\mathbf{V}$ はエンコーダ、Masked＝未来を $-\infty$ で遮蔽。
- マスクは softmax の**前**に加える（後から 0 にすると各行の和が 1 にならない）。
- Multi-Head は、ヘッドごとに別の $\mathbf{W}_i^Q,\mathbf{W}_i^K,\mathbf{W}_i^V$ を持ち、連結して $\mathbf{W}^O$（$hd_v\times d_{\mathrm{model}}$）で戻す。$d_k=d_{\mathrm{model}}/h$。
- 位置エンコーディングを足す理由は、Self-Attention が**語順を区別できない**から。$\sin,\cos$ は、値が有界・任意の長さに対応・相対位置が線形変換で表せるという利点がある。位置埋め込みを**学習**する方式（BERT・GPT）もある。
- 位置エンコーディングの式の指数は $2i/d_{\mathrm{model}}$（$i$ は次元の組の番号）。実装で添字 $i$ を 2 刻みで回すと、指数を $2\times$ 余分にかけるミスが起こりやすい。
- Position-wise FFN は**各位置に独立**。Add & Norm は「残差接続＋**層**正規化」（バッチ正規化ではない）。
- BERT：エンコーダ・双方向・MLM（15 %、80/10/10）＋NSP・`[CLS]`/`[SEP]`・トークン＋セグメント＋位置の埋め込み。GPT：デコーダ・Masked Attention・次語予測。
- Self-Attention の計算量は $O(n^2d)$、逐次処理 $O(1)$、最長経路 $O(1)$。RNN は $O(nd^2)$、$O(n)$、$O(n)$。

---

## 参考

- [Attention Is All You Need（Vaswani ほか, 2017）](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（Devlin ほか, 2018）](https://arxiv.org/abs/1810.04805)
- [Language Models are Few-Shot Learners（Brown ほか, 2020）](https://arxiv.org/abs/2005.14165)
- [Gaussian Error Linear Units (GELUs)（Hendrycks, Gimpel, 2016）](https://arxiv.org/abs/1606.08415)
- [On Layer Normalization in the Transformer Architecture（Xiong ほか, 2020）](https://arxiv.org/abs/2002.04745)
