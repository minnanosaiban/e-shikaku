# 系列変換・Attention

<p class="kw">キーワード：系列変換、エンコーダ・デコーダ、sequence-to-sequence（Seq2Seq）、教師強制、アテンション（注意）機構、Additive Attention（Bahdanau）、Multiplicative Attention（Luong）、スコア関数、ビームサーチ、CTC（Connectionist Temporal Classification）、BLEU、サブワード</p>

!!! abstract "要点"
    - **Seq2Seq** は、入力系列を**エンコーダ**でベクトルに要約し、**デコーダ**がそこから出力系列を1語ずつ生成する。入出力の長さが違ってよい。
    - 固定長ベクトル1本に押し込むと長い文で情報が足りなくなる。**Attention** は、デコーダの各ステップが**エンコーダの全状態の重み付き和**を作り、必要な箇所に注目する。
    - 生成時は確率の高い列を**ビームサーチ**で探す。位置の対応が不明な系列（音声・文字認識）には **CTC** で学習する。

---

## エンコーダ・デコーダ（Seq2Seq） {#encoder-decoder}

入力 $X=(\mathbf{x}_1,\dots,\mathbf{x}_T)$ から出力 $Y=(y_1,\dots,y_J)$ への条件付き確率を、1語ずつの積に分解する。

$$
P(Y\mid X)=\prod_{i=1}^{J}P\bigl(y_i\mid y_1,\dots,y_{i-1},\,X\bigr)
$$

$$
\begin{aligned}
&\text{エンコーダ} && \mathbf{h}_j=\mathrm{RNN}_{enc}(\mathbf{x}_j,\mathbf{h}_{j-1}),\qquad \mathbf{c}=q(\mathbf{h}_1,\dots,\mathbf{h}_T)\quad(\text{多くは }\mathbf{c}=\mathbf{h}_T) \\[2mm]
&\text{デコーダ} && \mathbf{s}_i=g\bigl(\mathbf{y}_{i-1},\,\mathbf{s}_{i-1},\,\mathbf{c}\bigr),\qquad P(y_i\mid\cdot)=\mathrm{softmax}(\mathbf{W}_o\mathbf{s}_i+\mathbf{b}_o)
\end{aligned}
$$

- $\mathbf{s}_0$ は $\mathbf{c}$ から作る（エンコーダの最終状態をそのまま渡す）。デコーダへの入力 $\mathbf{y}_{i-1}$ は直前の単語の埋め込み。最初は開始記号 `<bos>`、終了は終了記号 `<eos>` を出した時点。
- 損失は正解の対数尤度の和：$L=-\sum_{i=1}^{J}\log P(y_i^{*}\mid y^{*}_{<i},X)$。
- 用途：機械翻訳、要約、対話、質問応答、音声認識、画像キャプション（エンコーダに CNN を使う）。

| | 学習時 | 推論時 |
|:--|:--|:--|
| デコーダの入力 $y_{i-1}$ | **正解**の前の単語（**教師強制**） | 自分が**生成した**前の単語 |
| 並列性 | 正解が分かるので全時刻を並列に計算しやすい | 1語ずつ順番に生成 |

- **教師強制**（teacher forcing）：学習が速く安定するが、推論時は自分の誤りを入力として受け取る。この食い違いを**露出バイアス**といい、学習中に確率的に自分の出力を混ぜる scheduled sampling で和らげる。
- 工夫：入力系列の**逆順入力**（出力の先頭と近い入力が近くなり、学習しやすくなる）、多層 LSTM、**双方向エンコーダ**、パディング部分を損失から除くマスク。
- **限界**：文の情報を固定長の $\mathbf{c}$ 1本に詰めるので、**長い入力で性能が落ちる**。

---

## Attention {#attention}

デコーダの各ステップ $i$ で、**エンコーダの全状態** $\mathbf{h}_1,\dots,\mathbf{h}_T$ を見直し、今どこが重要かを重みで決める。固定長のボトルネックがなくなる。

$$
\begin{aligned}
&\text{スコア} && e_{ij}=\mathrm{score}(\text{クエリ},\ \mathbf{h}_j) \\[2mm]
&\text{重み} && \alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k=1}^{T}\exp(e_{ik})}\qquad(\alpha_{ij}\ge0,\ \textstyle\sum_j\alpha_{ij}=1) \\[2mm]
&\text{コンテキスト} && \mathbf{c}_i=\sum_{j=1}^{T}\alpha_{ij}\,\mathbf{h}_j
\end{aligned}
$$

- 重みは位置 $j$ について softmax で正規化される。$\mathbf{c}_i$ は $\mathbf{h}_j$ の**凸結合**で、ステップごとに変わる。
- $\alpha_{ij}$ を並べた行列は、出力の語と入力の語の**対応（アライメント）**を表す。翻訳での語順の入れ替わりが見え、解釈にも使える。
- 従来の $\mathbf{c}$（1本）の代わりに、各ステップの $\mathbf{c}_i$ をデコーダに渡す。

### Additive Attention（Bahdanau） {#bahdanau}

$$
\begin{aligned}
e_{ij}&=\mathbf{v}^{\top}\tanh\bigl(\mathbf{W}_h\mathbf{h}_j+\mathbf{W}_s\mathbf{s}_{i-1}\bigr) \\[1mm]
\mathbf{s}_i&=g(\mathbf{y}_{i-1},\mathbf{s}_{i-1},\mathbf{c}_i),\qquad P(y_i\mid\cdot)=\mathrm{softmax}\bigl(\mathbf{W}_o[\mathbf{s}_i;\mathbf{c}_i]+\mathbf{b}_o\bigr)
\end{aligned}
$$

- クエリは**直前の**デコーダ状態 $\mathbf{s}_{i-1}$。小さな1隠れ層のネットワーク（$\tanh$）でスコアを作るので「加法的」。
- 形：$\mathbf{v}\in\mathbb{R}^{d_a}$、$\mathbf{W}_h\in\mathbb{R}^{d_a\times d_h}$、$\mathbf{W}_s\in\mathbb{R}^{d_a\times d_s}$（$d_h$ と $d_s$ が違ってもよい）。
- エンコーダは**双方向 RNN**で、$\mathbf{h}_j=[\overrightarrow{\mathbf{h}}_j;\overleftarrow{\mathbf{h}}_j]$（前後の文脈を持たせる）。

### Multiplicative Attention（Luong） {#luong}

$$
\begin{aligned}
\mathbf{s}_i&=\mathrm{RNN}(\mathbf{y}_{i-1},\mathbf{s}_{i-1}) &&\text{（先に今の状態を作る）}\\[1mm]
e_{ij}&=\mathrm{score}(\mathbf{s}_i,\mathbf{h}_j) \\[1mm]
\tilde{\mathbf{s}}_i&=\tanh\bigl(\mathbf{W}_c[\mathbf{c}_i;\mathbf{s}_i]\bigr),\qquad P(y_i\mid\cdot)=\mathrm{softmax}(\mathbf{W}_o\tilde{\mathbf{s}}_i)
\end{aligned}
$$

| スコア関数 | 式 | 備考 |
|:--|:--|:--|
| dot（内積） | $\mathbf{s}_i^{\top}\mathbf{h}_j$ | パラメータなし。$d_s=d_h$ が必要 |
| general（双線形） | $\mathbf{s}_i^{\top}\mathbf{W}\mathbf{h}_j$ | $\mathbf{W}\in\mathbb{R}^{d_s\times d_h}$ |
| concat（MLP） | $\mathbf{v}^{\top}\tanh\bigl(\mathbf{W}[\mathbf{s}_i;\mathbf{h}_j]\bigr)$ | $\mathbf{W}[\mathbf{s};\mathbf{h}]=\mathbf{W}_1\mathbf{s}+\mathbf{W}_2\mathbf{h}$ なので Bahdanau と同じ形 |

| | Bahdanau | Luong |
|:--|:--|:--|
| クエリ | 直前の状態 $\mathbf{s}_{i-1}$ | 今の状態 $\mathbf{s}_i$ |
| コンテキストの使い方 | RNN の入力に入れて状態を作る | 状態を作ったあとで連結して出力へ |
| スコア | MLP（加法） | 内積・双線形（乗法）が中心。計算が軽い |

- **まとめ方（Q・K・V）**：クエリ $\mathbf{q}$ とキー $\mathbf{k}_j$ の類似度を softmax して、値 $\mathbf{v}_j$ の重み付き和を取る。ここでは $\mathbf{k}_j=\mathbf{v}_j=\mathbf{h}_j$。同じ枠組みを系列内部に使うと [Transformer](transformer.md) の自己注意になり、そこでは $\mathbf{q}^{\top}\mathbf{k}/\sqrt{d_k}$ とスケールする。
- **大域的（global）と局所的（local）**：全位置を見る方式と、注目位置の近傍だけを見る方式（Luong ら）。
- 計算量は入力長×出力長 $O(TJ)$ に比例して増える。
- 画像キャプションでは、画像の各領域に対する Attention で「どこを見て語を出したか」が分かる（hard／soft attention）。

<div class="ew" data-widget="seq2seq-attention"></div>

!!! tip "動かしてみる"
    - 内積では、クエリ $\mathbf{s}$ と**向きが近い** $\mathbf{h}_j$ ほどスコアが高く、重み $\alpha_j$ が大きくなります。矢印を動かすと、重みの山が移ります。
    - コンテキスト $\mathbf{c}$（紺の点）は、$\mathbf{h}_j$ を $\alpha_j$ で混ぜた点です。重みが1つに集中すると、その $\mathbf{h}_j$ に重なります。
    - $\mathbf{s}$ を原点に近づけると、スコアがどれも 0 に近く、$\alpha$ がほぼ均等になり、$\mathbf{c}$ は平均に近づきます。
    - 加法に切り替えると、スコアは固定パラメータ $\mathbf{W}_h,\mathbf{W}_s,\mathbf{v}$ を通して作られるため、内積とは違う形の重みになります（例示用に固定した値です）。

---

## 推論：ビームサーチ {#beam-search}

全系列の確率最大化は組合せ爆発（語彙 $|V|$、長さ $J$ で $|V|^J$）で不可能。1語ずつ決める近似を使う。

| 方法 | 内容 | 特徴 |
|:--|:--|:--|
| 貪欲法（greedy） | 各ステップで確率最大の1語を選ぶ | 速いが、早い段階の選択ミスを取り戻せない |
| ビームサーチ | 上位 $B$ 個の**部分列**を保持して伸ばす | $B=1$ が貪欲法。$B$ を増やすと質は上がり、コストは $B$ 倍 |
| サンプリング | 確率に従って抽出（温度・top-k・top-p） | 多様な文を生成。対話・文章生成向き |

**ビームサーチの手順**

1. 各ステップで、保持している $B$ 個の列それぞれに、すべての語を続けた $B|V|$ 個の候補を作る。
2. 累積対数確率 $\sum_i\log P(y_i\mid\cdot)$ が高い上位 $B$ 個だけを残す。
3. `<eos>` を出した列は完成として取り出し、$B$ 個の完成列がそろうか最大長で終了。最も良い列を出力する。

- 例（$B=2$）：1語目が $P(\mathrm{A})=0.5,\ P(\mathrm{B})=0.4$、2語目が $P(x\mid\mathrm{A})=0.4$、$P(x\mid\mathrm{B})=0.9$ とする。貪欲法は A→$x$ で $0.5\times0.4=0.20$。ビームは B も残すので B→$x$ の $0.4\times0.9=0.36$ を見つけられる。
- **短い列が有利になる**問題：確率は 1 以下の積なので、長い列ほど小さくなる。**長さの正規化**で補正する。

$$
\mathrm{score}(Y)=\frac{1}{|Y|^{\alpha}}\sum_{i=1}^{|Y|}\log P(y_i\mid y_{<i},X)
\qquad\text{（GNMT：}\ s(Y,X)=\frac{\log P(Y\mid X)}{lp(Y)}+cp(X;Y)\text{）}
$$

$$
lp(Y)=\frac{(5+|Y|)^{\alpha}}{(5+1)^{\alpha}},\qquad
cp(X;Y)=\beta\sum_{j=1}^{|X|}\log\Bigl(\min\Bigl(\sum_{i=1}^{|Y|}\alpha_{ij},\ 1\Bigr)\Bigr)
$$

- $lp$ は長さペナルティ、$cp$ は**カバレッジペナルティ**：入力のどの語にも注意の総量が 1 近く届くよう（訳し漏らしを減らすよう）促す。

**サンプリング**：ソフトマックスの温度 $T$ で確率の鋭さを変える。$T\to0$ で貪欲法、$T$ が大きいと一様に近づく。**top-k** は上位 $k$ 語だけ、**top-p（nucleus）**は確率の合計が $p$ に達するまでの語だけから引く。

<div class="ew" data-widget="temperature"></div>

!!! tip "動かしてみる"
    - $T$ を小さくすると最大のロジットの語に確率が集中します（貪欲法に近づく）。$T$ を大きくすると 3 語の確率が近づき、生成が多様になります。

---

## CTC（Connectionist Temporal Classification） {#ctc}

音声認識や文字認識のように、**入力フレームと出力ラベルの位置の対応が分からない**（入力が出力より長い）とき、対応を決めずに学習する。

- ラベル集合に**空白記号** $\varepsilon$ を加え、各フレームで1つ出力する（長さ $T$ の**パス** $\pi$）。
- 変換 $\mathcal{B}$：**連続した同じ記号をまとめ、そのあと空白を取り除く**。

| パス $\pi$ | $\mathcal{B}(\pi)$ |
|:--|:--|
| `a a ε a b b` | `a a b`（`a a` → `a`、`ε` で切れた次の `a` は別、`b b` → `b`） |
| `a a b` | `a b` |
| `a ε a` | `a a`（同じ文字が連続するには空白が必要） |

$$
P(Y\mid X)=\sum_{\pi\in\mathcal{B}^{-1}(Y)}\prod_{t=1}^{T}P(\pi_t\mid X),\qquad L_{CTC}=-\log P(Y\mid X)
$$

- 正解 $Y$ に変換されるすべてのパスの確率を**足し合わせる**（どの対応でもよい）。パスの数は指数的だが、**動的計画法（前向き・後ろ向きアルゴリズム）**で $O(T\,|Y|)$ で計算できる。
- 前向き変数：ラベルの前後・間に空白を入れた拡張ラベル $\mathbf{l}'$（長さ $2|Y|+1$）の位置 $s$ について、$\alpha_t(s)=\bigl(\alpha_{t-1}(s)+\alpha_{t-1}(s-1)+\alpha_{t-1}(s-2)\bigr)\,P_t(l'_s\mid X)$。ただし最後の項は $l'_s\ne\varepsilon$ かつ $l'_s\ne l'_{s-2}$ のときだけ加える（空白を飛ばして進めるのは、前後が別の文字のとき）。ここで $P_t(k\mid X)$ は時刻 $t$ に記号 $k$ を出す確率。
- 例：$T=3$、$Y=$`a` のパスは、全 $2^3=8$ 通りのうち、`εεε`（空）と `aεa`（`aa` になる）を除く **6 通り**。$Y=$`aa` なら `aεa` の **1 通り**だけ。
- 前提：各フレームの出力は入力が与えられたとき**互いに独立**と仮定する（言語モデル的な依存は持たない）。出力は入力以下の長さ。
- 推論：各フレームで最大確率の記号を選び $\mathcal{B}$ を適用する（貪欲）、またはビームサーチ（prefix search）。
- 比較：CTC は**単調な対応**が前提で並列に計算でき、Attention 型のエンコーダ・デコーダは語順の入れ替えに強い。出力側の依存も組み込む **RNN-T** もある。詳しくは [音声処理](../app/speech.md)。

---

## 評価：BLEU {#bleu}

$$
\mathrm{BLEU}=\mathrm{BP}\cdot\exp\Bigl(\sum_{n=1}^{N}w_n\log p_n\Bigr),\qquad
\mathrm{BP}=\min\bigl(1,\ e^{\,1-r/c}\bigr)\qquad(N=4,\ w_n=1/N)
$$

- $p_n$：出力の $n$-gram のうち参照訳に現れる割合（**クリッピング**：同じ $n$-gram は参照訳に出る回数まで数える）。例：`the the the the` は参照 `the cat sat` に対し 1-gram 適合率 $1/4$。
- BP（簡潔性ペナルティ）：出力の長さ $c$ が参照の長さ $r$ より短いと減点（短く出して適合率を稼ぐ不正を防ぐ）。
- 言語モデルはパープレキシティ（[RNN](rnn.md#language-model)）、要約は ROUGE（参照と出力の $n$-gram の再現率）で評価することが多い。

---

## 実用上の工夫：GNMT とサブワード {#gnmt}

- **サブワード**（BPE・WordPiece・SentencePiece）：語を頻出する部分文字列に分割して語彙を数万に抑える。未知語や活用形に強い。
- **GNMT**（Google の機械翻訳）：8層の LSTM のエンコーダ・デコーダ。
    - 最下層のエンコーダだけ**双方向**（全層を双方向にすると並列化できないため）、深い層は**残差接続**で勾配消失を防ぐ。
    - デコーダの最下層の出力をクエリにして、エンコーダの最上層に Attention をかけ、その結果を全デコーダ層へ渡す。
    - 層ごとに別の GPU に置く**モデル並列**、**ワードピース**、最大尤度のあと**強化学習**（翻訳の評価指標を報酬）で微調整 $O_{mixed}=\alpha\,O_{ML}+O_{RL}$、推論の**量子化**、上記の長さ・カバレッジ付きビームサーチ。
- 自己注意だけで系列を処理する [Transformer](transformer.md) が、RNN 型のエンコーダ・デコーダにほぼ置き換わった。

---

## 試験の着眼点

- Seq2Seq の構成：エンコーダの最終状態 → デコーダの初期状態。**学習時は教師強制、推論時は自己回帰**。
- 固定長ベクトルのボトルネックが長文で問題になり、Attention がそれを解く。
- Attention の流れ：**スコア → softmax（重み）→ 重み付き和（コンテキスト）**。重みの合計は 1。
- Bahdanau：クエリは $\mathbf{s}_{i-1}$、加法スコア $\mathbf{v}^{\top}\tanh(\mathbf{W}_h\mathbf{h}+\mathbf{W}_s\mathbf{s})$。Luong：クエリは $\mathbf{s}_i$、内積・双線形・concat。
- ビームサーチ：幅 $B$ の上位を保持。$B=1$ は貪欲法。長さの正規化がないと短い文が選ばれやすい。
- CTC：空白記号、パスの足し合わせ、$\mathcal{B}$（連続を縮約してから空白を除く）。位置の対応が不要。
- BLEU は $n$-gram 適合率の幾何平均×簡潔性ペナルティ。

---

## 参考

- [Sequence to Sequence Learning with Neural Networks（Sutskever ら, 2014）](https://arxiv.org/abs/1409.3215)
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation（Cho ら, 2014）](https://arxiv.org/abs/1406.1078)
- [Neural Machine Translation by Jointly Learning to Align and Translate（Bahdanau ら, 2014）](https://arxiv.org/abs/1409.0473)
- [Effective Approaches to Attention-based Neural Machine Translation（Luong ら, 2015）](https://arxiv.org/abs/1508.04025)
- [Google's Neural Machine Translation System（Wu ら, 2016）](https://arxiv.org/abs/1609.08144)
- [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks（Bengio ら, 2015）](https://arxiv.org/abs/1506.03099)
- [Show, Attend and Tell（Xu ら, 2015）](https://arxiv.org/abs/1502.03044)
- [Neural Machine Translation of Rare Words with Subword Units（Sennrich ら, 2015）](https://arxiv.org/abs/1508.07909)
- [SentencePiece（Kudo, Richardson, 2018）](https://arxiv.org/abs/1808.06226)
- [The Curious Case of Neural Text Degeneration（Holtzman ら, 2019）](https://arxiv.org/abs/1904.09751)（top-p サンプリング）
- [Sequence Transduction with Recurrent Neural Networks（Graves, 2012）](https://arxiv.org/abs/1211.3711)（RNN-T）
- [BLEU: a Method for Automatic Evaluation of Machine Translation（Papineni ら, 2002）](https://aclanthology.org/P02-1040/)
- Graves ら, Connectionist Temporal Classification（ICML 2006）
