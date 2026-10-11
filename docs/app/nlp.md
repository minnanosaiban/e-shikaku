# 自然言語処理

<p class="kw">キーワード：自然言語処理のタスク、形態素解析、n-gram、Bag-of-Words、TF-IDF、サブワード（BPE・WordPiece・SentencePiece）、分散表現、Word2Vec（CBOW・skip-gram・負例サンプリング）、言語モデル、パープレキシティ、教師強制、ビームサーチ、GNMT、BERT、GPT、ゼロショット・Few-shot（in-context learning）</p>

!!! abstract "要点"
    - 文を**トークンの列**にして、次のトークンの確率（**言語モデル**）を学ぶ。道具は n-gram → RNN → **Transformer** と変わった。
    - 単語は one-hot ではなく、意味が近いほど近くなる**分散表現**（Word2Vec）で表す。さらに**文脈ごとに**変わる表現（BERT・GPT）が主流になった。
    - 大量の文で**事前学習**し、タスクごとに**ファインチューニング**する（BERT）。さらに大きくすると、**プロンプトに例を並べるだけ**で解ける（GPT-3）。

---

## タスクと系列の形 {#tasks}

| 入力 → 出力の形 | 例 |
|:--|:--|
| one to one | 1 語の品詞の分類（時間方向のない分類） |
| one to many | 画像キャプション生成（画像 1 枚 → 文） |
| many to one | 感情分析・文書分類（文 → ラベル） |
| many to many（同期） | 系列ラベリング（品詞タグ付け・固有表現抽出） |
| many to many（非同期） | 機械翻訳・要約・対話・文章生成 |

- そのほかの代表的なタスク：質問応答（文章から答えの範囲を抜き出す）、含意関係認識（2 文の関係）、文の類似度。

---

## 前処理と文書の表現 {#preprocessing}

### 形態素解析と n-gram

- **形態素解析**：文を意味を持つ最小の単位（形態素）に分け、品詞や原形を付ける。日本語は語の区切りがないので前処理として重要（MeCab など）。英語は空白で分けられる。
- **n-gram**：連続する $n$ 個の語（または文字）。$n=1$ はユニグラム、$2$ はバイグラム。文書の特徴量にも、言語モデルにも使う。

### Bag-of-Words と TF-IDF {#tfidf}

- **Bag-of-Words**：語の**出現回数**を並べたベクトルで文書を表す。語順は捨てる。長い文書ほど値が大きくなり、**語の重要度**も表せない。
- **TF-IDF**：その文書でよく出て（TF）、他の文書ではあまり出ない（IDF）語に大きな重みを付ける。

$$
\mathrm{tfidf}(t,d)=\mathrm{tf}(t,d)\cdot\mathrm{idf}(t),\qquad
\mathrm{tf}(t,d)=\frac{n_{t,d}}{\sum_{k}n_{k,d}},\qquad
\mathrm{idf}(t)=\log\frac{N}{\mathrm{df}(t)}
$$

- $n_{t,d}$：語 $t$ の文書 $d$ での出現回数、$N$：文書の総数、$\mathrm{df}(t)$：語 $t$ を含む文書の数。実装では $\log\frac{N}{1+\mathrm{df}}$ などに少し変える。
- 全文書に出る語は $\mathrm{df}=N$ で $\mathrm{idf}=0$（区別の役に立たない）。3 文書のうち 1 文書にしか出ない語は $\mathrm{idf}=\log3\approx1.10$。

### サブワード {#subword}

- 語を単位にすると、語彙が膨大になり、**未知語**（語彙にない語）が出る。**文字**にすると系列が長くなる。その中間が**サブワード**（語の一部）。頻出の語は 1 語、珍しい語は部品に分ける。

| 手法 | 考え方 |
|:--|:--|
| **BPE**（バイト対符号化） | 文字から始め、**最も頻出する隣り合う対**を 1 つの記号に併合することを、語彙の大きさになるまで繰り返す |
| **WordPiece** | 併合の基準が、頻度ではなく、**言語モデルの尤度**が最も上がる対。BERT で使用（`##` で語の続きを表す）。GNMT でも使用 |
| **SentencePiece** | **空白で区切る前処理を必要としない**。文を文字の列として扱い、空白も記号として含めて学習する。日本語・中国語のように語の区切りがない言語に向く。BPE か Unigram 言語モデルで語彙を作る |

- 未知語がなくなり、語彙を小さくできる。

---

## 単語の分散表現 {#embedding}

- **one-hot** は、次元が語彙数 $V$ と大きく、どの 2 語も直交して意味の近さが表せない。**分散表現**は、各語を数百次元の密なベクトルにして、**意味が近い語を近くに**置く。根拠は**分布仮説**：「似た文脈に現れる語は似た意味を持つ」。
- 埋め込み層は、$V\times d$ の行列 $\mathbf{W}$ から該当する行を取り出す操作（one-hot とのかけ算と同じ）。

### Word2Vec {#word2vec}

- 2 層の浅いネットワーク。入力側の重み $\mathbf{W}_{\mathrm{in}}$（各語の入力ベクトル $\mathbf{v}$）と出力側の重み $\mathbf{W}_{\mathrm{out}}$（出力ベクトル $\mathbf{u}$）を持ち、**$\mathbf{W}_{\mathrm{in}}$ を単語の埋め込みとして使う**。

| | 入力 → 予測 | 特徴 |
|:--|:--|:--|
| **CBOW** | 周りの語（文脈）→ 真ん中の語 | 速い。頻出語に強い |
| **skip-gram** | 真ん中の語 → 周りの語それぞれ | 計算は重いが、**低頻度語**でも精度が高い |

$$
\max\ \frac1T\sum_{t=1}^{T}\sum_{\substack{-c\le j\le c\\ j\ne0}}\log p(w_{t+j}\mid w_t),\qquad
p(w_O\mid w_I)=\frac{\exp(\mathbf{u}_{w_O}^{\top}\mathbf{v}_{w_I})}{\sum_{w=1}^{V}\exp(\mathbf{u}_{w}^{\top}\mathbf{v}_{w_I})}
$$

- 分母が語彙全体の和で、1 回の更新に $O(V)$ かかる。そこで、**負例サンプリング**（Negative Sampling）で、正解 1 個と、ランダムに選んだ負例 $K$ 個だけの **$K+1$ 個の 2 クラス分類**に置き換える。

$$
\log\sigma\bigl(\mathbf{u}_{w_O}^{\top}\mathbf{v}_{w_I}\bigr)+\sum_{k=1}^{K}\mathbb{E}_{w_k\sim P_n(w)}\Bigl[\log\sigma\bigl(-\mathbf{u}_{w_k}^{\top}\mathbf{v}_{w_I}\bigr)\Bigr],\qquad P_n(w)\propto U(w)^{3/4}
$$

- $U(w)$：語の出現頻度の分布（ユニグラム分布）。**$3/4$ 乗**すると、頻出語の確率が少し下がり、まれな語の確率が少し上がる（例：$0.9,0.1$ が $0.84,0.16$ に）。$K$ は小さなデータで 5〜20、大きなデータで 2〜5 が目安。
- **課題**：1 語に **1 つのベクトル**しかないので、**多義語**（銀行の bank と土手の bank）を区別できない。文脈ごとに変わる表現を作る **ELMo**（双方向 LSTM）や **BERT** が、これを解決した。語をサブワードの和で表す **fastText** は、未知語にも対応する。

---

## 言語モデル {#language-model}

$$
P(y_1,\dots,y_n)=\prod_{t=1}^{n}P(y_t\mid y_1,\dots,y_{t-1})\ \ \approx\ \prod_{t=1}^{n}P(y_t\mid y_{t-n+1},\dots,y_{t-1})\quad(\text{n-gram：直前の }n-1\text{ 語だけ使う})
$$

- **n-gram 言語モデル**は、出現回数から確率を数える：$P(y_t\mid y_{t-1})=\dfrac{C(y_{t-1}y_t)}{C(y_{t-1})}$。弱点は、$n$ を大きくすると**データがまばら**（見たことのない組が確率 0）になることと、**遠い語の関係**が使えないこと。0 を避けるために**スムージング**（加算・Kneser–Ney など）をする。
- **RNN 言語モデル**は、それまでの文脈を隠れ状態にまとめて渡すので、固定の長さに縛られない。→ [RNN・LSTM・GRU](../dl/rnn.md)。今は Transformer が主流。
- 評価は**パープレキシティ**（小さいほどよい）。語彙 $V$ 個を一様に予測するモデルのパープレキシティは $V$ で、「平均して何択から選んでいるか」に当たる。→ [性能指標](../ml/evaluation.md#generation)。

$$
\mathrm{Perplexity}=\exp\Bigl(-\frac1N\sum_{n=1}^{N}\log P(y_n\mid y_{<n})\Bigr)
$$

- **教師強制**（teacher forcing）：学習時、デコーダへの入力に**正解の前の語**を与える。計算が安定して収束が速い。ただし**推論時は自分の出力を入力**にするので、学習と推論で入力の分布が食い違う（**exposure bias**）。
- **生成の方法**：**貪欲法**は毎回最も確率の高い語を選ぶ。**ビームサーチ**は上位 $B$ 個の候補を保って探索する。**サンプリング**は確率に従って選び、温度（[温度つきソフトマックス](../dl/activation.md)の考え方）や top-$k$ で多様性を調整する。

---

## 系列変換と機械翻訳 {#seq2seq}

- **Seq2seq**：エンコーダが入力文を状態にまとめ、デコーダがそこから出力文を生成する。**Attention** は、出力の各時点で、入力のどこを見るかを重みで選ぶ。詳しくは [系列変換・Attention](../dl/seq2seq.md)。

### GNMT（Google の機械翻訳、2016） {#gnmt}

| 工夫 | 内容 |
|:--|:--|
| 深いエンコーダ・デコーダ | **LSTM を 8 層ずつ**。深い層は勾配が届かないので**残差接続**で $\mathbf{x}^{i}_t=\mathbf{m}^{i}_t+\mathbf{x}^{i-1}_t$ とつなぐ |
| 双方向は**最下層だけ** | 全層を双方向にすると、各層が前後の計算の完了を待ち、並列化できない |
| Attention | デコーダの**最下層**の出力と、エンコーダの**最上層**の出力で計算し、デコーダの全層へ送る（最上層どうしだと待ち時間が出るため） |
| 並列化 | **モデル並列**（層ごとに別の GPU）と**データ並列**（非同期の更新） |
| サブワード | **WordPiece**（未知語を部品に分ける。語彙は 32,000 程度） |
| 強化学習による再学習 | 最大尤度 $O_{\mathrm{ML}}$ と、翻訳の報酬（BLEU に近い指標）を使う $O_{\mathrm{RL}}$ を混ぜる |
| 量子化 | 推論で、重みを 8 ビットにして高速化（→ [軽量化・高速化](efficiency.md#quantization)） |

$$
O_{\mathrm{mixed}}(\theta)=\alpha\,O_{\mathrm{ML}}(\theta)+O_{\mathrm{RL}}(\theta),\qquad
O_{\mathrm{RL}}(\theta)=\sum_{i}\sum_{Y}P_\theta(Y\mid X^{(i)})\,r(Y,Y^{*(i)})
$$

- 最大尤度の学習は「次の語の当てやすさ」を最適化するが、評価指標（BLEU）とは別物。強化学習で**評価指標に近い報酬**を直接上げる。

**ビームサーチの得点**：長い文は確率の積が小さくなるので、**長さの正規化**（lp）と、入力の語を**漏れなく訳したか**（cp：被覆率）の補正を加える。

$$
s(Y,X)=\frac{\log P(Y\mid X)}{lp(Y)}+cp(X;Y),\qquad lp(Y)=\frac{(5+|Y|)^{\alpha}}{(5+1)^{\alpha}},\qquad cp(X;Y)=\beta\sum_{i=1}^{|X|}\log\Bigl(\min\Bigl(\sum_{j=1}^{|Y|}p_{i,j},\,1\Bigr)\Bigr)
$$

- $p_{i,j}$：出力の $j$ 語目が入力の $i$ 語目に払った Attention の重み。入力の語 $i$ に払った重みの合計が 1 以上なら 0、足りないと負（罰）になる。$|Y|=1$ なら $lp=1$。$\alpha=0.6,\ |Y|=10$ で $lp\approx1.73$。

---

## 事前学習済みモデル {#pretrained}

- Transformer の構造・BERT と GPT の違いは [Transformer](../dl/transformer.md#bert-gpt) にまとめてある。ここでは流れを整理する。

| モデル | 要点 |
|:--|:--|
| ELMo（2018） | 前向きと後ろ向きの LSTM を**別々に**学習して結合（**浅い双方向**）。文脈つきの単語表現 |
| GPT（2018） | Transformer の**デコーダ**。次の語の予測で事前学習し、ファインチューニング |
| **BERT**（2018） | Transformer の**エンコーダ**（**深い双方向**）。MLM（15 %）と NSP で事前学習し、出力層を足して**ファインチューニング** |
| RoBERTa（2019） | BERT の学習を見直し（NSP を外す、動的マスク、大きなバッチ） |
| GPT-2（2019） | 大規模化。**ファインチューニングなし**（zero-shot）でも多くのタスクを解く |
| T5（2019） | すべてのタスクを「文章から文章へ」の形に統一したエンコーダ・デコーダ |
| **GPT-3**（2020） | 約 1750 億パラメータ。**パラメータを更新せず**、プロンプトの例だけで解く（**in-context learning**） |
| 指示に従うモデル | 人が書いた指示と回答での**教師ありファインチューニング**、人の好みを学習した報酬による**強化学習**（RLHF）で、指示に沿った出力にそろえる |

- **使い方の違い**：**ファインチューニング**（パラメータを更新）／**特徴量として利用**（BERT の出力を固定して使う）／**プロンプト**（更新せず、例を並べる）。大きなモデルの全パラメータの更新は重いので、一部だけ更新する方法（**LoRA** など。→ [軽量化・高速化](efficiency.md#lora)）がある。
- **N-shot の用語**：GPT の Zero-shot は「タスクの説明だけ」、One-shot は「説明と例 1 つ」、Few-shot は「説明と例数個」をプロンプトに入れる。一般の機械学習の「学習時にないクラスを扱う」などとは意味が少し違う。
- **規模の法則**：モデルの大きさ・データ量・計算量を増やすと、損失がべき乗則に従って下がる（Kaplan ほか, 2020）。
- 評価に使うデータ：**GLUE**（文の理解の複数タスクの集まり）、**SQuAD**（文章中から答えを抜き出す）、**SWAG**（常識推論の 4 択）。

---

## 試験の着眼点

- **TF-IDF** $=\mathrm{tf}\times\mathrm{idf}$、$\mathrm{idf}=\log(N/\mathrm{df})$。全文書に出る語の重みは 0。Bag-of-Words は**語順を捨てる**。
- **WordPiece**（尤度基準）・**BPE**（頻度基準）・**SentencePiece**（**空白の前処理が不要**）の違い。サブワードにすると**未知語が減り**、語彙が小さくなる。
- **skip-gram**：中心語から周りを予測。**CBOW**：周りから中心語を予測。skip-gram のほうが重いが低頻度語に強い。**負例サンプリング**は $K+1$ 個の 2 クラス分類で、$P_n\propto U^{3/4}$。Word2Vec は**多義語を区別できない**。
- n-gram は**データのまばらさ**、RNN は**勾配消失**、Transformer は**系列長の 2 乗**が弱点。
- **教師強制**は学習時に正解を入力、推論時は自分の出力を入力（**食い違いが exposure bias**）。
- **GNMT**：LSTM 8 層・**残差接続**・**最下層だけ双方向**・Attention はデコーダ最下層とエンコーダ最上層・WordPiece・モデル並列とデータ並列・強化学習・量子化。
- **BERT**：エンコーダ、MLM 15 %（80/10/10）、NSP。**GPT**：デコーダ、Masked Attention、次語予測。GPT-3 は**勾配を使わない in-context learning**。

---

## 参考

- [Efficient Estimation of Word Representations in Vector Space（Word2Vec, Mikolov ほか, 2013）](https://arxiv.org/abs/1301.3781)
- [Distributed Representations of Words and Phrases and their Compositionality（負例サンプリング, Mikolov ほか, 2013）](https://arxiv.org/abs/1310.4546)
- [Deep contextualized word representations（ELMo, Peters ほか, 2018）](https://arxiv.org/abs/1802.05365)
- [Neural Machine Translation of Rare Words with Subword Units（BPE, Sennrich ほか, 2015）](https://arxiv.org/abs/1508.07909)
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer（Kudo, Richardson, 2018）](https://arxiv.org/abs/1808.06226)
- [Google's Neural Machine Translation System（GNMT, Wu ほか, 2016）](https://arxiv.org/abs/1609.08144)
- [Attention Is All You Need（Vaswani ほか, 2017）](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（Devlin ほか, 2018）](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach（Liu ほか, 2019）](https://arxiv.org/abs/1907.11692)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer（T5, Raffel ほか, 2019）](https://arxiv.org/abs/1910.10683)
- [Language Models are Few-Shot Learners（GPT-3, Brown ほか, 2020）](https://arxiv.org/abs/2005.14165)
- [Training language models to follow instructions with human feedback（InstructGPT, Ouyang ほか, 2022）](https://arxiv.org/abs/2203.02155)
- [Scaling Laws for Neural Language Models（Kaplan ほか, 2020）](https://arxiv.org/abs/2001.08361)
- [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding（Wang ほか, 2018）](https://arxiv.org/abs/1804.07461)
