# RNN・LSTM・GRU

<p class="kw">キーワード：リカレントニューラルネットワーク（RNN）、順伝播の計算、逆伝播の計算（BPTT：Back Propagation Through Time）、Truncated BPTT、勾配消失・勾配爆発、勾配クリッピング、双方向 RNN、ゲート機構、忘却ゲート、入力ゲート、出力ゲート、LSTM（長期記憶と短期記憶）、メモリーセル（記憶セル）、覗き穴結合、GRU、更新ゲート、リセットゲート</p>

!!! abstract "要点"
    - RNN は**前の時刻の隠れ状態 $\mathbf{h}_{t-1}$ を今の入力と一緒に使い**、同じ重みを時刻ごとに繰り返す。可変長の系列を扱え、時間方向に展開した重み共有のネットワークとして学習できる。
    - 時間をさかのぼる逆伝播（BPTT）では、$\mathbf{W}_{hh}$ と $\tanh'$ の**積**が $T$ 個掛かるため、勾配が消えるか爆発する。
    - **LSTM / GRU はゲートで記憶を「足し算」で更新**し、勾配が時間を越えて保たれる通り道を作る。

---

## RNN の順伝播 {#forward}

$$
\begin{aligned}
\mathbf{u}_t&=\mathbf{W}_{xh}\mathbf{x}_t+\mathbf{W}_{hh}\mathbf{h}_{t-1}+\mathbf{b}_h,\qquad \mathbf{h}_t=\tanh(\mathbf{u}_t)\qquad(\mathbf{h}_0=\mathbf{0}) \\[1mm]
\mathbf{y}_t&=\mathbf{W}_{hy}\mathbf{h}_t+\mathbf{b}_y \qquad(\text{分類ならさらに softmax})
\end{aligned}
$$

| 記号 | 意味 | 形 |
|:--|:--|:--|
| $\mathbf{x}_t$ | 時刻 $t$ の入力 | $D$ |
| $\mathbf{h}_t$ | 隠れ状態（過去の要約） | $H$ |
| $\mathbf{W}_{xh},\ \mathbf{W}_{hh},\ \mathbf{W}_{hy}$ | 入力→隠れ、隠れ→隠れ、隠れ→出力 | $H\times D,\ H\times H,\ M\times H$ |
| $\mathbf{y}_t$ | 時刻 $t$ の出力 | $M$ |

- 重みは**全時刻で共有**。パラメータ数は系列の長さ $T$ によらず $H(D+H+1)+M(H+1)$。
- バッチで行方向に並べる書き方では $\mathbf{H}_t=\tanh(\mathbf{X}_t\mathbf{W}_{xh}+\mathbf{H}_{t-1}\mathbf{W}_{hh}+\mathbf{b}_h)$（$\mathbf{X}_t$：$N\times D$、$\mathbf{H}_t$：$N\times H$。重みは上の式の転置になる）。
- **多層（深い）RNN**：下の層の隠れ状態を上の層の入力にする。$\mathbf{h}^{(l)}_t=\tanh(\mathbf{W}^{(l)}_{xh}\mathbf{h}^{(l-1)}_t+\mathbf{W}^{(l)}_{hh}\mathbf{h}^{(l)}_{t-1}+\mathbf{b}^{(l)})$（$\mathbf{h}^{(0)}_t=\mathbf{x}_t$）。

| 入出力の型 | 例 |
|:--|:--|
| 多対1 | 文の感情分析（最後の $\mathbf{h}_T$ だけ使う） |
| 1対多 | 画像からのキャプション生成 |
| 多対多（同期） | 品詞タグ付け、フレームごとのラベル付け |
| 多対多（非同期） | 機械翻訳（[Seq2Seq](seq2seq.md)） |

### 言語モデルとして使う {#language-model}

- 単語列の確率を、**次の単語の予測**の積に分解する。RNN は各時刻で $P(w_t\mid w_{<t})$ を出力する。

$$
P(w_1,\dots,w_T)=\prod_{t=1}^{T}P(w_t\mid w_1,\dots,w_{t-1}),\qquad
\mathrm{PPL}=\exp\!\Bigl(-\frac{1}{T}\sum_{t=1}^{T}\log P(w_t\mid w_{<t})\Bigr)
$$

- 入力は単語 ID を**埋め込み層**（Embedding：語彙数 $\times$ 埋め込み次元の行列の行を引く）でベクトルにしたもの。形は $(N,T)\to(N,T,D)$。
- **パープレキシティ**（PPL）は、平均して「何択で迷っているか」を表す。小さいほどよい。
- 損失は各時刻の交差エントロピーの和（または平均）$L=\sum_tL_t$（[出力層と損失関数](mlp-loss.md)）。

---

## BPTT（時間をさかのぼる逆伝播） {#bptt}

時間方向に展開すると、重みを共有した $T$ 層のネットワークになる。通常の[誤差逆伝播法](backprop.md)をそのまま適用し、**共有された重みの勾配は時刻ごとの寄与の合計**にする。

$\boldsymbol{\delta}_t\equiv\partial L/\partial\mathbf{u}_t$ と定義する（$\boldsymbol{\delta}_{T+1}=\mathbf{0}$）。

$$
\begin{aligned}
&\frac{\partial L}{\partial \mathbf{h}_t}=\mathbf{W}_{hy}^{\top}\frac{\partial L_t}{\partial \mathbf{y}_t}+\mathbf{W}_{hh}^{\top}\boldsymbol{\delta}_{t+1}
&&\text{（出力側と次の時刻の2方向から来る）} \\[2mm]
&\boldsymbol{\delta}_t=\frac{\partial L}{\partial \mathbf{h}_t}\odot\bigl(1-\mathbf{h}_t^{\odot2}\bigr)
&&(\tanh'(u)=1-\tanh^2(u)) \\[2mm]
&\frac{\partial L}{\partial \mathbf{W}_{hh}}=\sum_{t=1}^{T}\boldsymbol{\delta}_t\,\mathbf{h}_{t-1}^{\top},\quad
\frac{\partial L}{\partial \mathbf{W}_{xh}}=\sum_{t=1}^{T}\boldsymbol{\delta}_t\,\mathbf{x}_t^{\top},\quad
\frac{\partial L}{\partial \mathbf{b}_h}=\sum_{t=1}^{T}\boldsymbol{\delta}_t
\end{aligned}
$$

- 形の検算：$\boldsymbol{\delta}_t\mathbf{h}_{t-1}^{\top}$ は $(H\times1)(1\times H)=H\times H$ で $\mathbf{W}_{hh}$ と同じ。
- ソフトマックス＋交差エントロピーなら $\partial L_t/\partial\mathbf{y}_t=\mathbf{y}_t-\mathbf{t}_t$。

### Truncated BPTT

- 長い系列では、$T$ 全体を展開するとメモリと計算が増える。そこで $\tau$ 時刻ごとに**逆伝播の鎖を切る**。
- 順伝播では隠れ状態を次のブロックへ**引き継ぐ**（stateful）。逆伝播だけをブロック内に限る。ミニバッチは系列の別々の位置から取り、順番にずらす。

### 勾配消失と勾配爆発 {#vanishing}

$$
\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_k}=\prod_{t=k+1}^{T}\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}=\prod_{t=k+1}^{T}\mathrm{diag}\bigl(1-\mathbf{h}_t^{\odot2}\bigr)\,\mathbf{W}_{hh}
$$

- $T-k$ 個の行列の積。1ステップの倍率は $\tanh'\le1$ と $\mathbf{W}_{hh}$ で決まる。
- $\mathbf{W}_{hh}$ の最大特異値が 1 未満なら、積は指数的に**0 へ**（勾配消失：遠い過去の入力が学習に効かず、長期依存を学べない）。
- 最大固有値の絶対値が 1 を超えると、積が**爆発しうる**（勾配爆発：更新が暴れて損失が発散する）。
- 1次元で単純化すれば、倍率は $|w|\tanh'(u)$ の $T$ 乗。tanh の微分は最大 1 で、$u\ne0$ では 1 より小さいので、$|w|\le1$ ならほぼ必ず消える。

<div class="ew" data-widget="rnn-gradient"></div>

!!! tip "動かしてみる"
    - 初期設定（$|w|=0.9$）でも、20 ステップさかのぼると約 $10^{-3}$ まで小さくなります。$u=0$ に近づけると $\tanh'$ が 1 に近づき、少し持ちこたえます。
    - $|w|$ を 1.5 以上にすると、$u$ が小さいときは右へ伸びて**爆発**します。$|u|$ を大きくすると $\tanh'$ が小さくなり、爆発を抑えます。
    - LSTM の記憶セルの経路は、$f$ を 1 に近づけると**ほとんど減りません**（$0.95^{20}\approx0.36$）。これが LSTM の狙いです。

**対策**

| 問題 | 対策 |
|:--|:--|
| 勾配爆発 | **勾配クリッピング**：$\lVert\mathbf{g}\rVert>\theta$ なら $\mathbf{g}\leftarrow\theta\,\mathbf{g}/\lVert\mathbf{g}\rVert$（向きを保ったまま大きさだけ抑える） |
| 勾配消失 | **ゲート機構**（LSTM・GRU）、ReLU＋単位行列での初期化、残差接続、Truncated BPTT で短く切る |
| 過学習 | ドロップアウトは**再帰でない結合（層間）にだけ**かける。時刻をまたぐ結合にかけると記憶が壊れる |

---

## 双方向 RNN {#birnn}

$$
\overrightarrow{\mathbf{h}}_t=\phi\bigl(\overrightarrow{\mathbf{W}}_{xh}\mathbf{x}_t+\overrightarrow{\mathbf{W}}_{hh}\overrightarrow{\mathbf{h}}_{t-1}+\overrightarrow{\mathbf{b}}\bigr),\quad
\overleftarrow{\mathbf{h}}_t=\phi\bigl(\overleftarrow{\mathbf{W}}_{xh}\mathbf{x}_t+\overleftarrow{\mathbf{W}}_{hh}\overleftarrow{\mathbf{h}}_{t+1}+\overleftarrow{\mathbf{b}}\bigr),\quad
\mathbf{h}_t=\bigl[\overrightarrow{\mathbf{h}}_t;\overleftarrow{\mathbf{h}}_t\bigr]
$$

- 系列を**前向きと後ろ向きの2つの RNN** で処理し、各時刻で連結する（次元は $2H$）。過去と未来の両方の文脈を使える。
- 出力層には連結した $\mathbf{h}_t$ を入れる：$\mathbf{y}_t=\mathbf{W}_{hy}\mathbf{h}_t+\mathbf{b}_y$。$\mathbf{W}_{hy}$ は $M\times2H$。
- **系列全体が先に手に入る**場面（タグ付け、音声認識、翻訳の**エンコーダ**）に向く。次の単語を予測して生成するデコーダには使えない（未来を見てしまう）。

---

## LSTM {#lstm}

RNN の隠れ状態に加えて、**記憶セル** $\mathbf{c}_t$（長期記憶）を持つ。$\mathbf{h}_t$ は短期記憶・出力にあたる。3つのゲート（0〜1 の値、$\sigma$ の出力）が、記憶セルへの出入りを調節する。

$$
\begin{aligned}
\mathbf{f}_t&=\sigma\bigl(\mathbf{W}_f[\mathbf{x}_t;\mathbf{h}_{t-1}]+\mathbf{b}_f\bigr)&&\text{忘却ゲート} \\
\mathbf{i}_t&=\sigma\bigl(\mathbf{W}_i[\mathbf{x}_t;\mathbf{h}_{t-1}]+\mathbf{b}_i\bigr)&&\text{入力ゲート} \\
\mathbf{o}_t&=\sigma\bigl(\mathbf{W}_o[\mathbf{x}_t;\mathbf{h}_{t-1}]+\mathbf{b}_o\bigr)&&\text{出力ゲート} \\
\tilde{\mathbf{c}}_t&=\tanh\bigl(\mathbf{W}_c[\mathbf{x}_t;\mathbf{h}_{t-1}]+\mathbf{b}_c\bigr)&&\text{記憶の候補} \\[1mm]
\mathbf{c}_t&=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t&&\text{記憶の更新} \\
\mathbf{h}_t&=\mathbf{o}_t\odot\tanh(\mathbf{c}_t)&&\text{出力}
\end{aligned}
$$

| ゲート | はたらき | 0 のとき | 1 のとき |
|:--|:--|:--|:--|
| 忘却 $\mathbf{f}$ | 過去の記憶 $\mathbf{c}_{t-1}$ をどれだけ残すか | 忘れる | そのまま保つ |
| 入力 $\mathbf{i}$ | 新しい候補 $\tilde{\mathbf{c}}_t$ をどれだけ書き込むか | 書き込まない | 全部書き込む |
| 出力 $\mathbf{o}$ | 記憶のうちどれだけを外へ出すか | 出さない（記憶は残る） | 出す |

- $[\mathbf{x}_t;\mathbf{h}_{t-1}]$ は連結。$\mathbf{W}_f=[\mathbf{W}_{xf}\ \ \mathbf{W}_{hf}]$ なので、$\mathbf{W}_{xf}\mathbf{x}_t+\mathbf{W}_{hf}\mathbf{h}_{t-1}$ と同じ。
- パラメータ数は $4H(D+H+1)$（普通の RNN の4倍）。**4組のアフィン変換を1回の行列積にまとめ**（形 $(D+H)\times4H$）、結果を4つに切り分けて使う実装が一般的。
- **勾配が消えにくい理由**：記憶の更新が掛け算でなく**足し算**で、記憶セルを直接さかのぼる経路の倍率は $\partial\mathbf{c}_t/\partial\mathbf{c}_{t-1}=\mathrm{diag}(\mathbf{f}_t)$。$\mathbf{f}_t\approx1$ なら勾配がほぼそのまま時間を越える（**CEC：定誤差カルーセル**）。ゲートや $\tilde{\mathbf{c}}_t$ を通る経路もあるが、この直接の経路が長期の流れを担う。
- 忘却ゲートのバイアス $\mathbf{b}_f$ は**正の値（1 など）で初期化**し、最初は忘れにくくする。
- **覗き穴結合（peephole）**：ゲートにも記憶セルの値を見せる。$\mathbf{f}_t,\mathbf{i}_t$ の入力に $\mathbf{w}\odot\mathbf{c}_{t-1}$、$\mathbf{o}_t$ の入力に $\mathbf{w}\odot\mathbf{c}_t$ を足す。
- 初期の LSTM（1997）には忘却ゲートがなく、後から追加された。

## GRU {#gru}

LSTM を簡単にした。記憶セルを持たず、**隠れ状態 $\mathbf{h}_t$ だけ**で、ゲートは2つ。

$$
\begin{aligned}
\mathbf{z}_t&=\sigma\bigl(\mathbf{W}_z[\mathbf{x}_t;\mathbf{h}_{t-1}]+\mathbf{b}_z\bigr)&&\text{更新ゲート} \\
\mathbf{r}_t&=\sigma\bigl(\mathbf{W}_r[\mathbf{x}_t;\mathbf{h}_{t-1}]+\mathbf{b}_r\bigr)&&\text{リセットゲート} \\
\tilde{\mathbf{h}}_t&=\tanh\bigl(\mathbf{W}_h[\mathbf{x}_t;\ \mathbf{r}_t\odot\mathbf{h}_{t-1}]+\mathbf{b}_h\bigr)&&\text{候補} \\
\mathbf{h}_t&=\mathbf{z}_t\odot\mathbf{h}_{t-1}+(1-\mathbf{z}_t)\odot\tilde{\mathbf{h}}_t&&\text{更新}
\end{aligned}
$$

- **更新ゲート** $\mathbf{z}$：古い状態と候補の**混ぜ合わせ比**。$\mathbf{z}=\mathbf{1}$ なら何も更新せず $\mathbf{h}_t=\mathbf{h}_{t-1}$、$\mathbf{z}=\mathbf{0}$ なら全部を候補に置き換える。LSTM の忘却と入力を $\mathbf{f}=\mathbf{z},\ \mathbf{i}=1-\mathbf{z}$ と**連動**させたものにあたる。
- **リセットゲート** $\mathbf{r}$：候補を作るときに過去をどれだけ参照するか。$\mathbf{r}=\mathbf{0}$ なら過去を無視して現在の入力だけで候補を作る。
- 文献により $\mathbf{z}$ と $1-\mathbf{z}$ の役割が逆の書き方もある（意味は同じ）。上は元論文（Cho ら）の向き。
- パラメータ数は $3H(D+H+1)$。例：$D=100,\ H=128$ で RNN $29{,}312$、GRU $87{,}936$、LSTM $117{,}248$。

| | RNN | LSTM | GRU |
|:--|:--|:--|:--|
| 状態 | $\mathbf{h}$ | $\mathbf{h},\ \mathbf{c}$ | $\mathbf{h}$ |
| ゲート数 | 0 | 3 | 2 |
| パラメータ（相対） | 1 | 約 4 | 約 3 |
| 長期依存 | 苦手 | 得意 | 得意（LSTM と同程度のことが多い） |

<div class="ew" data-widget="rnn-lstm"></div>

!!! tip "動かしてみる"
    - LSTM で $a_f$ を大きな負の値にすると $f\approx0$ になり、$c_t$ は過去 $c_{t-1}$ を失って新しい候補だけになります。逆に $a_f$ を大きな正の値にして $a_i$ を負にすると、$c_t\approx c_{t-1}$ で記憶がそのまま保たれます。
    - $a_o$ を負にすると $h_t\approx0$ ですが、$c_t$ は変わりません。**記憶していることと、出力することは別**です。
    - GRU に切り替えて $a_z$ を大きくすると $z\approx1$ になり、$h_t\approx h_{t-1}$（更新しない）。小さくすると候補に置き換わります。$a_r$ を負にすると、候補が過去 $h_{t-1}$ を無視します。

---

## 試験の着眼点

- RNN の隠れ状態の更新式 $\mathbf{h}_t=\tanh(\mathbf{W}_{xh}\mathbf{x}_t+\mathbf{W}_{hh}\mathbf{h}_{t-1}+\mathbf{b})$ と、**重みが時刻をまたいで共有される**こと。
- BPTT：$\mathbf{W}_{hh}$ などの勾配は**時刻の合計**。隠れ状態への勾配は**出力側と次の時刻の2方向**から来るので足す。
- 勾配消失・爆発の原因は $\mathbf{W}_{hh}$ と $\tanh'$ の**積**。爆発には勾配クリッピング、消失にはゲート機構。
- Truncated BPTT は、逆伝播だけを切り、順伝播の状態は引き継ぐ。
- LSTM の式（6本）を書ける。**記憶セルは足し算で更新**され、$\partial\mathbf{c}_t/\partial\mathbf{c}_{t-1}=\mathrm{diag}(\mathbf{f}_t)$。出力ゲートが0でも記憶は消えない。
- GRU：ゲートは更新とリセットの2つ、記憶セルなし。パラメータ数は LSTM の約 3/4。
- 双方向 RNN は未来も使うので、**逐次生成のデコーダには使えない**。
- 言語モデルの評価はパープレキシティ（小さいほどよい）。

---

## 参考

- [Long Short-Term Memory（Hochreiter, Schmidhuber, 1997）](https://doi.org/10.1162/neco.1997.9.8.1735)
- [On the difficulty of training Recurrent Neural Networks（Pascanu ら, 2012）](https://arxiv.org/abs/1211.5063)（勾配消失・爆発の解析、勾配クリッピング）
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation（Cho ら, 2014）](https://arxiv.org/abs/1406.1078)（GRU の提案）
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling（Chung ら, 2014）](https://arxiv.org/abs/1412.3555)
- [Recurrent Neural Network Regularization（Zaremba ら, 2014）](https://arxiv.org/abs/1409.2329)（再帰でない結合にだけドロップアウト）
- [A Simple Way to Initialize Recurrent Networks of Rectified Linear Units（Le ら, 2015）](https://arxiv.org/abs/1504.00941)
