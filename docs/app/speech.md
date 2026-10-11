# 音声処理

<p class="kw">キーワード：標本化定理、量子化、PCM、離散フーリエ変換、短時間フーリエ変換、窓関数、スペクトログラム、メル尺度、メルフィルタバンク、ケプストラム、MFCC、スペクトル包絡、フォルマント、音声認識、CTC、WaveNet、Dilated Causal Convolution、μ-law、Gated Activation Unit、音声合成</p>

!!! abstract "要点"
    - 音声は**波形**のままでは扱いにくい。**短時間フーリエ変換**で時間ごとの周波数の強さ（スペクトログラム）にし、人の聞こえ方に合わせた**メル尺度**で整える。声の特徴は **MFCC**（スペクトル包絡）に出る。
    - **音声認識**は、音の列 → 文字の列への変換。入出力の長さが違い、対応が分からないので、**CTC**（ブランクを使った全経路の和）などで学習する。
    - **音声合成**の **WaveNet** は、過去のサンプルから次のサンプルを予測する自己回帰モデル。**拡張（dilated）因果畳み込み**で、少ない層で長い過去を見る。

---

## 音声のデジタル化 {#digitization}

$$
\text{アナログ信号}\ \xrightarrow{\text{標本化}}\ \text{離散時間}\ \xrightarrow{\text{量子化}}\ \text{離散値}\ \xrightarrow{\text{符号化}}\ \text{2進数（PCM）}
$$

- **標本化**（サンプリング）：一定の間隔で値を取る。1 秒あたりの回数が**標本化周波数** $f_s$（例：16000 Hz）。
- **量子化**：値を決まった段階に丸める。$b$ ビットなら $2^b$ 段階（16 ビットで 65,536）。ビット数が大きいほど振幅が細かく表せる。
- **符号化**：量子化した値を 2 進数にする。この方式を **PCM**（パルス符号変調）という。
- **標本化定理**：信号に含まれる最大の周波数が $f_{\max}$ のとき、$f_s>2f_{\max}$ で標本化すれば元の波形を復元できる。$f_s/2$ を**ナイキスト周波数**といい、これより高い周波数は**折り返し**（エイリアシング）で低い周波数に化ける。
- 例：音声認識は 16 kHz（0〜8 kHz の成分を保つ）、電話は 8 kHz（〜4 kHz）、CD は 44.1 kHz・16 ビット。

---

## 周波数領域へ：STFT とスペクトログラム {#stft}

- **離散フーリエ変換**（DFT）は、長さ $N$ の信号を、周波数ごとの複素数の係数（実部＝cos 成分、虚部＝sin 成分）に分ける。**絶対値**が強さ（スペクトル）、2 乗がパワースペクトル。

$$
X[k]=\sum_{n=0}^{N-1}x[n]\,e^{-j2\pi kn/N},\qquad f_k=k\frac{f_s}{N},\qquad P[k]=|X[k]|^2=\mathrm{Re}(X[k])^2+\mathrm{Im}(X[k])^2
$$

| 時間 | 周波数 | 変換 |
|:--|:--|:--|
| 連続・周期的 | 離散 | フーリエ級数展開 |
| 連続 | 連続 | フーリエ変換 |
| 離散 | 連続 | 離散時間フーリエ変換 |
| 離散 | 離散 | **離散フーリエ変換（DFT）**。高速に計算するのが **FFT**（$N$ は 2 の冪が速い） |

- 音声は時間とともに変わるので、**短い区間ごと**に切り出して DFT をする（**短時間フーリエ変換**、STFT）。

$$
\text{窓で切り出す（長さ }N\text{）}\ \to\ \text{窓関数を掛ける}\ \to\ \mathrm{FFT}\ \to\ \text{窓を }H\text{ サンプルずらして繰り返す}
$$

- 結果を「時間 × 周波数」の図にして、強さを色で表したものが**スペクトログラム**（時間・周波数・強さの 3 次元）。
- **窓の長さのトレードオフ**：窓の時間長は $N/f_s$、周波数の間隔は $f_s/N$。**窓を長くすると周波数分解能が上がり、時間分解能が下がる**（逆も同じ）。例：$f_s=16$ kHz、$N=1024$ なら、窓は 64 ms、周波数の間隔は 15.625 Hz、**0〜8 kHz の 513 点**（$N/2+1$）が得られる。
- **窓関数**：DFT は切り出した区間が**周期的に繰り返す**と仮定するので、両端の値が違うと急な段差ができ、**高調波の歪み**（スペクトルの漏れ）が出る。両端がなめらかに 0 に近づく関数（ハン窓など）を掛けて抑える。

$$
w[n]=0.5\Bigl(1-\cos\frac{2\pi n}{N-1}\Bigr)\quad(\text{ハン窓})
$$

- 窓関数の良さは、**メインローブが細い**（周波数分解能が高い）ことと、**サイドローブが小さい**（弱い成分が見える）こと。矩形窓はメインローブが最も細いが、サイドローブが大きい。

---

## メル尺度と MFCC {#mfcc}

- 人の耳は、**低い周波数の違いには敏感で、高い周波数の違いには鈍い**（ほぼ対数的）。この知覚に合わせた尺度が**メル尺度**。

$$
m=2595\log_{10}\Bigl(1+\frac{f}{700}\Bigr)\qquad(1000\ \mathrm{Hz}\approx1000\ \mathrm{mel},\quad 8000\ \mathrm{Hz}\approx2840\ \mathrm{mel})
$$

- **メルフィルタバンク**：メル尺度で等間隔に並べた三角形のフィルタ（低域は細かく、高域は粗い）。パワースペクトルに掛けて足すと、フィルタ数（例：40）の**メル帯域スペクトル**になる。人の知覚が対数的なので、さらに**対数**をとる（**対数メルスペクトル**）。
- **ケプストラム**：スペクトルの対数を、さらに信号とみなして変換したもの（スペクトルのスペクトル）。横軸は**ケフレンシー**。
    - **低いケフレンシー**はスペクトルの**ゆるやかな変動**（**スペクトル包絡**＝声道の形、母音の特徴）、高いケフレンシーは**細かい山谷**（声帯の振動＝声の高さ）を表す。
    - 包絡の**山（ピーク）**を**フォルマント**といい、そこの周波数（フォルマント周波数）が音素の特徴を持つ。
- **MFCC**（メル周波数ケプストラム係数）：**対数メルスペクトルに離散コサイン変換（DCT）**をして、低次の 12〜13 次元だけを使う。DCT で隣り合う帯域の相関が取れ、低次だけ残すので包絡が残る。時間方向の変化 **$\Delta$MFCC**（と、その変化 $\Delta\Delta$）を足すことが多い（子音の特徴）。

| 手順 | 処理 | 得られるもの |
|:-:|:--|:--|
| 1 | AD 変換 | 音声データ（PCM） |
| 2 | 窓で切り出し、窓関数 | フレーム |
| 3 | FFT → 2 乗 | パワースペクトル |
| 4 | メルフィルタバンク | メル帯域スペクトル |
| 5 | **対数** | 対数メルスペクトル |
| 6 | **DCT**（低次だけ残す） | **MFCC** |

- ニューラルネットワークで認識する現在は、MFCC まで進めず、**対数メルスペクトログラム**をそのまま CNN や Transformer に入れることも多い。生の波形から学習するモデル（wav2vec 2.0 など）もある。

---

## 音声認識 {#asr}

- 音の列 $\mathbf{X}$ から文字（単語）の列 $W$ を求める。古典的な方式は、ベイズの定理で**音響モデル**と**言語モデル**に分ける。

$$
\hat{W}=\arg\max_W P(W\mid\mathbf{X})=\arg\max_W\ \underbrace{P(\mathbf{X}\mid W)}_{\text{音響モデル}}\ \underbrace{P(W)}_{\text{言語モデル}}
$$

- 音響モデルは **HMM**（隠れマルコフモデル。→ [自己教師あり学習・生成モデル](../ml/ssl-generative.md)）＋ GMM、のちに DNN に置き換わった（DNN-HMM）。いまは音から文字へ**直接**変換する **end-to-end** が主流。

| 方式 | 要点 |
|:--|:--|
| **CTC** | フレームごとに文字（とブランク）を出力し、系列の対応づけを**周辺化**。入力と出力の**長さが違ってもよい** |
| Attention エンコーダ・デコーダ（LAS など） | Seq2seq + Attention。出力文字を 1 つずつ生成。単調な対応の保証がない |
| RNN-Transducer | CTC の独立性の仮定を、予測ネットワークで緩める。ストリーミング認識に向く |
| wav2vec 2.0 / Whisper | 大量の**ラベルなし**音声（自己教師あり）／大量の**弱いラベル付き**音声で事前学習 |

- 認識の評価は**単語誤り率** $\mathrm{WER}=\dfrac{S+D+I}{N}$（置換・削除・挿入の数を、正解の単語数 $N$ で割る。編集距離）。

### CTC {#ctc}

- 入力 $\mathbf{x}$（$T$ フレーム）に対し、ネットワークは各時刻 $t$ に、**文字とブランク**の確率 $y^t_k$（softmax）を出す。各時刻にラベルを 1 つ選んだ長さ $T$ の列を**パス** $\pi$ という。
- **縮約関数** $\mathcal{B}$：パスから、**連続する同じ文字を 1 つにまとめ、ブランクを消して**ラベル列にする。例：`_aa_i` → `ai`、`aai` → `ai`。**同じ文字が連続するラベル**（cutter の tt）は、間にブランクが必要。

$$
p(\boldsymbol{\pi}\mid\mathbf{x})=\prod_{t=1}^{T}y^t_{\pi_t},\qquad
p(\mathbf{l}\mid\mathbf{x})=\sum_{\boldsymbol{\pi}\in\mathcal{B}^{-1}(\mathbf{l})}p(\boldsymbol{\pi}\mid\mathbf{x}),\qquad
L_{\mathrm{CTC}}=-\log p(\mathbf{l}\mid\mathbf{x})
$$

- 正解のラベル列 $\mathbf{l}$ になる**すべてのパスの確率を足す**。パスの数は膨大だが、**動的計画法**（前向き・後ろ向きアルゴリズム）で効率よく足せる。
- 計算では、$\mathbf{l}$ の両端と各文字の間にブランクを入れた**拡張ラベル列** $\mathbf{l}'$（長さ $|\mathbf{l}'|=2|\mathbf{l}|+1$）を使う。**前向き確率** $\alpha_t(s)$：時刻 $t$ までに、$\mathbf{l}'$ の $s$ 番目までを出力する全パスの確率の和。

$$
\alpha_t(s)=\begin{cases}
\bigl(\alpha_{t-1}(s)+\alpha_{t-1}(s-1)\bigr)\,y^t_{l'_s} & (l'_s=\text{ブランク},\ \text{または}\ l'_{s-2}=l'_s)\\[1mm]
\bigl(\alpha_{t-1}(s)+\alpha_{t-1}(s-1)+\alpha_{t-1}(s-2)\bigr)\,y^t_{l'_s} & (\text{それ以外})
\end{cases}
$$

- 初期値は $\alpha_1(1)=y^1_{\_},\ \alpha_1(2)=y^1_{l_1}$（先頭はブランクか最初の文字）、ほかは 0。
- $s-2$ からの遷移（ブランクを 1 つ飛ばして次の文字へ進む）は、$l'_s$ が文字で、かつ $l'_{s-2}
e l'_s$ のときだけ許される。**同じ文字**どうしでブランクを飛ばすと、縮約で 2 文字が 1 文字にまとめられてしまう。**ブランクへ**は、直前の文字を飛ばして進めない。最後は $p(\mathbf{l}\mid\mathbf{x})=\alpha_T(|\mathbf{l}'|)+\alpha_T(|\mathbf{l}'|-1)$（最後の文字で終わるか、その後のブランクで終わるか）。

**【例】** $T=4$、文字は `a`・`i`・ブランク `_`。ネットワークの出力確率を次とする。

| ラベル | $t=1$ | $t=2$ | $t=3$ | $t=4$ |
|:-:|:-:|:-:|:-:|:-:|
| a | 1/2 | 1/2 | 1/4 | 0 |
| i | 1/4 | 0 | 1/4 | 1 |
| _ | 1/4 | 1/2 | 1/2 | 0 |

- 正解を `ai` とすると、`ai` になるパスは 9 本（`aaai`、`aaii`、`aa_i`、`a_ii`、`a__i`、`_aai`、`_aii`、`_a_i`、`__ai`）で、確率の和は $p(\texttt{ai}\mid\mathbf{x})=19/32\approx0.594$（前向き確率の計算でも同じ値になる）。損失は $-\log(19/32)\approx0.52$。
- **Best Path Decoding**（推論での近似）：各時刻で**最も確率の高いラベル**を選び、縮約する。ここでは `aa_i`（確率 $1/8$）→ `ai`。ラベル列の確率（$19/32$）とパス 1 本の確率（$1/8$）は**別物**で、最良のパスが最良のラベル列とは限らない。
- 制約：フレーム間が**条件付き独立**と仮定する（言語モデルを併用して補う）。ラベル列の長さ $\le$ フレーム数。

---

## 音声合成と WaveNet {#wavenet}

- 音声合成は、**テキスト → 音響特徴（メルスペクトログラム）→ 波形**の 2 段が一般的（Tacotron 2 は前半を Seq2seq + Attention、後半を**ボコーダ**にする）。波形を生成するボコーダに WaveNet が使われた。

### WaveNet {#wavenet-model}

- PixelCNN をもとにした、**生の波形を直接生成する自己回帰モデル**。

$$
p(\mathbf{x})=\prod_{t=1}^{T}p(x_t\mid x_1,\dots,x_{t-1})
$$

- **因果畳み込み**（causal convolution）：**過去の**サンプルだけを使う（未来にマスクした畳み込み）。系列全体を並列に計算できるので、**学習は RNN より速い**。ただし受容野を広げるには層が多く必要。
- **拡張因果畳み込み**：層が深くなるにつれて、畳み込むサンプルの**間隔（dilation）を 1, 2, 4, 8, …, 512 と倍々に**する。カーネル幅 2 で、この 10 層の受容野は $1+(1+2+\dots+512)=1024$ サンプル（16 kHz で 64 ms）。通常の畳み込みなら 1023 層が必要。

$$
\text{受容野}=1+\sum_{\ell}(k-1)\,d_\ell\qquad(k:\text{カーネル幅},\ d_\ell:\text{層 }\ell\text{ の拡張率})
$$

- **$\mu$-law 圧縮**：16 ビットの音声は 65,536 通りで、softmax が重い。値を**対数的に圧縮**して **256 通り**に量子化し、「256 クラスの分類」として次のサンプルを予測する（小さい音を細かく表せる）。

$$
f(x_t)=\mathrm{sign}(x_t)\,\frac{\ln(1+\mu|x_t|)}{\ln(1+\mu)},\qquad -1<x_t<1,\ \ \mu=255
$$

- **ゲート付き活性化**（Gated Activation Unit）：出力候補（tanh）と、通す量を決めるゲート（sigmoid）に**別の重み**を使い、要素ごとに掛ける。$\ast$ は畳み込み、$f$ はフィルタ、$g$ はゲート、$k$ は層。

$$
\mathbf{z}=\tanh\bigl(W_{f,k}\ast\mathbf{x}\bigr)\odot\sigma\bigl(W_{g,k}\ast\mathbf{x}\bigr)
$$

- **残差接続**と**スキップ接続**：各層の入出力を足して（残差）、各層の出力は別に集めて最後の層へ送る（スキップ）。チャネル数は **1×1 畳み込み**で調整する。
- **条件付き WaveNet**：付加情報 $\mathbf{h}$ を与えて、出力を制御する。$p(\mathbf{x}\mid\mathbf{h})=\prod_t p(x_t\mid x_{<t},\mathbf{h})$。

$$
\begin{aligned}
&\text{全体条件（話者 ID など）} && \mathbf{z}=\tanh\bigl(W_{f,k}\ast\mathbf{x}+V_{f,k}^{\top}\mathbf{h}\bigr)\odot\sigma\bigl(W_{g,k}\ast\mathbf{x}+V_{g,k}^{\top}\mathbf{h}\bigr) \\[2mm]
&\text{局所条件（言語特徴・メルスペクトログラムなど、時間とともに変わる）} && \mathbf{z}=\tanh\bigl(W_{f,k}\ast\mathbf{x}+V_{f,k}\ast\mathbf{y}\bigr)\odot\sigma\bigl(W_{g,k}\ast\mathbf{x}+V_{g,k}\ast\mathbf{y}\bigr)
\end{aligned}
$$

- 局所条件の $\mathbf{y}=f(\mathbf{h})$ は、$\mathbf{h}$ を音声と同じ時間の細かさまで**アップサンプリング**（転置畳み込み）したもの。
- **欠点**：生成は**1 サンプルずつ**で非常に遅い（毎秒 16000 回）。これを解決した **Parallel WaveNet**（蒸留）、フロー（WaveGlow。→ [生成モデル](generative.md#flow)）、GAN（HiFi-GAN）のボコーダが続いた。

---

## 試験の着眼点

- **標本化定理**：$f_s>2f_{\max}$。ナイキスト周波数は $f_s/2$。足りないと**折り返し（エイリアシング）**。
- PCM は「**標本化 → 量子化 → 符号化**」。量子化のビット数が振幅の細かさ。
- STFT：窓で切り出し → 窓関数 → FFT → ずらして繰り返す。**窓を長くすると周波数分解能は上がり、時間分解能は下がる**。窓関数は、切り出しの両端の段差による**歪み**を抑える。
- **MFCC** の順序：パワースペクトル → **メルフィルタバンク** → **対数** → **DCT**。低次の係数が**スペクトル包絡**（声道特性）。$\Delta$MFCC は時間方向の変化。
- **CTC**：ブランクと縮約 $\mathcal{B}$、$p(\mathbf{l}\mid\mathbf{x})$ は**全パスの和**、拡張ラベル列の長さは $2|\mathbf{l}|+1$、前向き確率で計算、推論は Best Path（近似）。
- **WaveNet**：**拡張因果畳み込み**（dilation が 1, 2, 4, …、受容野が指数的に増える）、**$\mu$-law**（256 クラス）、**ゲート付き活性化**、残差・スキップ接続、**条件付け**（全体／局所）。学習は並列、**生成は逐次で遅い**。

---

## 参考

- [WaveNet: A Generative Model for Raw Audio（van den Oord ほか, 2016）](https://arxiv.org/abs/1609.03499)
- [Deep Speech: Scaling up end-to-end speech recognition（Hannun ほか, 2014）](https://arxiv.org/abs/1412.5567)
- [Listen, Attend and Spell（Chan ほか, 2015）](https://arxiv.org/abs/1508.01211)
- [Sequence Transduction with Recurrent Neural Networks（RNN-Transducer, Graves, 2012）](https://arxiv.org/abs/1211.3711)
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations（Baevski ほか, 2020）](https://arxiv.org/abs/2006.11477)
- [Robust Speech Recognition via Large-Scale Weak Supervision（Whisper, Radford ほか, 2022）](https://arxiv.org/abs/2212.04356)
- [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions（Tacotron 2, Shen ほか, 2017）](https://arxiv.org/abs/1712.05884)
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis（Kong ほか, 2020）](https://arxiv.org/abs/2010.05646)
- CTC の原論文：Graves ほか「Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks」（ICML 2006）
