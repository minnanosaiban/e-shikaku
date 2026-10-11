# 最適化

<p class="kw">キーワード：確率的勾配降下法（SGD）、学習率、最急降下法、ミニバッチ、モメンタム、Pathological Curvature、Momentum、Nesterov Accelerated Gradient、AdaGrad、RMSProp、Adam</p>

!!! abstract "要点"
    - 損失 $L$ を減らす向きに、勾配 $\mathbf{g}=\partial L/\partial\boldsymbol{\theta}$ の逆向きへパラメータを動かす。実際は**ミニバッチ**ごとに勾配を推定する（**SGD**）。
    - 谷が細長い（**悪条件**）と、SGD は谷の壁でジグザグして進まない。**Momentum** は過去の勾配をためて加速し、**Nesterov** は先読みした位置の勾配を使う。
    - **AdaGrad・RMSprop・Adam** は、勾配の2乗の履歴で**パラメータごとに歩幅を自動調整**する。Adam = Momentum + RMSprop + バイアス補正。

---

## 勾配降下法とミニバッチ {#gd}

$$
\boldsymbol{\theta}\leftarrow\boldsymbol{\theta}-\eta\,\mathbf{g},\qquad
\mathbf{g}=\frac{\partial L}{\partial\boldsymbol{\theta}}
$$

- $\eta$：**学習率**。大きすぎると谷を飛び越えて発散し、小さすぎると進まない・浅い谷にとどまる。
- 損失は全データの平均 $L=\frac1N\sum_i\ell_i$ なので、勾配も平均になる。どの範囲で平均するかで3種類に分かれる。

| 呼び方 | 1回の更新に使うデータ | 特徴 |
|:--|:--|:--|
| 最急降下法（バッチ勾配降下法） | 全データ $N$ 個 | 勾配は正確だが、1回が重く、メモリも要る |
| 確率的勾配降下法（SGD） | 1サンプル | 軽いが勾配のばらつき（ノイズ）が大きい |
| **ミニバッチ**（実際に SGD と呼ぶ） | $B$ 個（32, 64, 128 など） | 両者の中間。並列計算（GPU）に向く |

- ミニバッチ勾配 $\mathbf{g}_B=\frac1B\sum_{i\in B}\partial\ell_i/\partial\boldsymbol{\theta}$ は、全データの勾配の**不偏推定**で、ばらつき（分散）は $1/B$ に比例する。
- ノイズは悪いだけでなく、鞍点や浅い局所解から抜け出す助けになる（[正則化](regularization.md#implicit) の陰的正則化も見る）。

!!! note "学習の流れ"
    1. 学習データをシャッフルし、$B$ 個ずつのミニバッチに分ける。
    2. ミニバッチごとに 順伝播 → 損失 → [誤差逆伝播法](backprop.md) → パラメータ更新、を繰り返す。
    3. 全ミニバッチを1周したら **1エポック**。1エポックの更新回数は $\lceil N/B\rceil$。エポックの終わりに検証データで評価し、学習率の調整や[早期終了](regularization.md#early-stopping)を行う。

---

## 悪条件（Pathological Curvature）と学習率 {#curvature}

損失が谷の形で、方向によって曲がり方が大きく違うと、勾配は谷底ではなく**谷の壁**を向く。2次関数 $L=\frac12\sum_i\lambda_i\theta_i^2$ なら、SGD の更新は成分ごとに独立で

$$
\theta_i\leftarrow(1-\eta\lambda_i)\,\theta_i
$$

- 収束には、すべての成分で $|1-\eta\lambda_i|<1$、つまり $\eta<2/\lambda_{\max}$ が必要。
- 曲がりの小さい方向（$\lambda_{\min}$）は、$1-\eta\lambda_{\min}\approx1$ で**ほとんど進まない**。曲がりの大きい方向は $1-\eta\lambda_{\max}<0$ に近づくと**符号を変えながら振動（ジグザグ）**する。
- 最良の学習率は $\eta=\dfrac{2}{\lambda_{\max}+\lambda_{\min}}$ で、1ステップの縮み率は $\dfrac{\kappa-1}{\kappa+1}$（$\kappa=\lambda_{\max}/\lambda_{\min}$ は**条件数**）。$\kappa$ が大きいほど遅い。
- 学習率を下げれば振動は収まるが、遅い方向がさらに遅くなる。この板挟みを和らげるのが、次の Momentum と適応的な学習率。

---

## Momentum と Nesterov {#momentum}

$$
\begin{aligned}
&\text{Momentum} && \mathbf{v}\leftarrow\beta\,\mathbf{v}-\eta\,\mathbf{g}(\boldsymbol{\theta}),\qquad \boldsymbol{\theta}\leftarrow\boldsymbol{\theta}+\mathbf{v} \\[2mm]
&\text{Nesterov} && \mathbf{v}\leftarrow\beta\,\mathbf{v}-\eta\,\mathbf{g}(\boldsymbol{\theta}+\beta\mathbf{v}),\qquad \boldsymbol{\theta}\leftarrow\boldsymbol{\theta}+\mathbf{v}
\end{aligned}
$$

- $\mathbf{v}$：**速度**（過去の更新量の指数移動和）、$\beta\in[0,1)$：**モメンタム係数**（0.9 など。$\alpha$ と書く本もある）。
- 勾配が同じ向きに続くと、速度は $\mathbf{v}=-\dfrac{\eta}{1-\beta}\mathbf{g}$ まで溜まり、**歩幅が最大 $1/(1-\beta)$ 倍**（$\beta=0.9$ なら 10 倍）になる。向きが反転する方向（ジグザグ）では打ち消し合って小さくなる。
- **Nesterov**：まず慣性で進む位置 $\boldsymbol{\theta}+\beta\mathbf{v}$ へ**先読み**し、そこでの勾配で修正する。行きすぎを早く察知できるため、Momentum より振動しにくい。
- 2次関数での安定条件は、Momentum が $\eta\lambda<2(1+\beta)$、Nesterov が $\eta\lambda<\dfrac{2(1+\beta)}{1+2\beta}$（$\beta=0.9$ で 3.8 と約 1.36）。

---

## 適応的な学習率 {#adaptive}

勾配の2乗を成分ごとに記録し、**よく動く（勾配が大きい）成分は歩幅を小さく、動きの少ない成分は大きく**する。以下の積 $\odot$ と除算・平方根は成分ごと。$\epsilon$ は 0 除算を防ぐ小さな定数（$10^{-8}$ など）。

$$
\begin{aligned}
&\text{AdaGrad} && \mathbf{h}\leftarrow\mathbf{h}+\mathbf{g}\odot\mathbf{g},\qquad \boldsymbol{\theta}\leftarrow\boldsymbol{\theta}-\eta\,\frac{\mathbf{g}}{\sqrt{\mathbf{h}}+\epsilon} \\[3mm]
&\text{RMSprop} && \mathbf{h}\leftarrow\rho\,\mathbf{h}+(1-\rho)\,\mathbf{g}\odot\mathbf{g},\qquad \boldsymbol{\theta}\leftarrow\boldsymbol{\theta}-\eta\,\frac{\mathbf{g}}{\sqrt{\mathbf{h}}+\epsilon}
\end{aligned}
$$

- **AdaGrad**：$\mathbf{h}$ は勾配2乗の**累積和**。学習率を手で調整しなくてよいが、$\mathbf{h}$ は増える一方なので**見かけの学習率が単調に小さくなり、途中で止まる**。
- **RMSprop**：累積の代わりに**指数移動平均**（$\rho=0.9$ など）にして、古い履歴を忘れる。AdaGrad の「学習率が減り続ける」欠点を直す。勾配が小さい領域では $\mathbf{h}$ が減って歩幅が戻るので、プラトーを抜けやすい。
- 元の記事は RMSprop だけ $\epsilon$ を根号の中（$\sqrt{\mathbf{h}+\epsilon}$）に入れていた。実装によってどちらもあるが、ページ内では AdaGrad と同じ $\sqrt{\mathbf{h}}+\epsilon$ に揃えた。

### Adam {#adam}

Momentum の**1次モーメント**（勾配の移動平均）と、RMSprop の**2次モーメント**（勾配2乗の移動平均）を両方持つ。$t=1,2,\dots$ はステップ数。

$$
\begin{aligned}
\mathbf{m}&\leftarrow\beta_1\mathbf{m}+(1-\beta_1)\,\mathbf{g},&\qquad
\mathbf{v}&\leftarrow\beta_2\mathbf{v}+(1-\beta_2)\,\mathbf{g}\odot\mathbf{g} \\[2mm]
\hat{\mathbf{m}}&=\frac{\mathbf{m}}{1-\beta_1^{\,t}},&\qquad
\hat{\mathbf{v}}&=\frac{\mathbf{v}}{1-\beta_2^{\,t}} \\[2mm]
\boldsymbol{\theta}&\leftarrow\boldsymbol{\theta}-\eta\,\frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}}+\epsilon}
\end{aligned}
$$

- 推奨値：$\eta=10^{-3},\ \beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$。元のノートには $\beta_2=0.99$ と書いたものがあったが、原論文の推奨は **0.999**。
- ここでの $\mathbf{v}$ は Momentum の速度ではなく、勾配2乗の移動平均（別の量）。
- **バイアス補正**：$\mathbf{m},\mathbf{v}$ は 0 で初期化するので、初期は 0 に引っ張られて小さい。$1-\beta^t$ で割って、初期の偏りを取り除く。$t$ が大きいと $\beta^t\to0$ で、補正は効かなくなる。
- 勾配が一定の特殊な場合を考えると、補正がないと 1 ステップ目から $\dfrac{m}{\sqrt{v}}=\dfrac{1-\beta_1}{\sqrt{1-\beta_2}}\approx3.2$ 倍になる。補正すると $\hat m/\sqrt{\hat v}=1$。最初のステップは、勾配の大きさによらず $\eta\cdot\mathrm{sign}(\mathbf{g})$ になる。

<div class="ew" data-widget="opt-adam-bias"></div>

!!! tip "動かしてみる"
    - $t=1$ では、補正なしの値が約 3.16 になります。$m_1=0.1g$、$v_1=0.001g^2$ で、根号をとっても $\sqrt{v_1}\approx0.032g$ が $m_1$ より小さく、分母が小さすぎて歩幅が大きくなるためです。補正ありは常に 1 です。
    - $\beta_2$ を 0.9 に下げると、$v$ の補正が早く終わり、補正なしの曲線も 1 に近づきます。$\beta_2$ を 0.9999 にすると、補正がないまま大きな歩幅が長く続きます。
    - $t$ を 100 まで動かしても、$\beta_2=0.999$ ではまだ補正なしが 1 に戻りません（$1-0.999^{100}\approx0.095$）。

### AdamW と Lion

$$
\begin{aligned}
&\text{AdamW} && \boldsymbol{\theta}\leftarrow\boldsymbol{\theta}-\eta\left(\frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}}+\epsilon}+\lambda\,\boldsymbol{\theta}\right) \\[3mm]
&\text{Lion} && \mathbf{c}=\beta_1\mathbf{m}+(1-\beta_1)\,\mathbf{g},\qquad \boldsymbol{\theta}\leftarrow\boldsymbol{\theta}-\eta\bigl(\mathrm{sign}(\mathbf{c})+\lambda\,\boldsymbol{\theta}\bigr),\qquad \mathbf{m}\leftarrow\beta_2\mathbf{m}+(1-\beta_2)\,\mathbf{g}
\end{aligned}
$$

- **AdamW**：Adam の $\mathbf{m},\hat{\mathbf{v}}$ の計算は同じで、**重み減衰を勾配ではなく更新式に直接入れる**（decoupled weight decay）。損失に $L_2$ 項を足して勾配に $\lambda\boldsymbol\theta$ を混ぜると、適応的な割り算を通ってしまい、勾配の大きい重みほど減衰が弱くなる。AdamW は全重みを同じ割合で減衰させる。Transformer の標準的な最適化手法。
- **Lion**：更新量は $\mathrm{sign}(\mathbf{c})$、つまり**各成分が必ず $\pm\eta$**（大きさの情報を捨てる）。保持するのは $\mathbf{m}$ だけ（Adam は $\mathbf{m},\mathbf{v}$）でメモリが少ない。$\beta_1=0.9,\ \beta_2=0.99$ が基本。
- 元の記事の Lion の式は、$\beta_1$ と $\beta_2$ の役割が逆だった。原論文では、**更新の向き $\mathbf{c}$ を $\beta_1$ で作り、モメンタム $\mathbf{m}$ の更新に $\beta_2$ を使う**（$\mathbf{c}$ は更新前の $\mathbf{m}$ から作る）。上の式はそれに直した。

---

## 手法を並べて比べる {#compare}

| 手法 | 持つ状態 | 歩幅の決まり方 | 一言 |
|:--|:--|:--|:--|
| SGD | なし | $\eta\mathbf{g}$ | 基本。悪条件でジグザグ |
| Momentum | 速度 $\mathbf{v}$ | 過去の勾配の和 | 一方向なら加速、振動は相殺 |
| Nesterov | 速度 $\mathbf{v}$ | 先読み位置の勾配 | Momentum より振動が少ない |
| AdaGrad | $\mathbf{h}$（和） | $\eta/\sqrt{\mathbf{h}}$ | 学習率は減る一方 |
| RMSprop | $\mathbf{h}$（移動平均） | $\eta/\sqrt{\mathbf{h}}$ | 履歴を忘れる |
| Adam | $\mathbf{m},\mathbf{v}$ | $\eta\hat{\mathbf{m}}/\sqrt{\hat{\mathbf{v}}}$ | Momentum + RMSprop + 補正 |
| AdamW | $\mathbf{m},\mathbf{v}$ | Adam + 分離した重み減衰 | Transformer で標準 |
| Lion | $\mathbf{m}$ | $\eta\,\mathrm{sign}(\mathbf{c})$ | 省メモリ |

細長い谷 $L=\frac12(\theta_1^2+\kappa\,\theta_2^2)$ を、同じ出発点から下る。ボタンで手法を選ぶと軌跡が紺、ほかの手法の軌跡がグレーで表示される。$\beta$ は Momentum・Nesterov の係数、RMSprop の $\rho$、Adam の $\beta_1$ に共通で使う（Adam の $\beta_2$ は 0.999 に固定）。

<div class="ew" data-widget="opt-descent"></div>

!!! tip "動かしてみる"
    - SGD で $\eta$ を 0.17 近くまで上げると、$\theta_2$ 方向で符号が反転しながらジグザグします（$1-\eta\kappa$ が $-0.7$ 付近）。$\eta=0.2$（$=2/\kappa$）を超えると発散します。
    - $\kappa$ を 1 にすると円形の谷になり、どの手法もまっすぐ進みます。$\kappa$ を大きくするほど SGD は遅くなります。
    - Momentum は $\beta$ を上げると加速しますが、行きすぎて原点のまわりを回ります。Nesterov は同じ $\eta,\beta$ でも振れ幅が小さいですが、$\eta=0.15$（$\kappa=10$）にすると Momentum が安定のまま Nesterov だけ発散します。
    - AdaGrad は歩幅が縮み続け、$t=60$ でもまだ原点に届きません。RMSprop と Adam は成分ごとに歩幅をそろえるため、谷の向きに関係なく進みます（ただし Adam は終盤に $\eta$ 程度の振れが残ります）。

---

## 学習率のスケジュール {#lr-schedule}

学習率 $\eta$ を一定にせず、**最初は大きく、終盤は小さく**する。大きい間は速く進み、小さい間は谷底に収まる。

| 方法 | 式 | 特徴 |
|:--|:--|:--|
| ステップ減衰 | $\eta_t=\eta_0\,\gamma^{\lfloor t/s\rfloor}$ | $s$ ステップごとに $\gamma$ 倍（例 0.1） |
| 指数減衰 | $\eta_t=\eta_0\,e^{-kt}$ | なめらかに減る |
| コサイン | $\eta_t=\eta_{\min}+\tfrac12(\eta_0-\eta_{\min})\bigl(1+\cos\frac{\pi t}{T}\bigr)$ | $\eta_0$ から $\eta_{\min}$ へなめらかに下げる。周期的に戻すと**ウォームリスタート** |
| ウォームアップ | 最初の $T_w$ ステップで 0 から $\eta_0$ まで直線的に上げる | 初期の不安定さを避ける。Adam・Transformer で必須に近い |

- Transformer の元論文のスケジュールは $\eta_t=d_{\mathrm{model}}^{-1/2}\min\bigl(t^{-1/2},\,t\,T_w^{-3/2}\bigr)$。$t=T_w$ で最大になり、前は直線的に増加、後は $t^{-1/2}$ で減少する（[Transformer](transformer.md)）。
- バッチサイズを $k$ 倍にしたら、学習率も約 $k$ 倍にする経験則がある（線形スケーリング則）。

---

## 勾配クリッピング {#clipping}

勾配のノルムが大きくなりすぎる（**勾配爆発**）と、1回の更新で損失が跳ね上がる。特に RNN で起きやすい。全パラメータの勾配を1本のベクトル $\mathbf{g}$ にまとめ、ノルムがしきい値 $v$ を超えたときだけ縮める。

$$
\text{if } \|\mathbf{g}\|>v:\qquad \mathbf{g}\leftarrow\frac{v}{\|\mathbf{g}\|}\,\mathbf{g},\qquad \|\mathbf{g}\|=\sqrt{\sum_{l}\left\|\frac{\partial L}{\partial\mathbf{W}^{(l)}}\right\|_F^2+\left\|\frac{\partial L}{\partial\mathbf{b}^{(l)}}\right\|^2}
$$

- 向きは変えずに**長さだけ** $v$ に揃えるのがポイント。要素ごとに値を切る方法（値クリッピング）もあるが、向きが変わる。
- 勾配爆発には clipping、**勾配消失**には活性化関数・初期化・残差接続・LSTM などで対処する（[活性化関数](activation.md)、[初期化](init.md)、[RNN](rnn.md)）。

---

## 学習が進みにくい形

- **局所最適解**：周りより低いが最小ではない点。高次元では、悪い局所解よりも鞍点が問題になるとされる。
- **鞍点**：勾配が 0 でも、ある方向には上り、別の方向には下る点。SGD のノイズや Momentum が抜け出す助けになる。
- **プラトー**：勾配がほぼ 0 の平らな領域。適応的な手法（RMSprop・Adam）は歩幅を広げて抜けやすい。

---

## 試験の着眼点

- **更新式を書ける**ようにする：SGD $\boldsymbol\theta-\eta\mathbf g$、Momentum $\mathbf v=\beta\mathbf v-\eta\mathbf g$ と $\boldsymbol\theta+\mathbf v$、AdaGrad の $\mathbf h$ の累積和。
- Nesterov は**先読み**した位置 $\boldsymbol\theta+\beta\mathbf v$ の勾配。Momentum は今の位置の勾配。
- AdaGrad の欠点は学習率が単調に減ること。RMSprop は指数移動平均で解決。
- Adam は $\mathbf m$（1次）と $\mathbf v$（2次）の**バイアス補正**を持つ。推奨値は $0.9,\ 0.999,\ 10^{-8}$。
- AdamW は重み減衰を**勾配から分離**する。Lion は符号だけを使う。
- ミニバッチの勾配のばらつきは $1/B$ に比例。バッチを大きくすると、並列で速いが、汎化が悪くなりやすい。
- 条件数が大きい（悪条件）と SGD が遅い。許される学習率は $\eta<2/\lambda_{\max}$。
- 勾配クリッピングは、勾配のノルムが $v$ を超えたときに $v/\|\mathbf g\|$ を掛ける。

---

## 参考

- [Adam: A Method for Stochastic Optimization（Kingma, Ba, 2015）](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization（Loshchilov, Hutter, 2019）](https://arxiv.org/abs/1711.05101)：AdamW
- [Symbolic Discovery of Optimization Algorithms（Chen ほか, 2023）](https://arxiv.org/abs/2302.06675)：Lion
- [SGDR: Stochastic Gradient Descent with Warm Restarts（Loshchilov, Hutter, 2017）](https://arxiv.org/abs/1608.03983)
- [On the difficulty of training recurrent neural networks（Pascanu ほか, 2013）](https://arxiv.org/abs/1211.5063)：勾配クリッピング
- [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization（Dauphin ほか, 2014）](https://arxiv.org/abs/1406.2572)
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour（Goyal ほか, 2017）](https://arxiv.org/abs/1706.02677)：線形スケーリング則
- [Attention Is All You Need（Vaswani ほか, 2017）](https://arxiv.org/abs/1706.03762)：ウォームアップ付きスケジュール
- [Deep Learning, Chapter 8 Optimization for Training Deep Models（Goodfellow, Bengio, Courville, 2016）](https://www.deeplearningbook.org/contents/optimization.html)
