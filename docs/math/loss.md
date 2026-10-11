# 損失関数まとめ

<p class="kw">キーワード：平均二乗誤差（MSE）、平均絶対誤差（MAE）、Huber損失、負の対数尤度、交差エントロピー、バイナリクロスエントロピー、重み付き交差エントロピー、ラベル平滑化、Focal Loss、KLダイバージェンス、温度つきソフトマックス、パープレキシティ、Contrastive Loss、Triplet Loss、InfoNCE、ランキング損失、順序回帰</p>

!!! abstract "要点"
    - 損失関数は「**予測と正解のズレ**を1つの数にして、最小化する」ための関数。多くは**確率モデルの負の対数尤度**として導ける（正規分布→MSE、ラプラス分布→MAE、カテゴリカル分布→交差エントロピー）。
    - 回帰は**残差の形**（2乗・絶対値・Huber）で外れ値への強さが決まり、分類は**正解クラスの確率**（交差エントロピー・Focal）や**マージン**（ヒンジ）で測る。
    - 距離学習・ランキングは、ペアや3つ組の**相対的な関係**を損失にする。正解ラベルそのものは使わない。

出力層との組み合わせや逆伝播での勾配（デルタ）は [多層パーセプトロン・出力層と損失関数](../dl/mlp-loss.md)、情報理論の定義は [情報理論](information-theory.md) を見る。ここでは損失の**式と性質を1か所に集める**。

記号：正解 $\mathbf{t}$（または $t$）、予測 $\mathbf{y}$（または $y$）、サンプル数 $N$、クラス数 $K$。残差は $r=y-t$。

---

## 損失の見取り図 {#overview}

| 分野 | 損失 | 一言 |
|:--|:--|:--|
| 回帰 | MSE、MAE、Huber、Smooth L1 | 残差の大きさを測る |
| 2値・多クラス分類 | BCE、交差エントロピー、重み付き、ラベル平滑化、Focal | 正解クラスの確率を高くする |
| マージン基準の分類 | ヒンジ、ロジスティック、指数 | 正しく分類できた「余裕」を測る |
| 確率分布の比較 | KLダイバージェンス、負の対数尤度 | 2つの分布のズレ |
| 言語モデル | 言語モデル損失、パープレキシティ | 次の単語の予測のしやすさ |
| 距離学習・対照学習 | Contrastive、Triplet、InfoNCE | 似たものは近く、違うものは遠く |
| ランキング・順序回帰 | ペアワイズ、累積 BCE | 順序を正しく並べる |

- **損失（学習で最小にする、微分できる）と評価指標（正解率・F値など）は別物**。正解率は微分できないので、代わりに滑らかな損失を最小にする（[評価指標](../ml/evaluation.md)）。

---

## 回帰 {#regression}

$$
\begin{aligned}
&\text{MSE} && L=\frac1N\sum_{i=1}^{N}(y_i-t_i)^2,\qquad \frac{\partial L}{\partial y_i}=\frac2N(y_i-t_i) \\[2mm]
&\text{MAE} && L=\frac1N\sum_{i=1}^{N}|y_i-t_i|,\qquad \frac{\partial L}{\partial y_i}=\frac1N\operatorname{sign}(y_i-t_i) \\[2mm]
&\text{Huber} && L_\delta(r)=\begin{cases}\tfrac12r^2&(|r|\le\delta)\\[1mm]\delta|r|-\tfrac12\delta^2&(|r|>\delta)\end{cases},\qquad \frac{\partial L_\delta}{\partial r}=\begin{cases}r&(|r|\le\delta)\\ \delta\operatorname{sign}(r)&(|r|>\delta)\end{cases} \\[2mm]
&\text{RMSE} && \sqrt{\mathrm{MSE}} \\[2mm]
&R^2 && 1-\frac{\sum_i(t_i-y_i)^2}{\sum_i(t_i-\bar t)^2}
\end{aligned}
$$

| 損失 | 性質 |
|:--|:--|
| MSE | 誤差の**2乗**。大きい誤差を強く罰するので**外れ値に弱い**。最適な予測は条件付き**平均** |
| MAE | 誤差の絶対値。**外れ値に強い**。最適な予測は条件付き**中央値**。勾配の大きさが常に一定で、$r=0$ で微分できない |
| Huber | $\lvert r\rvert\le\delta$ では MSE（なめらか）、それより大きいと MAE（勾配が $\delta$ で頭打ち）。切り替わりを $\delta$ で決める。両者の中間 |
| Smooth L1 | Huber で $\delta=1$ としたもの（$\tfrac12r^2$ と $\lvert r\rvert-\tfrac12$）。物体検出の位置回帰で使う。PyTorch の $\beta$ つきの式は Huber の $1/\beta$ 倍 |
| RMSE | MSE の平方根。**目的変数と同じ単位**で読める |
| $R^2$ | 平均で予測した場合に比べて、誤差が何割減ったか。1 が最良で、**平均より悪い予測だと負にもなる**。目的変数のスケールに依存しない |

- MSE を学習に使うとき、微分をきれいにするため $\tfrac12$ を付けて $\tfrac12\sum(y-t)^2$ とすることが多い（勾配が $y-t$）。このページの MSE は $\tfrac1N\sum$ の定義で、$\tfrac12$ の有無で勾配の係数が $2/N$ か $1/N$ か変わる。
- **L2 正則化つき線形回帰**：予測 $y_n=\mathbf{w}^\top\mathbf{x}_n$、損失 $L=\tfrac12\sum_{n}(y_n-t_n)^2+\lambda\sum_i w_i^2$ の勾配は $\nabla_{\mathbf{w}}L=\sum_n\mathbf{x}_n(y_n-t_n)+2\lambda\mathbf{w}$（[線形回帰](../ml/linear-regression.md)、[正則化](../dl/regularization.md)）。

<div class="ew" data-widget="loss-regression"></div>

!!! tip "動かしてみる"
    - $r$ を大きくすると、MSE の勾配は $r$ に比例して増え続け、MAE は $\pm1$ で一定、Huber は $\delta$ で頭打ちになります。**外れ値**（大きな $r$）の影響が MSE だけ膨らむことが分かります。
    - $\delta$ を大きくすると、Huber の山が MSE に近づきます。$\delta$ を小さくすると MAE に近づきます。
    - $r=0$ の付近では、MAE の勾配が $-1$ から $+1$ に飛びます（最小点で滑らかでない）。

---

## 分類 {#classification}

### 負の対数尤度と交差エントロピー

$$
\begin{aligned}
&\text{負の対数尤度（NLL）} && L=-\log p(t\mid\mathbf{x}) \\[2mm]
&\text{交差エントロピー} && L=-\sum_{k=1}^{K}t_k\log y_k,\qquad \frac{\partial L}{\partial u_k}=y_k-t_k\ \ \bigl(y_k=\mathrm{softmax}(\mathbf{u})_k\bigr) \\[2mm]
&\text{バイナリクロスエントロピー} && L=-\bigl[t\log y+(1-t)\log(1-y)\bigr],\qquad \frac{\partial L}{\partial u}=y-t\ \ \bigl(y=\sigma(u)\bigr)
\end{aligned}
$$

- 正解が one-hot（正解クラス $c$）なら、交差エントロピーは $-\log y_c$、つまり NLL と同じ。**正解クラスに割り当てた確率だけ**で決まる。
- 平均をとる損失では、勾配に $1/N$ が付く。1 サンプルの式では付かない。
- ロジット（ソフトマックスの入力）についての勾配が $y-t$ になることが、分類で交差エントロピーを使う大きな理由（[多層パーセプトロン・出力層と損失関数](../dl/mlp-loss.md#classification-loss)）。
- **logloss** も同じ式（BCE や交差エントロピーの平均）。範囲は $[0,\infty)$ で、確信して間違えるほど大きい。

### クラス不均衡への対策

$$
\begin{aligned}
&\text{重み付き交差エントロピー} && L=-\sum_{k=1}^{K}w_k\,t_k\log y_k,\qquad w_k=\frac{N}{K\,n_k} \\[2mm]
&\text{Focal Loss} && L=-\alpha_t\,(1-y_t)^{\gamma}\log y_t,\qquad y_t=\text{正解クラスの予測確率}
\end{aligned}
$$

- 重みは**クラスの個数 $n_k$ に反比例**させ、少数クラスの誤りを重く扱う。$w_k=N/(K n_k)$ なら、サンプルあたりの**平均の重みが 1**（$\sum_kn_kw_k/N=1$）になる。
- Focal Loss は、**易しい例**（$y_t$ が大きい例）の損失を $(1-y_t)^\gamma$ 倍して減らし、難しい例に学習を集中させる。$\gamma=0$ なら交差エントロピーに一致。$\alpha_t$ はクラスごとの重み。この $\gamma$ は強化学習の割引率とは別物。

<div class="ew" data-widget="loss-focal"></div>

!!! tip "動かしてみる"
    - $\gamma=2$、$p_t=0.9$（易しい例）では、変調項が $0.01$ になり、損失は交差エントロピーの約 $1/100$ になります。
    - $p_t$ を小さく（難しい例）すると、変調項が 1 に近づき、交差エントロピーとほぼ同じ損失です。**易しい例だけが大きく減る**ので、たくさんある易しい例に学習が引きずられません。
    - $\gamma=0$ にすると、2本の曲線が重なります。

### ラベル平滑化と温度

$$
\begin{aligned}
&\text{ラベル平滑化} && t_k^{\mathrm{LS}}=\begin{cases}1-\epsilon+\epsilon/K&(k=c)\\ \epsilon/K&(k\ne c)\end{cases} \\[2mm]
& && L_{\mathrm{LS}}=-\sum_kt_k^{\mathrm{LS}}\log y_k=-(1-\epsilon)\log y_c-\frac{\epsilon}{K}\sum_{k=1}^{K}\log y_k \\[2mm]
&\text{温度つきソフトマックス} && y_k=\frac{e^{u_k/T}}{\sum_je^{u_j/T}}
\end{aligned}
$$

- ラベル平滑化は、正解を少しだけ他のクラスに配ることで**過信を抑える**。損失は「正解クラスの通常の交差エントロピーの $(1-\epsilon)$ 倍」と「一様分布との交差エントロピーの $\epsilon$ 倍」の和（詳しくは [出力層と損失関数](../dl/mlp-loss.md#label-smoothing)）。
- 温度 $T>1$ で分布が平らに、$T<1$ で尖る（[活性化関数](../dl/activation.md)）。学習後に $T$ だけを調整して確率の過信を直す**温度スケーリング**（校正）や、**知識蒸留**で使う。

### マージン基準の損失（SVM・ブースティングの系統）

2 値分類で正解を $t\in\{-1,+1\}$、スコアを $f(\mathbf{x})$ とし、**マージン** $m=t\,f(\mathbf{x})$ とする（$m>0$ なら正しく分類、大きいほど余裕がある）。

| 損失 | $\ell(m)$ | $d\ell/dm$ | 使われる場面 |
|:--|:--|:--|:--|
| 0-1 損失 | $\mathbb{1}[m<0]$ | ほぼ 0（微分に使えない） | 本来の誤り数。直接は最小化できない |
| ヒンジ | $\max(0,\,1-m)$ | $-\mathbb{1}[m<1]$ | SVM |
| 2乗ヒンジ | $\max(0,\,1-m)^2$ | $-2\max(0,1-m)$ | 滑らかな SVM |
| ロジスティック | $\ln(1+e^{-m})$ | $-\sigma(-m)$ | ロジスティック回帰（BCE と同じ） |
| 指数 | $e^{-m}$ | $-e^{-m}$ | AdaBoost |

- 0-1 損失は微分できないので、代わりに**滑らかな関数（代理損失）**を最小にする。いずれも $m$ が大きくなると損失が小さくなる。ヒンジ・2乗ヒンジ・指数、および $\log_2(1+e^{-m})$（ロジスティックを $\ln2$ で割ったもの）は、0-1 損失を**上から押さえる**。
- ヒンジは $m\ge1$ で損失が**ちょうど 0**（余裕を持って正しい点は無視される → サポートベクトルだけが効く）。ロジスティックは決して 0 にならない。指数は $m$ が負で急激に大きくなり、外れ値に敏感。
- ロジスティック損失は、$y=\sigma(f)$ としたときの BCE に等しい（$t=+1$ なら $-\ln\sigma(f)=\ln(1+e^{-f})$）。

<div class="ew" data-widget="loss-margin"></div>

!!! tip "動かしてみる"
    - $m$ を 1 より大きくすると、ヒンジの損失と勾配が 0 になります。ロジスティックと指数はなだらかに 0 へ近づくだけです。
    - $m$ を負にする（誤分類）と、指数損失だけ急に大きくなります。
    - 0-1 損失（黒い階段）は、ヒンジ・2乗ヒンジ・指数の曲線よりも常に下にあります（上から押さえられている）。ロジスティックだけは $\ln$ のままなので、$m$ が 0 より少し小さい範囲で、階段の高さ 1 をわずかに下回ります。

---

## 確率分布の比較と言語モデル {#distribution}

$$
\begin{aligned}
&\text{KLダイバージェンス} && D_{\mathrm{KL}}(P\|Q)=H(P,Q)-H(P) \\[2mm]
&\text{言語モデル損失} && L_{\mathrm{LM}}=-\frac1S\sum_{s=1}^{S}\log p_\theta\bigl(w_s\mid w_1,\dots,w_{s-1}\bigr) \\[2mm]
&\text{パープレキシティ} && \mathrm{PPL}=\exp\bigl(L_{\mathrm{LM}}\bigr)
\end{aligned}
$$

- 正解分布 $P$ が one-hot なら $H(P)=0$ で、KL は交差エントロピー（NLL）に一致する。ソフトな正解（蒸留、ラベル平滑化）では $H(P)$ は定数なので、最小化は同じ（[情報理論](information-theory.md#kl)）。
- 言語モデルは、前の単語列から次の単語の確率を予測する**自己回帰**。損失は**全位置の交差エントロピーの平均**（$S$：系列長）。
- パープレキシティは、**平均して何択から選ぶ迷い**に相当する。語彙 $V$ 個から一様に選ぶモデルは $L_{\mathrm{LM}}=\ln V$ で $\mathrm{PPL}=V$、完璧な予測は $\mathrm{PPL}=1$。**小さいほどよい**。対数の底が 2 なら $2^{H}$（底と指数の底をそろえる）。
- 交差エントロピーを最小にすることと、パープレキシティを最小にすることは同じ。

---

## 距離学習・対照学習 {#metric-learning}

埋め込みベクトル間の距離 $d$ を、**似たものは近く、違うものは遠く**なるように学習する。

$$
\begin{aligned}
&\text{Contrastive} && L=\frac12\Bigl[Y\,d^2+(1-Y)\max(0,\,m-d)^2\Bigr]\qquad(Y=1\text{：類似ペア},\ Y=0\text{：非類似ペア}) \\[2mm]
&\text{Triplet} && L=\max\bigl(0,\ d(a,p)^2-d(a,n)^2+\alpha\bigr) \\[2mm]
&\text{InfoNCE} && L=-\log\frac{\exp\bigl(\mathrm{sim}(a,p)/\tau\bigr)}{\sum_{j=1}^{K}\exp\bigl(\mathrm{sim}(a,e_j)/\tau\bigr)}
\end{aligned}
$$

- **Contrastive**：類似ペアは距離 $d$ を小さくし、非類似ペアは距離がマージン $m$ 以上になるまで引き離す（$d\ge m$ なら損失 0）。ラベルの向き（類似を 0 とする流儀）は文献で逆のものもあるので、定義を確かめる。
- **Triplet**：基準 $a$（アンカー）、類似 $p$（ポジティブ）、非類似 $n$（ネガティブ）。「$a$ と $p$ の距離」＋ マージン $\alpha$ が「$a$ と $n$ の距離」より小さくなれば損失 0。距離を2乗にしない $\max(0,d(a,p)-d(a,n)+\alpha)$ の形もある。
- **InfoNCE**：アンカーに対して、**正例 $p$ 1 個**と**負例**を合わせた $K$ 個の候補 $e_j$（正例を含む）から正例を当てる $K$ クラスのソフトマックス交差エントロピー。$\tau$ は温度、$\mathrm{sim}$ はコサイン類似度など。
- **インバッチネガティブ**：同じミニバッチ内の他のサンプルを負例として使う。追加の計算なしに負例が増える。バッチが大きいほど $K$ が増えて性能が上がりやすい。
- InfoNCE は、アンカーと正例の**相互情報量の下界**になっている：$I(a;p)\ge\log K-L_{\mathrm{InfoNCE}}$。$K$ を増やすと、下界がより高い値まで測れる（[情報理論](information-theory.md#mutual-information)）。

---

## ランキング・順序回帰 {#ranking}

### ペアワイズ・ランキング損失

$$
\begin{aligned}
&\text{マージンランキング} && L=\max\bigl(0,\ \text{margin}-s_{\mathrm{pos}}+s_{\mathrm{neg}}\bigr) \\[2mm]
&\text{RankNet 型} && L=\log\bigl(1+e^{-(s_i-s_j)}\bigr)\qquad(i\text{ が }j\text{ より上位であるべきペア})
\end{aligned}
$$

- 順位をつけたい 2 つの項目のスコア $s_{\mathrm{pos}}$（上位であるべき）と $s_{\mathrm{neg}}$ の**差**だけを見る。正例のスコアが負例より **margin 以上高ければ損失 0**。
- RankNet 型は、スコア差を $\sigma(s_i-s_j)$ として「$i$ が上位である確率」と見て交差エントロピーを使う形（ヒンジをロジスティックに替えたもの）。
- **Triplet Loss も同じ形**で、ランキングの一種とみなせる。

### 順序回帰（累積 BCE）

ラベルに順序がある（評価 1〜5 など）とき、順位 $c\in\{1,\dots,K\}$ を「$c>k$ か？」という $K-1$ 個の 2 値問題に分け、各 BCE を足す。

$$
L=-\sum_{k=1}^{K-1}\Bigl[t_k\log y_k+(1-t_k)\log(1-y_k)\Bigr],\qquad t_k=\mathbb{1}[c>k],\quad y_k=\sigma\bigl(g(\mathbf{x})-\theta_k\bigr)
$$

- 詳しい導入と予測の方法は [多層パーセプトロン・出力層と損失関数](../dl/mlp-loss.md#ordinal)。閾値ごとの重みを共有する CORAL の考え方で、順序の矛盾を防げる。
- 元の記事にあった「$-\frac1N\sum\log\frac1{\mathrm{rank}_i}$」は、$\frac1N\sum\log\mathrm{rank}_i$ と同じで、**予測がどこにも現れず**、学習に使える損失になっていない。また、**ランキング損失（順位づけの学習）と順序回帰（順序つきラベルの予測）は別のタスク**。ここでは累積 BCE に置き換えた。

---

## 正則化項と、他のページの損失 {#others}

$$
L_{\mathrm{total}}=L_{\mathrm{data}}(\mathbf{w})+\lambda\,\Omega(\mathbf{w}),\qquad \Omega=\|\mathbf{w}\|_2^2\ \ (\text{L2}),\quad \|\mathbf{w}\|_1\ \ (\text{L1})
$$

- 正則化項 $\Omega$ を足すと、重みを小さくする力がはたらく。L2 では勾配に $2\lambda\mathbf{w}$ が加わり、これが**重み減衰**（[正則化](../dl/regularization.md)）。

| 場面 | 損失 | 参照 |
|:--|:--|:--|
| VAE | $-\text{ELBO}=-\mathbb{E}_{q}\bigl[\log p_\theta(\mathbf{x}\mid\mathbf{z})\bigr]+D_{\mathrm{KL}}\bigl(q_\phi(\mathbf{z}\mid\mathbf{x})\Vert p(\mathbf{z})\bigr)$（再構成誤差＋KL） | [生成モデル](../app/generative.md) |
| GAN | 識別器の BCE、生成器は識別器をだます向き | [生成モデル](../app/generative.md) |
| Q学習・TD（強化学習の批評家） | $\bigl(r+\gamma\max_{a'}Q(s',a')-Q(s,a)\bigr)^2$ の期待値 | [強化学習](../app/rl.md) |
| LIME（局所説明） | $\sum_z\pi_x(z)\bigl(f(z)-g(z')\bigr)^2+\Omega(g)$（近いサンプルほど重い2乗誤差＋モデルの複雑さ） | [説明性](../app/explainability.md) |
| 知識蒸留 | 教師と生徒の確率分布（温度つき）の KL | [情報理論](information-theory.md) |

---

## 試験の着眼点

- **MSE は外れ値に弱く、MAE は強い**。Huber は両者の中間（小さい残差では MSE、大きいと MAE）。Smooth L1 は $\delta=1$ の Huber。
- MSE の最適な予測は**平均**、MAE は**中央値**。
- 交差エントロピーは one-hot なら $-\log(\text{正解クラスの確率})$。ソフトマックス＋交差エントロピーの勾配は **$y-t$**（バッチ平均なら $1/N$ が付く）。
- BCE は $K=2$ の交差エントロピー。**マルチラベル**は BCE の和。
- 重み付き交差エントロピーの重みは**クラスの個数に反比例**（$N/(Kn_k)$）。Focal Loss は**易しい例の寄与を $(1-y_t)^\gamma$ で下げる**。
- ラベル平滑化は**過信の抑制**。温度 $T$ が大きいと確率が平ら。
- KL ＝ 交差エントロピー − エントロピー。one-hot の正解なら KL ＝ 交差エントロピー。
- パープレキシティ ＝ $\exp(\text{平均交差エントロピー})$。小さいほどよい。
- Contrastive は**類似ペアを近づけ、非類似ペアをマージンまで離す**。Triplet は $d(a,p)+\alpha<d(a,n)$ で損失 0。InfoNCE は**正例を当てる多クラス分類**。
- ヒンジ損失は $m\ge1$ で 0、SVM。指数損失は AdaBoost。ロジスティック損失は BCE。
- 順序回帰は「順位が $k$ より大きいか」の累積 BCE。ペアワイズのランキング損失とは別。

---

## 参考

- [Deep Learning, Chapter 6 Deep Feedforward Networks（Goodfellow, Bengio, Courville, 2016）](https://www.deeplearningbook.org/)：最尤推定と損失関数
- [Focal Loss for Dense Object Detection（Lin ほか, 2017）](https://arxiv.org/abs/1708.02002)
- [Rethinking the Inception Architecture for Computer Vision（Szegedy ほか, 2016）](https://arxiv.org/abs/1512.00567)：ラベル平滑化
- [On Calibration of Modern Neural Networks（Guo ほか, 2017）](https://arxiv.org/abs/1706.04599)：温度スケーリング
- [FaceNet: A Unified Embedding for Face Recognition and Clustering（Schroff ほか, 2015）](https://arxiv.org/abs/1503.03832)：Triplet Loss
- [Representation Learning with Contrastive Predictive Coding（van den Oord ほか, 2018）](https://arxiv.org/abs/1807.03748)：InfoNCE と相互情報量の下界
- [A Simple Framework for Contrastive Learning of Visual Representations（Chen ほか, 2020）](https://arxiv.org/abs/2002.05709)：インバッチネガティブ
- [Rank Consistent Ordinal Regression for Neural Networks（Cao, Mirjalili, Raschka, 2020）](https://arxiv.org/abs/1901.07884)：順序回帰（CORAL）
