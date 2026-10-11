# ロジスティック回帰

<p class="kw">キーワード：ロジット、シグモイド関数、ロジスティック関数、ソフトマックス関数、オッズ/オッズ比</p>

!!! abstract "要点"
    - 線形和 $z=\mathbf{w}^{\top}\mathbf{x}+b$（ロジット）を**シグモイド関数**に通して、クラス 1 の確率 $y=\sigma(z)$ を出す**分類**モデル。名前は回帰だが分類器である。
    - ベルヌーイ分布の**最尤推定**＝**交差エントロピー**の最小化。勾配は $\mathbf{X}^{\top}(\mathbf{y}-\mathbf{t})$ と単純で、損失は凸（大域的最小に収束する）。
    - **オッズの対数（ロジット）が入力の線形式**になる。係数の $\exp$ が**オッズ比**。多クラスへの拡張が**ソフトマックス関数**。

---

## 二値ロジスティック回帰 {#binary}

ラベル $t_n\in\{0,1\}$、入力 $\mathbf{X}\in\mathbb{R}^{N\times D}$、重み $\mathbf{w}\in\mathbb{R}^{D}$、バイアス $b$ とする。

$$
\begin{aligned}
&\text{ロジット} && z_n=\mathbf{w}^{\top}\mathbf{x}_n+b \\[2mm]
&\text{推論} && y_n=P(t=1\mid\mathbf{x}_n)=\sigma(z_n)=\frac{1}{1+e^{-z_n}} \\[2mm]
&\text{損失（負の対数尤度）} && L(\mathbf{w},b)=-\sum_{n=1}^{N}\Bigl[t_n\log y_n+(1-t_n)\log(1-y_n)\Bigr] \\[2mm]
&\text{勾配} && \frac{\partial L}{\partial \mathbf{w}}=\mathbf{X}^{\top}(\mathbf{y}-\mathbf{t}),\qquad \frac{\partial L}{\partial b}=\sum_{n=1}^{N}(y_n-t_n) \\[2mm]
&\text{更新式} && \mathbf{w}\leftarrow\mathbf{w}-\eta\,\mathbf{X}^{\top}(\mathbf{y}-\mathbf{t})
\end{aligned}
$$

??? note "勾配の導出"
    $L_n=-t\log y-(1-t)\log(1-y)$、$y=\sigma(z)$、$\dfrac{dy}{dz}=y(1-y)$ なので

    $$
    \frac{\partial L_n}{\partial z}=\Bigl(-\frac{t}{y}+\frac{1-t}{1-y}\Bigr)y(1-y)=-t(1-y)+(1-t)y=y-t
    $$

    $z=\mathbf{w}^{\top}\mathbf{x}+b$ より $\partial z/\partial\mathbf{w}=\mathbf{x}$ を掛けて、全サンプルの和をとると $\mathbf{X}^{\top}(\mathbf{y}-\mathbf{t})$。線形回帰の勾配 $\mathbf{X}^{\top}(\hat{\mathbf{y}}-\mathbf{y})$ と**同じ形**になる（「予測 − 正解」）。

| 記号 | 形 | 検算 |
|:--|:--|:--|
| $\mathbf{X}$ | $N\times D$ | — |
| $\mathbf{y}-\mathbf{t}$ | $N\times1$ | — |
| $\mathbf{X}^{\top}(\mathbf{y}-\mathbf{t})$ | $(D\times N)(N\times1)=D\times1$ | $\mathbf{w}$ と同じ形 |

- **判別**：$y\ge0.5\iff z\ge0$。判別境界 $\mathbf{w}^{\top}\mathbf{x}+b=0$ は**線形**（超平面）。しきい値を 0.5 以外にすると適合率と再現率のバランスが変わる（[評価](evaluation.md)）。
- 損失は $\mathbf{w}$ について**凸**。ただし線形回帰のような解析解はなく、勾配降下法、または**ニュートン法（IRLS）** $\mathbf{w}\leftarrow\mathbf{w}-(\mathbf{X}^{\top}\mathbf{R}\mathbf{X})^{-1}\mathbf{X}^{\top}(\mathbf{y}-\mathbf{t})$（$\mathbf{R}=\mathrm{diag}\bigl(y_n(1-y_n)\bigr)$）で解く。
- データが**線形分離可能**だと、$\|\mathbf{w}\|\to\infty$ で損失がいくらでも小さくなり、最尤解が定まらない。**L2 正則化**（$+\frac{\lambda}{2}\|\mathbf{w}\|^2$）で防ぐ。
- 二乗誤差を使わない理由：シグモイドと組み合わせると損失が非凸になり、$y$ が飽和したとき勾配が消える。交差エントロピーなら飽和しても $y-t$ が残る（[損失関数](../math/loss.md)）。

<div class="ew" data-widget="logistic-fit"></div>

!!! tip "動かしてみる"
    - 「最尤解にする」を押すと、$w\approx1.55,\ b\approx-0.08$ に移り、下の行の勾配 $(\partial L/\partial w,\ \partial L/\partial b)$ が**ほぼ (0, 0)** になります（最小点では勾配が 0）。
    - $w$ を大きくすると曲線が急な階段に近づきます。データの重なり（$x\approx-0.6$ の●と $x\approx0.8$ の○）があるため、大きすぎるとかえって損失 $L$ が増えます（確信を持って間違える点の損失が大きい）。
    - $x$ を動かすと、「オッズ $=e^{z}$」の行が、$z$ が 1 増えるごとに $e\approx2.72$ 倍になるのが分かります。

---

## オッズ・ロジット・ロジスティック関数 {#odds}

| 名称 | 式 | 値域 | 意味 |
|:--|:--|:--|:--|
| **オッズ** | $\dfrac{p}{1-p}$ | $(0,\infty)$ | 起こる確率と起こらない確率の比 |
| **ロジット** | $\mathrm{logit}(p)=\log\dfrac{p}{1-p}$ | $(-\infty,\infty)$ | オッズの対数 |
| **ロジスティック関数** | $\sigma(z)=\dfrac{1}{1+e^{-z}}$ | $(0,1)$ | ロジットの**逆関数**（シグモイド関数と同じもの） |
| **オッズ比** | $\dfrac{\text{条件 A のオッズ}}{\text{条件 B のオッズ}}$ | $(0,\infty)$ | 特徴量が 1 増えたときのオッズの倍率 |

$$
\begin{aligned}
&\text{モデル} && \log\frac{p}{1-p}=\mathbf{w}^{\top}\mathbf{x}+b \quad\Longleftrightarrow\quad p=\sigma(\mathbf{w}^{\top}\mathbf{x}+b) \\[2mm]
&\text{オッズ} && \frac{p}{1-p}=\frac{\sigma(z)}{1-\sigma(z)}=\frac{(1+e^{-z})^{-1}}{e^{-z}(1+e^{-z})^{-1}}=e^{z}=e^{b}\prod_{j}e^{w_jx_j} \\[2mm]
&\text{オッズ比} && x_j\to x_j+1\ \Rightarrow\ \text{オッズが } e^{w_j}\text{ 倍}
\end{aligned}
$$

- 確率 $p$ は 0〜1 に縛られて線形式で表せないが、**ロジットに変換すると全実数**になり、線形回帰と同じ形で書ける（一般化線形モデルで、リンク関数が logit、分布がベルヌーイ）。
- 係数 $w_j$ の解釈：他の変数を固定して $x_j$ を 1 増やすと、**オッズが $e^{w_j}$ 倍**（確率が何倍になるかではない）。$w_j>0$ ならオッズ比は 1 より大きい。
- 2×2 の分割表（曝露あり：$a$ 人が発症・$b$ 人が非発症、曝露なし：$c$ 人が発症・$d$ 人が非発症）では、オッズ比 $=\dfrac{a/b}{c/d}=\dfrac{ad}{bc}$。
- $p=0.5$ でオッズ 1、ロジット 0。$p=0.8$ でオッズ 4。

---

## ソフトマックス関数（多クラス） {#softmax}

$K$ クラスでは、クラスごとに重み $\mathbf{w}_k$ とバイアス $b_k$ を持ち、ロジット $z_k$ を**ソフトマックス関数**で確率にする。ラベルは one-hot ベクトル $\mathbf{t}$。

$$
\begin{aligned}
&\text{ロジット} && z_k=\mathbf{w}_k^{\top}\mathbf{x}+b_k\quad\Bigl(\mathbf{Z}=\mathbf{X}\mathbf{W}+\mathbf{1}\mathbf{b}^{\top}\Bigr) \\[2mm]
&\text{ソフトマックス} && y_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}} \\[2mm]
&\text{損失（交差エントロピー）} && L=-\sum_{n=1}^{N}\sum_{k=1}^{K}t_{nk}\log y_{nk} \\[2mm]
&\text{勾配} && \frac{\partial L}{\partial z_{nk}}=y_{nk}-t_{nk},\qquad \frac{\partial L}{\partial \mathbf{W}}=\mathbf{X}^{\top}(\mathbf{Y}-\mathbf{T})
\end{aligned}
$$

| 記号 | 形 |
|:--|:--|
| $\mathbf{X}$ | $N\times D$ |
| $\mathbf{W}$ | $D\times K$ |
| $\mathbf{Y},\ \mathbf{T}$ | $N\times K$ |
| $\mathbf{X}^{\top}(\mathbf{Y}-\mathbf{T})$ | $(D\times N)(N\times K)=D\times K$ |

- $y_k\in(0,1)$、$\sum_ky_k=1$。**定数を引いても不変**：$y_k(\mathbf{z})=y_k(\mathbf{z}-c\mathbf{1})$。実装では $c=\max_jz_j$ を引いて、$e^{z}$ のオーバーフローを防ぐ。
- ヤコビ行列：$\dfrac{\partial y_i}{\partial z_j}=y_i(\delta_{ij}-y_j)$。交差エントロピーと合わせると、勾配が $y_k-t_k$ に単純化する（[誤差逆伝播法](../dl/backprop.md)の出力層のデルタ）。
- $K=2$ のとき、$z=z_1-z_0$ とおくと $y_1=\sigma(z)$。ソフトマックスはシグモイドの多クラス版。
- ソフトマックスの温度 $e^{z_k/T}$ は [活性化関数](../dl/activation.md) を参照。深層学習では、出力層をソフトマックス＋交差エントロピーにするのが標準（[多層パーセプトロン](../dl/mlp-loss.md)）。
- 多クラス分類を二値分類器の組み合わせで行う方法：**1対他（OvR）**（$K$ 個）、**1対1（OvO）**（$K(K-1)/2$ 個）。ソフトマックス回帰は全クラスを同時に学習する。

---

## 試験の着眼点

- 出力は**確率**。損失は**交差エントロピー**（ベルヌーイ分布の負の対数尤度）。勾配は「**予測 − 正解**」を入力に掛けて足し合わせたもの。
- ロジット $=\log\dfrac{p}{1-p}$、ロジスティック関数はその逆関数。**オッズ $=e^{z}$、オッズ比 $=e^{w_j}$**。
- 判別境界は線形。線形分離可能なデータでは重みが発散するので正則化が要る。
- ソフトマックス：和が 1、差だけで決まる（定数を引いても同じ）。$K=2$ でシグモイドに一致。
- 「回帰」と名がつくが分類手法。線形回帰のような解析解はない。
- SVM のヒンジ損失との違いは [SVM](svm.md)、損失関数全体は [損失関数](../math/loss.md)。

---

## 参考

- [scikit-learn User Guide: Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [The Elements of Statistical Learning（Hastie, Tibshirani, Friedman）](https://hastie.su.domains/ElemStatLearn/)：第4章 Linear Methods for Classification
