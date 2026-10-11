# パラメータ推定（MLE・MAP・ベイズ推定）

<p class="kw">キーワード：ベイズ則、ナイーブベイズ、平均二乗誤差、対数尤度、ダイバージェンス、最尤推定（MLE）、MAP推定、ベイズ推定、共役事前分布、識別モデル・生成モデル</p>

!!! abstract "要点"
    - **最尤推定**：データの尤度 $p(\mathcal{D}\mid\boldsymbol{\theta})$ を最大にする $\boldsymbol{\theta}$ を選ぶ。**負の対数尤度を最小化**する形で学習に使われ、MSE・交差エントロピーはその特別な場合。
    - **MAP推定**：事前分布 $p(\boldsymbol{\theta})$ を加えて、事後確率を最大にする点を選ぶ。**事前分布 ＝ 正則化**（ガウスなら L2、ラプラスなら L1）。
    - **ベイズ推定**：点ではなく、事後分布 $p(\boldsymbol{\theta}\mid\mathcal{D})$ **全体**を求め、不確かさごと予測に使う。

---

## 尤度と対数尤度 {#likelihood}

データ $\mathcal{D}=\{x_1,\dots,x_N\}$ が独立に同じ分布 $f(x\mid\boldsymbol{\theta})$ から得られたとする（i.i.d.）。

$$
p(\mathcal{D}\mid\boldsymbol{\theta})=\prod_{i=1}^{N}f(x_i\mid\boldsymbol{\theta}),\qquad
\ell(\boldsymbol{\theta})=\log p(\mathcal{D}\mid\boldsymbol{\theta})=\sum_{i=1}^{N}\log f(x_i\mid\boldsymbol{\theta})
$$

- **尤度**は、データを固定して $\boldsymbol{\theta}$ の関数と見たもの。$\boldsymbol{\theta}$ について足しても（積分しても）1 にならないので、確率ではない。
- 対数をとると、積が和になる。確率の積が 0 に潰れる（アンダーフロー）のを避け、微分もしやすい。対数は単調増加なので、最大になる $\boldsymbol{\theta}$ は変わらない。
- 損失として使うときは符号を反転した**負の対数尤度**（NLL）$L(\boldsymbol{\theta})=-\ell(\boldsymbol{\theta})$ を最小化する（最適化アルゴリズムは最小化を前提にしているため）。

## 最尤推定（MLE） {#mle}

$$
\hat{\boldsymbol{\theta}}_{\mathrm{MLE}}
=\mathop{\arg\max}_{\boldsymbol{\theta}}\prod_{i=1}^{N}f(x_i\mid\boldsymbol{\theta})
=\mathop{\arg\max}_{\boldsymbol{\theta}}\sum_{i=1}^{N}\log f(x_i\mid\boldsymbol{\theta})
=\mathop{\arg\min}_{\boldsymbol{\theta}}\left\{-\sum_{i=1}^{N}\log f(x_i\mid\boldsymbol{\theta})\right\}
$$

- 解き方：$\partial\ell/\partial\boldsymbol{\theta}=\mathbf{0}$ が解けるなら閉じた形で、解けないなら勾配法で数値的に求める。

主な分布の対数尤度と最尤推定量。

| 分布 | 対数尤度 $\ell$ | 最尤推定量 |
|:--|:--|:--|
| ベルヌーイ | $\sum_i\bigl[x_i\log p+(1-x_i)\log(1-p)\bigr]$ | $\hat{p}=\dfrac1N\sum_ix_i$ |
| カテゴリカル | $\sum_i\sum_kx_{ik}\log p_k$（$\sum_kp_k=1$） | $\hat{p}_k=\dfrac{1}{N}\sum_ix_{ik}=\dfrac{N_k}{N}$ |
| 正規 | $-\dfrac N2\log(2\pi\sigma^2)-\dfrac{1}{2\sigma^2}\sum_i(x_i-\mu)^2$ | $\hat\mu=\bar{x},\ \ \hat\sigma^2=\dfrac1N\sum_i(x_i-\hat\mu)^2$ |
| ポアソン | $\sum_i\bigl(x_i\log\lambda-\lambda-\log x_i!\bigr)$ | $\hat\lambda=\bar{x}$ |
| 指数 | $N\log\lambda-\lambda\sum_ix_i$ | $\hat\lambda=\dfrac{N}{\sum_ix_i}=\dfrac1{\bar{x}}$ |
| 混合ガウス | $\sum_i\log\sum_k\pi_k\mathcal{N}(x_i\mid\mu_k,\sigma_k^2)$ | 閉じた形がない（EM アルゴリズム。[クラスタリング](../ml/clustering.md)） |

- ベルヌーイ：$\dfrac{\partial\ell}{\partial p}=\dfrac{\sum_ix_i}{p}-\dfrac{N-\sum_ix_i}{1-p}=0$ を解くと、表の割合 $\hat{p}$ になる。カテゴリカルは制約 $\sum_kp_k=1$ をラグランジュの未定乗数法で扱うと、各クラスの**出現割合** $N_k/N$ になる。
- 正規：$\hat\mu$ は標本平均。$\hat\sigma^2$ は $\frac1N$ で割るので**不偏でない**（$\mathbb{E}[\hat\sigma^2]=\frac{N-1}{N}\sigma^2$）。分母を $N-1$ にしたものが不偏分散。
- 混合ガウスは「和の対数」になるため、微分しても解が閉じない。

### MLE ＝ KL ダイバージェンスの最小化

真の分布 $p_{\mathrm{data}}$ をモデル $p_{\boldsymbol{\theta}}$ で近似することを考える。

$$
D_{\mathrm{KL}}\bigl(p_{\mathrm{data}}\,\Vert\,p_{\boldsymbol{\theta}}\bigr)
=\underbrace{\mathbb{E}_{p_{\mathrm{data}}}[\log p_{\mathrm{data}}(x)]}_{\boldsymbol{\theta}\text{ によらない}}
-\mathbb{E}_{p_{\mathrm{data}}}[\log p_{\boldsymbol{\theta}}(x)]
$$

- 第2項の期待値を、データの平均 $\frac1N\sum_i\log p_{\boldsymbol{\theta}}(x_i)$ で近似すると、対数尤度そのもの。よって「KL ダイバージェンスの最小化 ＝ **交差エントロピーの最小化** ＝ 最尤推定」（[情報理論](information-theory.md)）。

---

## 最小二乗法は正規分布の MLE {#least-squares}

回帰モデルを $y=f(\mathbf{x};\boldsymbol{\theta})+\varepsilon$、ノイズを $\varepsilon\sim\mathcal{N}(0,\sigma^2)$ とすると、$y\mid\mathbf{x}\sim\mathcal{N}\bigl(f(\mathbf{x};\boldsymbol{\theta}),\sigma^2\bigr)$。

$$
-\log p(\mathcal{D}\mid\boldsymbol{\theta})
=\frac{1}{2\sigma^2}\sum_{i=1}^{N}\bigl(y_i-f(\mathbf{x}_i;\boldsymbol{\theta})\bigr)^2+N\log\sqrt{2\pi\sigma^2}
$$

- $\boldsymbol{\theta}$ に関して第2項は定数、第1項の $\frac{1}{2\sigma^2}$ は正の定数倍。したがって最小化するのは $\frac12\sum_i\bigl(y_i-f(\mathbf{x}_i;\boldsymbol{\theta})\bigr)^2$、つまり**二乗誤差（平均二乗誤差 MSE）**。
- $\sigma^2$ の最尤推定量は残差の二乗平均 $\hat\sigma^2=\frac1N\sum_i\bigl(y_i-f(\mathbf{x}_i;\hat{\boldsymbol{\theta}})\bigr)^2$（MSE そのもの）。
- 線形モデル $f=\boldsymbol{\theta}^{\top}\mathbf{x}$ では、$\mathbf{X}$ を $N\times D$ の行列として $\partial L/\partial\boldsymbol{\theta}=\mathbf{X}^{\top}(\mathbf{X}\boldsymbol{\theta}-\mathbf{y})=\mathbf{0}$ から**正規方程式** $\hat{\boldsymbol{\theta}}=(\mathbf{X}^{\top}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{y}$。

どんな分布を仮定するかで、損失が決まる。

| モデル | 仮定する分布 | 損失（負の対数尤度） | 勾配 |
|:--|:--|:--|:--|
| 線形回帰 | $y\sim\mathcal{N}(\boldsymbol{\theta}^{\top}\mathbf{x},\sigma^2)$ | $\frac12\lVert\mathbf{y}-\mathbf{X}\boldsymbol{\theta}\rVert^2$ | $\mathbf{X}^{\top}(\mathbf{X}\boldsymbol{\theta}-\mathbf{y})$ |
| ニューラルネット（回帰） | $y\sim\mathcal{N}(f_{\boldsymbol{\theta}}(\mathbf{x}),\sigma^2)$ | $\frac12\lVert\mathbf{y}-\hat{\mathbf{y}}\rVert^2$ | 出力の勾配 $\hat{\mathbf{y}}-\mathbf{y}$ |
| ロジスティック回帰 | $y\sim\mathrm{Ber}\bigl(\sigma(\boldsymbol{\theta}^{\top}\mathbf{x})\bigr)$ | $-\sum_i\bigl[y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)\bigr]$ | $\mathbf{X}^{\top}(\hat{\mathbf{p}}-\mathbf{y})$ |
| ニューラルネット（多クラス分類） | $y\sim\mathrm{Cat}\bigl(\mathrm{softmax}(\mathbf{u})\bigr)$ | $-\log\hat{y}_{c}$（正解クラス $c$、交差エントロピー） | $\dfrac{\partial L}{\partial\mathbf{u}}=\hat{\mathbf{y}}-\mathbf{t}$ |

- 更新式はどれも $\boldsymbol{\theta}\leftarrow\boldsymbol{\theta}-\eta\,\partial L/\partial\boldsymbol{\theta}$。
- 詳しくは [線形回帰](../ml/linear-regression.md)、[ロジスティック回帰](../ml/logistic.md)、[出力層と損失関数](../dl/mlp-loss.md)、[損失関数まとめ](loss.md)。

---

## MAP推定とベイズ推定 {#map-bayes}

| 手法 | 求めるもの | 事前分布 |
|:--|:--|:--|
| 最尤推定（MLE） | 尤度 $p(\mathcal{D}\mid\boldsymbol{\theta})$ を最大にする**点** | 使わない |
| MAP推定 | 事後確率 $p(\boldsymbol{\theta}\mid\mathcal{D})$ を最大にする**点** | 使う |
| ベイズ推定 | 事後分布 $p(\boldsymbol{\theta}\mid\mathcal{D})$ **全体** | 使う |

$$
p(\boldsymbol{\theta}\mid\mathcal{D})=\frac{p(\mathcal{D}\mid\boldsymbol{\theta})\,p(\boldsymbol{\theta})}{p(\mathcal{D})}\ \propto\ p(\mathcal{D}\mid\boldsymbol{\theta})\,p(\boldsymbol{\theta}),\qquad
\hat{\boldsymbol{\theta}}_{\mathrm{MAP}}
=\mathop{\arg\min}_{\boldsymbol{\theta}}\Bigl\{-\log p(\mathcal{D}\mid\boldsymbol{\theta})-\log p(\boldsymbol{\theta})\Bigr\}
$$

- 分母 $p(\mathcal{D})$ は $\boldsymbol{\theta}$ によらないので、MAP では無視できる。
- 「**負の対数尤度 ＋ 負の対数事前分布**」なので、MLE の損失に**正則化項**を足した形になる。事前分布が一様なら MAP ＝ MLE。

| 事前分布 | $-\log p(\boldsymbol{\theta})$（定数を除く） | 対応する正則化 |
|:--|:--|:--|
| ガウス $\mathcal{N}(\mathbf{0},\tau^2I)$ | $\dfrac{1}{2\tau^2}\lVert\boldsymbol{\theta}\rVert_2^2$ | L2（重み減衰） |
| ラプラス $\propto\exp(-\lVert\boldsymbol{\theta}\rVert_1/b)$ | $\dfrac1b\lVert\boldsymbol{\theta}\rVert_1$ | L1 |

- 回帰（ノイズ $\sigma^2$）に当てはめると、$\frac{1}{2\sigma^2}\lVert\mathbf{y}-\mathbf{X}\boldsymbol{\theta}\rVert^2+\frac{1}{2\tau^2}\lVert\boldsymbol{\theta}\rVert^2$。$\sigma^2$ 倍すれば $\frac12\lVert\mathbf{y}-\mathbf{X}\boldsymbol{\theta}\rVert^2+\frac\lambda2\lVert\boldsymbol{\theta}\rVert^2$、$\lambda=\sigma^2/\tau^2$（リッジ回帰）。**事前分布が狭い（$\tau$ が小さい）ほど正則化が強い**。ラプラス事前分布では、L1 の係数が $\sigma^2/b$ になる（[正則化](../dl/regularization.md)）。

### ベイズ推定と共役事前分布

- ベイズ推定は事後分布そのものを使う。事後平均 $\mathbb{E}[\boldsymbol{\theta}\mid\mathcal{D}]=\int\boldsymbol{\theta}\,p(\boldsymbol{\theta}\mid\mathcal{D})\,d\boldsymbol{\theta}$ や、新しい入力 $x^{*}$ に対する**予測分布**で、パラメータの不確かさを平均して予測する。

$$
p(x^{*}\mid\mathcal{D})=\int p(x^{*}\mid\boldsymbol{\theta})\,p(\boldsymbol{\theta}\mid\mathcal{D})\,d\boldsymbol{\theta}
$$

- 積分が解析的に計算できないときは、**MCMC**（[確率過程](stochastic-process.md#mcmc)）や**変分推論**で近似する。
- **共役事前分布**：事前分布と事後分布が同じ分布族になる組み合わせ。事後分布が解析的に求まる。

| 尤度 | 共役事前分布 | 事後分布 |
|:--|:--|:--|
| ベルヌーイ・二項（$N$ 回中 $k$ 回成功） | ベータ $\mathrm{Beta}(\alpha,\beta)$ | $\mathrm{Beta}(\alpha+k,\ \beta+N-k)$ |
| カテゴリカル・多項 | ディリクレ $\mathrm{Dir}(\alpha_1,\dots,\alpha_K)$ | $\mathrm{Dir}(\alpha_1+N_1,\dots,\alpha_K+N_K)$ |
| ポアソン | ガンマ $\mathrm{Gamma}(a,b)$（$b$ は率） | $\mathrm{Gamma}(a+\sum_ix_i,\ b+N)$ |
| 正規（分散既知、平均が未知） | 正規 | 正規 |

- ベータ分布の例：事前分布は $\theta^{\alpha-1}(1-\theta)^{\beta-1}$ の形で、「表が $\alpha-1$ 回、裏が $\beta-1$ 回出た」という**仮想のデータ**と見なせる。データが来るたびに指数に回数を足していけばよい（ベイズ更新）。

$$
\hat\theta_{\mathrm{MLE}}=\frac{k}{N},\qquad
\hat\theta_{\mathrm{MAP}}=\frac{k+\alpha-1}{N+\alpha+\beta-2},\qquad
\mathbb{E}[\theta\mid\mathcal{D}]=\frac{k+\alpha}{N+\alpha+\beta}
$$

- 事後平均は、事前分布の平均 $\frac{\alpha}{\alpha+\beta}$ と MLE $\frac kN$ の**加重平均**。データ数 $N$ が増えるほど MLE に寄る。

<div class="ew" data-widget="est-beta"></div>

!!! tip "動かしてみる"
    - $N=0$ にすると、事後分布が事前分布と重なります（データがなければ事前の信念のまま）。
    - $N$ と $k$ を $N=50,\ k=35$ のように増やすと、事後分布の山が尖って尤度に近づき、MAP推定と事後平均が最尤推定の 0.7 に寄ります。**データが多いと、事前分布の影響は薄れます**。
    - $\alpha=\beta=1$（一様な事前分布）にすると、MAP推定が最尤推定に一致します。$\alpha=\beta=10$ のように事前分布を強くすると、推定値が 0.5 に引っ張られます。

---

## 識別モデルと生成モデル {#generative}

クラス $\mathcal{C}$ の予測は、事後確率を最大にするクラスを選ぶ問題 $\hat{\mathcal{C}}=\mathop{\arg\max}_{\mathcal{C}}\,p(\mathcal{C}\mid\mathbf{x})$。

| | 識別モデル | 生成モデル |
|:--|:--|:--|
| モデル化するもの | $p(\mathcal{C}\mid\mathbf{x})$ を直接 | $p(\mathbf{x}\mid\mathcal{C})$ と $p(\mathcal{C})$（または同時確率 $p(\mathbf{x},\mathcal{C})$） |
| 予測 | そのまま $\arg\max$ | ベイズの定理で $p(\mathcal{C}\mid\mathbf{x})=\dfrac{p(\mathbf{x}\mid\mathcal{C})p(\mathcal{C})}{p(\mathbf{x})}$ |
| 例 | ロジスティック回帰、ニューラルネットワーク、SVM | ナイーブベイズ、混合ガウスモデル、VAE、GAN |

- 生成モデルは $p(\mathbf{x})=\sum_{\mathcal{C}}p(\mathbf{x}\mid\mathcal{C})p(\mathcal{C})$ と**周辺化**して、入力そのものの分布 $p(\mathbf{x})$ も得られる。そのため、予測だけでなく欠損値の補完・データ拡張・異常検知にも使える。
- 連続の場合の $p(\mathbf{x})=\int p(\mathbf{x}\mid\mathcal{C})p(\mathcal{C})\,d\mathcal{C}$ は一般に解析的に計算できず、MCMC や変分推論で近似する（[自己教師あり学習・生成モデル](../ml/ssl-generative.md)）。

## ナイーブベイズ {#naive-bayes}

特徴量 $x_1,\dots,x_D$ が、**クラスを与えたもとで互いに条件付き独立**と仮定する（これが「ナイーブ」）。

$$
P(C_k\mid x_1,\dots,x_D)=\frac{P(C_k)\,P(x_1,\dots,x_D\mid C_k)}{P(x_1,\dots,x_D)}\ \propto\ P(C_k)\prod_{d=1}^{D}P(x_d\mid C_k)
$$

$$
\hat{C}=\mathop{\arg\max}_{k\in\{1,\dots,K\}}\left\{\log P(C_k)+\sum_{d=1}^{D}\log P(x_d\mid C_k)\right\}
$$

- 分母はクラスによらないので、最大化では無視できる。確率の積は小さくなりすぎるので、対数をとって和にする。
- クラスの事後確率を最大にする **MAP推定**の一種。$P(C_k)$ はクラスの頻度、$P(x_d\mid C_k)$ は各特徴量の分布を MLE で求める。
- 文書分類：$P(\text{カテゴリ}\mid\text{文書})\propto P(\text{カテゴリ})\prod_{w}P(w\mid\text{カテゴリ})$（語順を無視した、単語の出現だけのモデル）。
- **ゼロ頻度問題**：学習データに一度も出なかった単語は $P(w\mid C)=0$ となり、積全体が 0 になる。**ラプラス平滑化**（加算平滑化）で $P(w\mid C)=\dfrac{n_{Cw}+\alpha}{n_C+\alpha V}$（$n_{Cw}$：クラス $C$ での単語 $w$ の出現回数、$n_C$：クラス $C$ の総単語数、$V$：語彙数、$\alpha=1$）とする。ディリクレ事前分布を置いたベイズ推定（事後平均）に一致する。
- 特徴量の種類に応じた変種：**ガウス**（連続値を正規分布で）、**多項**（出現回数）、**ベルヌーイ**（出現の有無）。scikit-learn の `GaussianNB`・`MultinomialNB`・`BernoulliNB` に対応し、後の2つに平滑化の強さ `alpha` がある（[ナイーブベイズ](../ml/bayes-gp-crf.md)）。

---

## 試験の着眼点

- **MLE**：尤度最大。**MAP**：尤度×事前の最大（事後確率最大）。**ベイズ推定**：事後分布全体。事前分布が一様なら MAP ＝ MLE。
- 対数尤度で積が和になる。損失は**負の対数尤度**。正規分布ノイズなら二乗誤差、ベルヌーイなら二値交差エントロピー、カテゴリカルなら交差エントロピー。
- ガウス事前分布 ＝ L2 正則化、ラプラス事前分布 ＝ L1 正則化。
- MLE ＝ KL ダイバージェンスの最小化 ＝ 交差エントロピーの最小化。
- 正規分布の分散の MLE は $\frac1N$ で割る（不偏でない）。
- ナイーブベイズは**条件付き独立**を仮定し、分母を省略し、対数の和で計算する。ゼロ頻度問題にはラプラス平滑化。
- 識別モデルは $p(\mathcal{C}\mid\mathbf{x})$、生成モデルは $p(\mathbf{x}\mid\mathcal{C})p(\mathcal{C})$ をモデル化する。

## 参考

- [Pattern Recognition and Machine Learning, Chapter 1–2（Bishop, 2006）](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
- [Deep Learning, Chapter 5 Machine Learning Basics（Goodfellow, Bengio, Courville, 2016）](https://www.deeplearningbook.org/contents/ml.html)
- [Probabilistic Machine Learning: An Introduction（Murphy, 2022）](https://probml.github.io/pml-book/book1.html)
