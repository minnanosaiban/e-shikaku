# ナイーブベイズ・ガウス過程回帰・条件付き確率場

<p class="kw">キーワード：ナイーブベイズ、条件付き独立、MAP推定、ガウス過程回帰、カーネル、周辺尤度、条件付き確率場（CRF）、ビタビアルゴリズム</p>

!!! abstract "要点"
    - **ナイーブベイズ**は「特徴どうしがクラスを条件に独立」と仮定し、事後確率を $P(C_k)\prod_n P(x_n\mid C_k)$ に分解する**生成モデル**。
    - **ガウス過程回帰**は関数そのものに事前分布（ガウス過程）を置く。予測は**平均と分散**の閉じた式で出て、データから遠いほど不確かさが増える。
    - **条件付き確率場（CRF）**は系列ラベルを $P(\mathbf{y}\mid\mathbf{x})$ で直接モデル化する**識別モデル**。全系列で正規化し、デコードはビタビ法。

---

## ナイーブベイズ {#naive-bayes}

クラスの事後確率が最大のクラスを選ぶ（**MAP 推定**）。$\mathbf{x}=(x_1,\dots,x_D)$、クラスを $C_k\ (k=1,\dots,K)$ とする。

$$
\begin{aligned}
&\text{ベイズの定理} && P(C_k\mid\mathbf{x})=\frac{P(C_k)\,P(\mathbf{x}\mid C_k)}{P(\mathbf{x})} \\[2mm]
&\text{条件付き独立の仮定} && P(\mathbf{x}\mid C_k)=\prod_{n=1}^{D}P(x_n\mid C_k) \\[2mm]
&\text{対数化} && \log P(C_k\mid\mathbf{x})=\log P(C_k)+\sum_{n=1}^{D}\log P(x_n\mid C_k)-\log P(\mathbf{x}) \\[2mm]
&\text{予測} && \hat{C}=\operatorname*{arg\,max}_{C_k}\Bigl(\log P(C_k)+\sum_{n=1}^{D}\log P(x_n\mid C_k)\Bigr)
\end{aligned}
$$

- $P(\mathbf{x})$ はクラスによらないので、$\arg\max$ では無視できる（確率そのものが欲しいときは全クラスの和で正規化する）。
- 対数をとる理由は、小さい確率の積による**アンダーフロー**を避けるため。積が和になり、後の説明（線形判別）にも都合がよい。
- 学習は**数え上げ**だけ。$P(C_k)$ はクラスの頻度、$P(x_n\mid C_k)$ は各クラス内での特徴の分布を最尤推定する。反復計算は要らない。

| 種類 | 特徴 $x_n$ | $P(x_n\mid C_k)$ | 用途 |
|:--|:--|:--|:--|
| ガウスNB | 連続値 | $\mathcal{N}(\mu_{kn},\sigma_{kn}^2)$（クラス・特徴ごとに平均と分散） | 数値特徴 |
| 多項NB | 出現回数 | 語 $w$ の出現確率 $\theta_{kw}$ の積（$\prod_w\theta_{kw}^{x_w}$） | 文書分類 |
| ベルヌーイNB | 0/1（有無） | 出現する確率 $\theta_{kn}$、しない確率 $1-\theta_{kn}$ | 短い文書 |

### 小さな計算例

迷惑メール判定。$P(\text{迷惑})=0.4$、語「無料」「会議」について $P(\text{無料}\mid\text{迷惑})=0.5,\ P(\text{無料}\mid\text{普通})=0.05,\ P(\text{会議}\mid\text{迷惑})=0.1,\ P(\text{会議}\mid\text{普通})=0.4$。両方を含むメールは、出現した語だけで比べる簡略化をすると次のようになる。

$$
\begin{aligned}
&\text{迷惑：}\ 0.4\times0.5\times0.1=0.020,\qquad \text{普通：}\ 0.6\times0.05\times0.4=0.012 \\[1mm]
&P(\text{迷惑}\mid\mathbf{x})=\frac{0.020}{0.020+0.012}=0.625
\end{aligned}
$$

### ゼロ頻度問題とスムージング

- 訓練データに一度も現れなかった語があると $P(x_n\mid C_k)=0$ になり、積全体が 0 になる。
- **ラプラス平滑化**（加算スムージング）：語彙数 $V$、クラス $k$ 内の語 $w$ の出現数 $n_{kw}$、総語数 $n_k$ として

$$
\hat\theta_{kw}=\frac{n_{kw}+\alpha}{n_k+\alpha V}\qquad(\alpha=1\text{ がラプラス平滑化})
$$

- 分子に $\alpha$、分母に $\alpha V$ を足すのは、全語の確率の和を 1 に保つため。

### 性質

- 独立の仮定は現実には成り立たないことが多いが、**分類（$\arg\max$）は意外に当たる**。ただし出力の確率値は 0 か 1 に寄りやすく、信頼できない（較正が必要）。
- 多項NB・ベルヌーイNB・等分散のガウスNBでは、対数事後確率が特徴の**線形関数**になり、決定境界は直線（超平面）。
- ロジスティック回帰は同じ形の**識別モデル**。データが少ないときはNB、多いときはロジスティック回帰が有利と言われる（[ロジスティック回帰](logistic.md)）。
- ガウスNBは共分散を対角（特徴間で独立）に限った生成モデル。クラス間で共通の一般の共分散行列を許したものが LDA（[次元圧縮](dimension-reduction.md#lda)）。

---

## ガウス過程回帰 {#gpr}

関数 $f$ に**ガウス過程**を事前分布として置く。任意の入力の有限個の組 $f(x_1),\dots,f(x_N)$ が同時に多変量正規分布に従う、というのが定義である。

$$
f\sim\mathcal{GP}\bigl(m(x),\,k(x,x')\bigr),\qquad y=f(x)+\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,\sigma_n^2)
$$

- 平均関数 $m(x)$ は 0 とすることが多い（以下 $m=0$）。形を決めるのは**カーネル** $k(x,x')$：入力が近いほど出力も似る、という強さを表す。
- 代表的なカーネル：**RBF**（ガウス）$k(x,x')=\sigma_f^2\exp\!\bigl(-\|x-x'\|^2/2\ell^2\bigr)$（なめらかな関数）、線形、周期、マターン（RBFより粗い関数）。
- 訓練データ $(\mathbf{X},\mathbf{y})$（$\mathbf{y}\in\mathbb{R}^N$）から、**カーネル行列** $\mathbf{K}_{ij}=k(\mathbf{x}^{(i)},\mathbf{x}^{(j)})$ を作る。

$$
\begin{aligned}
&\text{結合分布} && \begin{pmatrix}\mathbf{y}\\ f_*\end{pmatrix}\sim\mathcal{N}\!\left(\mathbf{0},\ \begin{pmatrix}\mathbf{K}+\sigma_n^2\mathbf{I} & \mathbf{k}_*\\ \mathbf{k}_*^{\top} & k(x_*,x_*)\end{pmatrix}\right),\quad (\mathbf{k}_*)_i=k(\mathbf{x}^{(i)},x_*) \\[3mm]
&\text{予測平均} && \mu_*=\mathbf{k}_*^{\top}\bigl(\mathbf{K}+\sigma_n^2\mathbf{I}\bigr)^{-1}\mathbf{y} \\[3mm]
&\text{予測分散} && \sigma_*^2=k(x_*,x_*)-\mathbf{k}_*^{\top}\bigl(\mathbf{K}+\sigma_n^2\mathbf{I}\bigr)^{-1}\mathbf{k}_* \\[3mm]
&\text{周辺尤度} && \log p(\mathbf{y}\mid\mathbf{X})=\underbrace{-\tfrac12\mathbf{y}^{\top}\bigl(\mathbf{K}+\sigma_n^2\mathbf{I}\bigr)^{-1}\mathbf{y}}_{\text{データへの適合}}\ \underbrace{-\tfrac12\log\bigl|\mathbf{K}+\sigma_n^2\mathbf{I}\bigr|}_{\text{複雑さへの罰}}\ -\tfrac{N}{2}\log2\pi
\end{aligned}
$$

- 予測は結合正規分布の**条件付き分布**を求めただけ（正規分布の条件付けの公式）。事後分布も正規分布になるので、閉じた式で出る。
- $\sigma_*^2$ は潜在関数 $f_*$ の不確かさ。観測値 $y_*$ の予測分散にはノイズ $\sigma_n^2$ を足す。
- 平均が $m\neq0$ のときは $\mu_*=m(x_*)+\mathbf{k}_*^{\top}(\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}(\mathbf{y}-\mathbf{m})$。
- $\boldsymbol\alpha=(\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}\mathbf{y}$ とおくと $\mu_*=\sum_i\alpha_i\,k(\mathbf{x}^{(i)},x_*)$。予測平均は**カーネルの重ね合わせ**で、カーネルリッジ回帰（$\sigma_n^2$ が正則化係数）と一致する。

<div class="ew" data-widget="bayes-gp"></div>

!!! tip "動かしてみる"
    - $x_*$ を観測点のない右端（10 付近）へ動かすと、予測平均は 0 に戻り、$\sigma_*^2$ は $k(x_*,x_*)=1$（事前分布の分散）に近づきます（データから遠いほど何も分からない）。
    - $\ell$ を小さくすると曲線は観測点ごとに細かく振れ、大きくすると滑らかになります。周辺尤度の「データへの適合」と「複雑さ」の和が、どこで大きくなるかを探してみてください。
    - $\sigma_n$ を小さくすると曲線は観測点をそのまま通り（帯が観測点で細くなる）、大きくすると観測点を無視した緩い曲線になります。

### ハイパーパラメータの学習

- カーネルのパラメータ（$\ell,\sigma_f$）とノイズ $\sigma_n$ は、**周辺尤度の最大化**（第2種の最尤推定）で決める。勾配法で最適化する。
- 周辺尤度には「適合」と「複雑さ」が両方入るため、**検証データを使わなくても**過剰適合が抑えられる（自動的な Occam の剃刀）。
- 計算量は、コレスキー分解が $O(N^3)$、メモリが $O(N^2)$。大規模データには疑似入力点を使う近似などがある。

### 使いどころ

- 少ないデータで、**不確かさ付き**の回帰をしたいとき（実験計画、ベイズ最適化）。ベイズ最適化では予測平均と分散から獲得関数（EI、UCB など）を作り、次に試す点を選ぶ。
- ニューラルネットワークの幅を無限にした極限はガウス過程になる（関連研究）。

---

## 条件付き確率場（CRF） {#crf}

入力系列 $\mathbf{x}=(x_1,\dots,x_T)$ に対するラベル系列 $\mathbf{y}=(y_1,\dots,y_T)$ の条件付き確率を直接モデル化する（品詞タグ付け、固有表現抽出など）。ここでは隣り合うラベルだけが結びつく**線形連鎖 CRF** を扱う。

$$
\begin{aligned}
&\text{モデル} && P(\mathbf{y}\mid\mathbf{x})=\frac{1}{Z(\mathbf{x})}\exp\!\left(\sum_{t=1}^{T}\sum_{k}w_k\,f_k(y_{t-1},y_t,\mathbf{x},t)\right) \\[3mm]
&\text{分配関数} && Z(\mathbf{x})=\sum_{\mathbf{y}'}\exp\!\left(\sum_{t=1}^{T}\sum_{k}w_k\,f_k(y'_{t-1},y'_t,\mathbf{x},t)\right) \\[3mm]
&\text{対数尤度} && \log P(\mathbf{y}\mid\mathbf{x})=\sum_{t,k}w_k f_k-\log Z(\mathbf{x}) \\[3mm]
&\text{勾配} && \frac{\partial \log P(\mathbf{y}\mid\mathbf{x})}{\partial w_k}=\sum_{t}f_k(y_{t-1},y_t,\mathbf{x},t)-\mathbb{E}_{P(\mathbf{y}'\mid\mathbf{x})}\!\Bigl[\sum_{t}f_k(y'_{t-1},y'_t,\mathbf{x},t)\Bigr] \\[3mm]
&\text{デコード} && \mathbf{y}^*=\operatorname*{arg\,max}_{\mathbf{y}}P(\mathbf{y}\mid\mathbf{x})=\operatorname*{arg\,max}_{\mathbf{y}}\sum_{t,k}w_k f_k
\end{aligned}
$$

- **特徴量関数** $f_k$ は人が設計する（遷移特徴：ラベルの並び $y_{t-1}\to y_t$、出力特徴：語と $y_t$ の組、など）。重み $w_k$ が学習の対象。
- $Z(\mathbf{x})$ は**全ラベル系列**（$M^T$ 通り、$M$ はラベル数）にわたる和。ただし系列の構造を使った**前向きアルゴリズム**で $O(TM^2)$ で計算できる。
- 勾配は「観測された特徴の合計」と「モデルの下での特徴の期待値」の**差**。期待値は前向き・後ろ向きアルゴリズムで得る周辺確率から計算する。差が 0 になる点が最尤解（指数型分布族の性質）。
- 実際には過剰適合を防ぐため、L2 正則化を加えて勾配法（L-BFGS や SGD）で最適化する。
- **デコード**では $Z(\mathbf{x})$ が系列によらないので不要。**ビタビアルゴリズム**で動的計画法により $O(TM^2)$ で $\arg\max$ を求める。

| アルゴリズム | 漸化式 | 求めるもの |
|:--|:--|:--|
| 前向き | $\alpha_t(j)=\sum_i\alpha_{t-1}(i)\,\exp\psi_t(i,j)$ | $Z(\mathbf{x})=\sum_j\alpha_T(j)$ |
| ビタビ | $\delta_t(j)=\max_i\bigl[\delta_{t-1}(i)+\psi_t(i,j)\bigr]$ | 最尤系列（argmax を覚えて逆にたどる） |

ここで $\psi_t(i,j)=\sum_k w_k f_k(i,j,\mathbf{x},t)$ は位置 $t$ でラベルが $i\to j$ と遷移するときのスコアである。

### HMM・ソフトマックス回帰との関係

- ロジスティック回帰（ソフトマックス回帰）：$T=1$ の CRF にあたる。CRF は**系列全体を1つのソフトマックスで正規化**したもの。
- [隠れマルコフモデル](ssl-generative.md#hmm)（HMM）は同時確率 $P(\mathbf{x},\mathbf{y})$ を作る生成モデルで、観測の独立を仮定する。CRF は条件付き確率だけを扱うので、**入力に任意の特徴（前後の語、接頭辞など）を使える**。
- 各時刻で別々に正規化する MEMM は、遷移先が少ない状態に確率が偏る**ラベルバイアス**を起こす。CRF は全体で正規化するので起きない。
- 深層学習では、BiLSTM や Transformer の出力を $f_k$ の代わりにするニューラル CRF（BiLSTM-CRF）が使われる。

---

## 試験の着眼点

- ナイーブベイズの「ナイーブ」は**クラスを与えたときの特徴の条件付き独立**。特徴どうしが無相関という意味ではない。
- 分母 $P(\mathbf{x})$ は $\arg\max$ では無視できる。実装は対数の和で行う。
- ゼロ頻度にはラプラス平滑化（分子 $+\alpha$、分母 $+\alpha V$）。
- ガウス過程は**関数の分布**。予測が平均と分散で出る点、$O(N^3)$ の計算量、周辺尤度でハイパーパラメータを決める点を押さえる。
- 周辺尤度の第1項は適合、第2項（$\log$ 行列式）は複雑さへの罰。
- CRF は**識別モデル**（HMM は生成モデル）。学習は対数尤度の最大化、推論（デコード）はビタビ。分配関数 $Z$ は全系列の和で、前向きアルゴリズムで効率的に計算する。

---

## 参考

- [Naive Bayes（scikit-learn 公式ドキュメント）](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Gaussian Processes for Machine Learning（Rasmussen, Williams, 2006）](https://gaussianprocess.org/gpml/)
- [An Introduction to Conditional Random Fields（Sutton, McCallum, 2012）](https://arxiv.org/abs/1011.4088)
- Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data（Lafferty, McCallum, Pereira, 2001, ICML）
