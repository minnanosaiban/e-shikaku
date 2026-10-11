# 線形回帰・正則化

<p class="kw">キーワード：ノルム、過少適合、過剰適合、最小二乗法、相関係数、多重共線性、L1正則化、L2正則化</p>

!!! abstract "要点"
    - 線形回帰は、予測 $\hat{y}=\mathbf{w}^{\top}\mathbf{x}$ と正解の**二乗誤差の和を最小**にする。勾配を 0 とおいた**正規方程式**で解析的に解ける（誤差が正規分布なら最尤推定と同じ）。
    - モデルが複雑すぎると**過剰適合**、単純すぎると**過少適合**になる。説明変数どうしが強く相関すると $\mathbf{X}^{\top}\mathbf{X}$ が不安定になる（**多重共線性**）。
    - 損失に**ノルムの罰則**を足すのが正則化。**L2（Ridge）は係数を縮め**、**L1（Lasso）は一部の係数を 0 にする**。

---

## 最小二乗法 {#ols}

入力 $N$ 個を並べた計画行列 $\mathbf{X}\in\mathbb{R}^{N\times D}$（バイアスは、全サンプルで 1 の列を足して $\mathbf{w}$ に含める）、正解 $\mathbf{y}\in\mathbb{R}^{N}$ とする。

$$
\begin{aligned}
&\text{推論} && \hat{\mathbf{y}}=\mathbf{X}\mathbf{w} \\[2mm]
&\text{損失} && L(\mathbf{w})=\frac12\|\mathbf{y}-\mathbf{X}\mathbf{w}\|_2^2=\frac12\sum_{n=1}^{N}\bigl(y_n-\mathbf{w}^{\top}\mathbf{x}_n\bigr)^2 \\[2mm]
&\text{勾配} && \frac{\partial L}{\partial \mathbf{w}}=\mathbf{X}^{\top}\mathbf{X}\mathbf{w}-\mathbf{X}^{\top}\mathbf{y}=\mathbf{X}^{\top}(\mathbf{X}\mathbf{w}-\mathbf{y}) \\[2mm]
&\text{正規方程式} && \mathbf{X}^{\top}\mathbf{X}\,\mathbf{w}=\mathbf{X}^{\top}\mathbf{y}\ \ \Longrightarrow\ \ \hat{\mathbf{w}}=(\mathbf{X}^{\top}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{y} \\[2mm]
&\text{勾配降下法} && \mathbf{w}\leftarrow\mathbf{w}-\eta\,\mathbf{X}^{\top}(\mathbf{X}\mathbf{w}-\mathbf{y})
\end{aligned}
$$

| 記号 | 形 | 検算 |
|:--|:--|:--|
| $\mathbf{X}$ | $N\times D$ | — |
| $\mathbf{X}^{\top}(\mathbf{X}\mathbf{w}-\mathbf{y})$ | $(D\times N)(N\times 1)=D\times 1$ | 勾配は $\mathbf{w}$ と同じ形 |
| $\mathbf{X}^{\top}\mathbf{X}$ | $D\times D$ | 半正定値。逆行列が使えるのは $\mathrm{rank}\,\mathbf{X}=D$ のとき |

- 二乗誤差は $\mathbf{w}$ の凸な二次関数（ヘッセ行列 $\mathbf{X}^{\top}\mathbf{X}$ が半正定値）。勾配 0 の点が**大域的最小**。
- **幾何**：$\hat{\mathbf{y}}=\mathbf{X}(\mathbf{X}^{\top}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{y}$ は、$\mathbf{y}$ を $\mathbf{X}$ の列が張る空間へ**正射影**したもの。残差 $\mathbf{y}-\hat{\mathbf{y}}$ は各列と直交する（これが正規方程式）。
- **確率の見方**：$y=\mathbf{w}^{\top}\mathbf{x}+\varepsilon,\ \varepsilon\sim\mathcal{N}(0,\sigma^2)$ とすると、対数尤度は $-\frac{N}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\|\mathbf{y}-\mathbf{X}\mathbf{w}\|^2$。$\mathbf{w}$ について最大化すると二乗誤差の最小化と一致する（[パラメータ推定](../math/estimation.md)）。
- $D>N$ や列が従属のときは $(\mathbf{X}^{\top}\mathbf{X})^{-1}$ が存在しない。擬似逆行列 $\mathbf{X}^{+}\mathbf{y}$（最小ノルム解、[特異値分解](../math/eigen-svd.md)）か、後述の正則化を使う。実装では逆行列を直接作らず、QR 分解や SVD で解く。

---

## 相関係数と決定係数 {#correlation}

$$
r_{xy}=\frac{\sum_n(x_n-\bar{x})(y_n-\bar{y})}{\sqrt{\sum_n(x_n-\bar{x})^2}\sqrt{\sum_n(y_n-\bar{y})^2}}=\frac{\mathrm{Cov}(x,y)}{\sigma_x\sigma_y}\in[-1,\,1],\qquad
R^2=1-\frac{\sum_n(y_n-\hat{y}_n)^2}{\sum_n(y_n-\bar{y})^2}
$$

- $r=\pm1$ は完全な直線関係、$r=0$ は**線形な**関係がない（非線形な関係は見逃す）。**相関は因果ではない**。外れ値に弱い。
- 説明変数が 1 つの単回帰では、傾き $\hat{w}_1=r_{xy}\,\sigma_y/\sigma_x$、決定係数 $R^2=r_{xy}^2$。
- $R^2$ は「$y$ の分散のうちモデルが説明した割合」。説明変数を足すだけで訓練データでは必ず増えるので、モデル比較には検証データでの誤差を使う（[評価](evaluation.md)）。

---

## 過少適合・過剰適合 {#fitting}

線形回帰の「線形」は**パラメータ $\mathbf{w}$ について線形**という意味。$\phi_j(x)=x^j$ などの基底に置き換えれば、曲線も同じ式で扱える（多項式回帰）。

$$
\hat{y}=\sum_{j=0}^{d}w_j\,\phi_j(x),\qquad \phi_j(x)=x^{j}\ \text{（ここでは }(2x-1)^j\text{）}
$$

| 状態 | 次数・モデル | 訓練誤差 | 汎化誤差（テスト誤差） |
|:--|:--|:--|:--|
| **過少適合** | 単純すぎる（$d$ が小さい） | 大きい | 大きい |
| 適切 | ちょうどよい | 小さい | 小さい |
| **過剰適合** | 複雑すぎる（$d$ が大きい） | ほぼ 0 | **大きい**（ノイズまで覚える） |

- 対策：データを増やす、モデルを単純にする、**正則化**、早期終了、検証データでの選択（[評価](evaluation.md)）。

<div class="ew" data-widget="linreg-poly"></div>

!!! tip "動かしてみる"
    - 初期状態は $d=9$（データ 10 点に対して十分に複雑）で、訓練 RMSE はほぼ 0、テスト RMSE が大きい過剰適合です。曲線がデータの点を縫うように振れています。
    - $d=0$ や $1$ にすると、訓練もテストも誤差が大きいままです（過少適合）。$d=3\sim5$ あたりでテスト誤差が最小になります。
    - $d=9$ のまま $\log_{10}\lambda$ を $-6$ から大きくすると、曲線がなめらかになり、$\|\mathbf{w}\|^2$ が急に小さくなります（正則化の効果。大きくしすぎると過少適合）。

---

## 多重共線性 {#multicollinearity}

説明変数どうしがほぼ線形従属（強く相関）しているとき、$\mathbf{X}^{\top}\mathbf{X}$ は**ほぼ特異**になる。

- 係数の分散は $\mathrm{Var}(\hat{\mathbf{w}})=\sigma^2(\mathbf{X}^{\top}\mathbf{X})^{-1}$。固有値が 0 に近い方向があると、**係数が不安定**（データが少し変わるだけで符号まで変わる）になる。予測そのものは大きく崩れないこともある。
- 診断：説明変数どうしの相関係数、**分散拡大係数** $\mathrm{VIF}_j=\dfrac{1}{1-R_j^2}$（$R_j^2$：第 $j$ 変数を他の変数で回帰したときの決定係数。目安として 10 以上は要注意）、条件数。
- 対策：相関の強い変数を除く・まとめる、主成分分析で無相関にする（[次元圧縮](dimension-reduction.md)）、**Ridge 回帰**（$\mathbf{X}^{\top}\mathbf{X}+\lambda\mathbf{I}$ は常に正則）。

---

## ノルムと正則化 {#regularization}

### ノルム

| ノルム | 式 | 単位球（$\lVert\mathbf{w}\rVert=1$）の形（2次元） |
|:--|:--|:--|
| L1 | $\lVert\mathbf{w}\rVert_1=\sum_i\lvert w_i\rvert$ | 菱形（頂点が軸上） |
| L2 | $\lVert\mathbf{w}\rVert_2=\sqrt{\sum_i w_i^2}$ | 円 |
| Lp | $\lVert\mathbf{w}\rVert_p=\bigl(\sum_i\lvert w_i\rvert^{p}\bigr)^{1/p}$ | $p$ が大きいほど正方形に近づく |
| L∞ | $\lVert\mathbf{w}\rVert_\infty=\max_i\lvert w_i\rvert$ | 正方形 |
| L0（準ノルム） | 0 でない成分の個数 | 軸上の点（最適化が組合せ問題で難しい。L1 で代用する） |

### 罰則付き損失

$$
\begin{aligned}
&\text{一般形} && L_{\mathrm{reg}}(\mathbf{w})=L(\mathbf{w})+\lambda R(\mathbf{w}) \\[2mm]
&\text{Ridge（L2）} && R(\mathbf{w})=\frac12\|\mathbf{w}\|_2^2 \quad\Rightarrow\quad \hat{\mathbf{w}}=(\mathbf{X}^{\top}\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^{\top}\mathbf{y} \\[2mm]
&\text{Lasso（L1）} && R(\mathbf{w})=\|\mathbf{w}\|_1 \quad\Rightarrow\quad \text{閉じた形の解はなく、座標降下法・劣勾配法で解く} \\[2mm]
&\text{Elastic Net} && R(\mathbf{w})=\alpha\|\mathbf{w}\|_1+\frac{1-\alpha}{2}\|\mathbf{w}\|_2^2 \\[2mm]
&\text{勾配降下（L2）} && \mathbf{w}\leftarrow(1-\eta\lambda)\,\mathbf{w}-\eta\,\nabla L(\mathbf{w})\quad\text{（重み減衰 weight decay）}
\end{aligned}
$$

- $\lambda\ge0$：正則化の強さ（ハイパーパラメータ）。$\lambda=0$ で通常の最小二乗法、$\lambda\to\infty$ で $\hat{\mathbf{w}}\to\mathbf{0}$。交差検証で選ぶ。
- **バイアス（切片）は罰則に含めない**（閉じた形で書くときは、切片に対応する $\mathbf{I}$ の対角成分だけ 0 にする）。特徴量は標準化しておく（スケールが違うと罰則が不公平になる）。
- $R$ を $\tfrac12\|\mathbf{w}\|^2$ と書くか $\|\mathbf{w}\|^2$ と書くかで $\lambda$ の値が 2 倍変わる（後者の勾配は $2\lambda\mathbf{w}$）。教材ごとの違いに注意。
- Ridge の解は、$\mathbf{X}=\mathbf{U}\Sigma\mathbf{V}^{\top}$（特異値 $\sigma_i$）で書くと、各方向を $\dfrac{\sigma_i^2}{\sigma_i^2+\lambda}$ 倍に縮める。**$\sigma_i$ が小さい（情報の少ない）方向ほど強く縮む**。

### 直交な特徴量での解（縮み方の違い）

$\mathbf{X}^{\top}\mathbf{X}=\mathbf{I}$ のとき、最小二乗解を $w^{0}_j$ とすると

$$
\text{Ridge:}\ \ \hat{w}_j=\frac{w^{0}_j}{1+\lambda},\qquad
\text{Lasso:}\ \ \hat{w}_j=\mathrm{sign}(w^{0}_j)\max\bigl(\lvert w^{0}_j\rvert-\lambda,\,0\bigr)\ \ \text{（ソフトしきい値）}
$$

- Ridge は一律に割合で縮める（0 にはならない）。Lasso は一定量 $\lambda$ だけ引き、**$\lvert w^0_j\rvert\le\lambda$ なら 0 になる**＝特徴選択。

### 確率の見方（MAP 推定）

- $\mathbf{w}$ に**正規分布**の事前分布 $\mathcal{N}(0,\tau^2)$ を置いた MAP 推定が **L2**（$\lambda=\sigma^2/\tau^2$）。**ラプラス分布**の事前分布 $\propto e^{-\lvert w\rvert/b}$ なら **L1**（$\lambda=\sigma^2/b$）。ラプラス分布は 0 に尖っているので、解が 0 に張り付きやすい。

### 幾何：なぜ L1 は 0 になるのか

二乗誤差は $\mathbf{w}^{0}$ を中心とする楕円の等高線を持つ。罰則つきの解は、**楕円が罰則の等高線に最初に触れる点**（制約形では $R(\mathbf{w})\le t$ の領域の境界）。円は滑らかなので接点が軸上になることはまれだが、菱形は頂点が軸上にあり、楕円がそこで触れやすい。

<div class="ew" data-widget="linreg-l1l2"></div>

!!! tip "動かしてみる"
    - L1（Lasso）で $\lambda$ を 1.2 付近まで増やすと、解（紺の点）が菱形の頂点に乗り、$w_2$ がぴったり 0 になります。下の行の「0 でない係数の数」が 1 に減ります（さらに増やすと 2 つとも 0）。
    - L2（Ridge）に切り替えると、$\lambda$ を増やしても 2 つの係数は 0 に近づくだけで、ちょうど 0 にはなりません。
    - $\rho$ を変えると楕円の向きと細さが変わり、軌跡（灰の点線）の曲がり方と、$w_2$ が 0 になる $\lambda$ が変わります（$\rho=0$ で約 0.7、$\rho=0.9$ で約 1.3）。Elastic Net は L1 と L2 の中間の軌跡になります。

| 手法 | 罰則 | 性質 |
|:--|:--|:--|
| Ridge | L2 | 係数を縮める。多重共線性に強く、解は解析的に求まる |
| Lasso | L1 | 係数を 0 にする（スパース）。特徴選択ができるが、相関する変数のうち 1 つを選びがち |
| Elastic Net | L1＋L2 | 相関する変数をまとめて残す。スパース性と安定性の両立 |

---

## 試験の着眼点

- 正規方程式 $\mathbf{X}^{\top}\mathbf{X}\mathbf{w}=\mathbf{X}^{\top}\mathbf{y}$ を導けること。損失を $\tfrac12$ 付きで書くと勾配が $\mathbf{X}^{\top}(\mathbf{X}\mathbf{w}-\mathbf{y})$ になる。
- 誤差が正規分布 ⇒ 最小二乗法＝最尤推定。事前分布が正規分布 ⇒ L2、ラプラス分布 ⇒ L1（MAP 推定）。
- **L1＝スパース（特徴選択）、L2＝縮小（全部小さく）**。罰則の等高線が菱形か円かで説明できる。
- 多重共線性：$\mathbf{X}^{\top}\mathbf{X}$ がほぼ特異、係数が不安定、VIF で診断、Ridge で緩和。
- 過少適合は訓練誤差も大きい、過剰適合は訓練誤差が小さく汎化誤差が大きい。$\lambda$ や次数は**検証データ**で選ぶ。
- 切片は正則化しない。特徴量は標準化する。

---

## 参考

- [scikit-learn User Guide: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)（Ridge・Lasso・Elastic Net の定義と解き方）
- [Regression Shrinkage and Selection via the Lasso（Tibshirani, 1996）](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- [Regularization and variable selection via the elastic net（Zou, Hastie, 2005）](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
- [The Elements of Statistical Learning（Hastie, Tibshirani, Friedman）](https://hastie.su.domains/ElemStatLearn/)：第3章 Linear Methods for Regression
