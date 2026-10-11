# サポートベクターマシン（SVM）

<p class="kw">キーワード：サポートベクター、マージン最大化、ハードマージン・ソフトマージン、カーネル法</p>

!!! abstract "要点"
    - 2クラスを分ける超平面のうち、**最も近いデータ点までの距離（マージン）が最大**のものを選ぶ。目的は $\tfrac12\|\mathbf{w}\|^2$ の最小化で、**凸な二次計画**になる。
    - 解は**サポートベクター**（マージン上かその内側の点）だけで決まる。ラグランジュ双対にすると、データは**内積**でしか現れない。
    - 誤分類を許すのが**ソフトマージン**（$C$ で調整）。内積を**カーネル関数**に置き換えると、非線形な境界を同じ枠組みで扱える（**カーネルトリック**）。

---

## ハードマージン {#hard}

ラベルは $t_n\in\{-1,+1\}$。決定関数 $f(\mathbf{x})=\mathbf{w}^{\top}\mathbf{x}+b$、予測 $\hat{y}=\mathrm{sign}\bigl(f(\mathbf{x})\bigr)$。

- 点 $\mathbf{x}_n$ から超平面までの距離は $\dfrac{\lvert f(\mathbf{x}_n)\rvert}{\|\mathbf{w}\|}$。$\mathbf{w},b$ の定数倍は同じ超平面なので、**最も近い点で $t_nf(\mathbf{x}_n)=1$** となるよう尺度を決める。このとき、超平面から両側の最近点までが $\dfrac{1}{\|\mathbf{w}\|}$、**マージン幅**（両側あわせて）は $\dfrac{2}{\|\mathbf{w}\|}$。
- マージンの最大化 $\max \dfrac{2}{\|\mathbf{w}\|}$ は、$\tfrac12\|\mathbf{w}\|^2$ の最小化と同じ。

$$
\begin{aligned}
&\text{主問題} && \min_{\mathbf{w},\,b}\ \frac12\|\mathbf{w}\|^2\quad\text{s.t.}\quad t_n(\mathbf{w}^{\top}\mathbf{x}_n+b)\ge1\ \ (n=1,\dots,N) \\[2mm]
&\text{ラグランジュ関数} && \mathcal{L}(\mathbf{w},b,\boldsymbol{\alpha})=\frac12\|\mathbf{w}\|^2-\sum_{n=1}^{N}\alpha_n\bigl[t_n(\mathbf{w}^{\top}\mathbf{x}_n+b)-1\bigr],\quad \alpha_n\ge0 \\[2mm]
&\text{停留条件} && \frac{\partial\mathcal{L}}{\partial\mathbf{w}}=\mathbf{0}\Rightarrow\mathbf{w}=\sum_n\alpha_nt_n\mathbf{x}_n,\qquad \frac{\partial\mathcal{L}}{\partial b}=0\Rightarrow\sum_n\alpha_nt_n=0 \\[2mm]
&\text{双対問題} && \max_{\boldsymbol{\alpha}}\ \sum_{n}\alpha_n-\frac12\sum_{i,j}\alpha_i\alpha_jt_it_j\,\mathbf{x}_i^{\top}\mathbf{x}_j\quad\text{s.t.}\quad \alpha_n\ge0,\ \ \sum_n\alpha_nt_n=0 \\[2mm]
&\text{予測} && f(\mathbf{x})=\sum_{n}\alpha_nt_n\,\mathbf{x}_n^{\top}\mathbf{x}+b
\end{aligned}
$$

??? note "双対問題の導出"
    停留条件を $\mathcal{L}$ に代入する。$\mathbf{w}=\sum_n\alpha_nt_n\mathbf{x}_n$ より $\tfrac12\|\mathbf{w}\|^2=\tfrac12\sum_{i,j}\alpha_i\alpha_jt_it_j\mathbf{x}_i^{\top}\mathbf{x}_j$、$\sum_n\alpha_nt_n\mathbf{w}^{\top}\mathbf{x}_n=\|\mathbf{w}\|^2$、$b\sum_n\alpha_nt_n=0$。よって

    $$
    \mathcal{L}=\tfrac12\|\mathbf{w}\|^2-\|\mathbf{w}\|^2+\sum_n\alpha_n=\sum_n\alpha_n-\tfrac12\sum_{i,j}\alpha_i\alpha_jt_it_j\mathbf{x}_i^{\top}\mathbf{x}_j
    $$

    主問題は凸で制約も線形なので、**強双対性**が成り立ち、双対問題の最大値と主問題の最小値が一致する。

- **KKT 条件（相補性）**：$\alpha_n\bigl[t_nf(\mathbf{x}_n)-1\bigr]=0$。したがって
    - $\alpha_n=0$：マージンの外側の点（$t_nf>1$）。解に**寄与しない**。
    - $\alpha_n>0$：ちょうどマージン上の点（$t_nf=1$）＝**サポートベクター**。
- $\alpha_n$ の大半は 0（**疎な解**）。予測に使うのはサポートベクターだけ。
- $b$ は、サポートベクター $\mathbf{x}_s$ について $t_sf(\mathbf{x}_s)=1$ から $b=t_s-\sum_n\alpha_nt_n\mathbf{x}_n^{\top}\mathbf{x}_s$（複数あれば平均）。
- 双対問題は $\alpha$ についての二次計画。制約 $\sum_n\alpha_nt_n=0$ があるため、**SMO 法**は 2 つの $\alpha$ を同時に動かして、解析的に更新する。

---

## ソフトマージン {#soft}

線形分離できない、または外れ値がある場合に、制約違反を**スラック変数** $\xi_n\ge0$ で許す。

$$
\begin{aligned}
&\text{主問題} && \min_{\mathbf{w},\,b,\,\boldsymbol{\xi}}\ \frac12\|\mathbf{w}\|^2+C\sum_{n=1}^{N}\xi_n\quad\text{s.t.}\quad t_n(\mathbf{w}^{\top}\mathbf{x}_n+b)\ge1-\xi_n,\ \ \xi_n\ge0 \\[2mm]
&\text{ラグランジュ関数} && \mathcal{L}=\frac12\|\mathbf{w}\|^2+C\sum_n\xi_n-\sum_n\alpha_n\bigl[t_nf(\mathbf{x}_n)-1+\xi_n\bigr]-\sum_n\mu_n\xi_n,\quad \alpha_n,\mu_n\ge0 \\[2mm]
&\text{停留条件} && \mathbf{w}=\sum_n\alpha_nt_n\mathbf{x}_n,\quad \sum_n\alpha_nt_n=0,\quad \alpha_n+\mu_n=C \\[2mm]
&\text{双対問題} && \max_{\boldsymbol{\alpha}}\ \sum_{n}\alpha_n-\frac12\sum_{i,j}\alpha_i\alpha_jt_it_j\,\mathbf{x}_i^{\top}\mathbf{x}_j\quad\text{s.t.}\quad 0\le\alpha_n\le C,\ \ \sum_n\alpha_nt_n=0 \\[2mm]
&\text{スラック} && \xi_n=\max\bigl(0,\ 1-t_nf(\mathbf{x}_n)\bigr)\quad\text{（ヒンジ損失）}
\end{aligned}
$$

- ハードマージンとの違いは、双対問題の制約が $\alpha_n\ge0$ から **$0\le\alpha_n\le C$**（$\alpha_n+\mu_n=C,\ \mu_n\ge0$ から出る上限）に変わるだけ。
- $\xi_n$ を消すと、ヒンジ損失を使った形になる：

$$
\min_{\mathbf{w},\,b}\ \ \frac12\|\mathbf{w}\|^2+C\sum_{n=1}^{N}\max\bigl(0,\ 1-t_nf(\mathbf{x}_n)\bigr)
$$

| $\alpha_n$ | 点の位置 | $\xi_n$ |
|:--|:--|:--|
| $0$ | マージンの外側（$t_nf\ge1$） | $0$ |
| $0<\alpha_n<C$ | **マージン上**（$t_nf=1$） | $0$ |
| $C$ | マージンの内側、または誤分類（$t_nf\le1$） | $1-t_nf\ge0$（$\xi_n>1$ なら誤分類） |

- **$C$ の意味**：誤分類・マージン侵入への罰則の重み。**$C$ が大きい**ほど違反を許さず（ハードマージンに近づく、マージンが狭く過剰適合しやすい）、**$C$ が小さい**ほど違反を許してマージンを広くとる（過少適合しやすい）。$C\to\infty$ でハードマージン。
- **損失の比較**（$s=t\,f(\mathbf{x})$ の関数）：

| 損失 | 式 | 性質 |
|:--|:--|:--|
| 0-1 損失 | $\mathbb{1}[s<0]$ | 非凸・不連続で最適化しにくい |
| ヒンジ損失（SVM） | $\max(0,\,1-s)$ | $s\ge1$ で厳密に 0。→ サポートベクター以外が解に効かない |
| ロジスティック損失 | $\log(1+e^{-s})$ | どこでも正。全点が勾配に寄与し、確率が出せる（[ロジスティック回帰](logistic.md)） |

---

## カーネル法 {#kernel}

特徴写像 $\boldsymbol{\phi}(\mathbf{x})$ で高次元に写してから線形に分ければ、元の空間では非線形な境界になる。双対問題と予測式には**内積しか現れない**ので、$\boldsymbol{\phi}$ を計算せずに、**カーネル関数** $K(\mathbf{x},\mathbf{x}')=\boldsymbol{\phi}(\mathbf{x})^{\top}\boldsymbol{\phi}(\mathbf{x}')$ だけを計算すればよい（**カーネルトリック**）。

$$
\begin{aligned}
&\text{双対問題} && \max_{\boldsymbol{\alpha}}\ \sum_{n}\alpha_n-\frac12\sum_{i,j}\alpha_i\alpha_jt_it_j\,K(\mathbf{x}_i,\mathbf{x}_j)\quad\text{s.t.}\quad 0\le\alpha_n\le C,\ \ \sum_n\alpha_nt_n=0 \\[2mm]
&\text{予測} && f(\mathbf{x})=\sum_{n}\alpha_nt_n\,K(\mathbf{x}_n,\mathbf{x})+b
\end{aligned}
$$

| カーネル | 式 | 特徴 |
|:--|:--|:--|
| 線形 | $\mathbf{x}^{\top}\mathbf{x}'$ | 元の空間のまま。特徴量が多いときに有力 |
| 多項式 | $(c+\mathbf{x}^{\top}\mathbf{x}')^{q}$（$c\ge0,\ q\in\mathbb{N}$） | $q$ 次までの特徴量の積を暗黙に使う |
| **RBF（ガウス）** | $\exp\Bigl(-\dfrac{\lVert\mathbf{x}-\mathbf{x}'\rVert^2}{2\sigma^2}\Bigr)=\exp\bigl(-\gamma\lVert\mathbf{x}-\mathbf{x}'\rVert^2\bigr)$ | 無限次元の特徴空間。最もよく使う。$\gamma=\dfrac{1}{2\sigma^2}$ |
| シグモイド | $\tanh(a\,\mathbf{x}^{\top}\mathbf{x}'+c)$ | 半正定値にならない場合がある（ニューラルネット風） |

- **例**（2次元、$q=2$、$c=0$）：$(\mathbf{x}^{\top}\mathbf{x}')^2=(x_1x_1'+x_2x_2')^2=\boldsymbol{\phi}(\mathbf{x})^{\top}\boldsymbol{\phi}(\mathbf{x}')$、$\boldsymbol{\phi}(\mathbf{x})=(x_1^2,\ \sqrt2x_1x_2,\ x_2^2)$。3次元の内積を、2次元の内積の2乗で計算できる。
- カーネルの条件（**マーサーの条件**）：任意の点集合でグラム行列 $[K(\mathbf{x}_i,\mathbf{x}_j)]$ が**対称・半正定値**。
- RBF の $\gamma$：大きいと 1 点の影響範囲が狭く、境界が入り組む（過剰適合）。小さいと影響が広く、線形に近づく（過少適合）。$C$ と $\gamma$ は**グリッドサーチ＋交差検証**で選ぶ。特徴量は標準化する。
- 計算量：学習は $O(N^2)\sim O(N^3)$、予測はサポートベクターの数に比例。大規模データには不向き（線形 SVM なら確率的勾配降下法などで高速に解ける）。
- 多クラスは 1対他／1対1 の組み合わせ。回帰への拡張は **SVR**（$\varepsilon$ 以内の誤差を無視する $\varepsilon$-不感応損失）。確率が必要なときは出力にシグモイドを当てはめる（Platt scaling）。

<div class="ew" data-widget="svm-margin"></div>

!!! tip "動かしてみる"
    - 「線形・分離可能」で $\log_{10}C$ を 0 以上にすると、サポートベクターは 3 点（実線の輪）になり、マージン幅は約 2.14 で変わりません（ハードマージンと同じ）。下げていくと破線の輪（$\alpha=C$）が増え、マージンが広がります（$\log_{10}C=-2$ で約 7.2）。
    - 「線形・外れ値あり」に切り替えると、外れ値の点は常に誤分類になります。$C$ が小さいうちは外れ値を無視してマージンを広くとり、$C$ を上げるとマージンが狭まります。
    - 下の行の主問題と双対問題の値が**一致**します（強双対性）。$\sum_n\alpha_nt_n$ は常にほぼ 0 です。
    - 「円状・線形カーネル」では直線で分けられず、$\mathbf{w}\approx\mathbf{0}$ になって全部を外側クラスと予測します。「円状・RBFカーネル」に切り替えると、$\gamma=0.5$、$\log_{10}C\ge0$ で内側を囲む境界が得られます。$\gamma$ を 2 に上げると全点がサポートベクターになり、境界が点ごとに丸く囲む形になります。

---

## 試験の着眼点

- 目的は $\tfrac12\|\mathbf{w}\|^2$ の最小化（＝マージン $2/\|\mathbf{w}\|$ の最大化）。制約は $t_n(\mathbf{w}^{\top}\mathbf{x}_n+b)\ge1$。
- **サポートベクター**＝$\alpha_n>0$ の点。それ以外のデータを消しても解は変わらない。
- 双対問題：ハード＝$\alpha_n\ge0$、ソフト＝$0\le\alpha_n\le C$。どちらも $\sum_n\alpha_nt_n=0$。$\mathbf{w}=\sum_n\alpha_nt_n\mathbf{x}_n$。
- $C$ 大＝ハードに近い（過剰適合の傾向）、$C$ 小＝広いマージン（過少適合の傾向）。
- カーネルトリック：内積 $\mathbf{x}_i^{\top}\mathbf{x}_j$ を $K(\mathbf{x}_i,\mathbf{x}_j)$ に置き換える。**特徴写像を明示的に計算しない**。RBF は無限次元。
- ヒンジ損失 $\max(0,1-t f)$ は、マージンの外側の点で 0。ロジスティック損失との違いを説明できること。
- KKT 条件・ラグランジュ双対の一般論は [最適化理論](../math/optimization.md)。

---

## 参考

- [Support-vector networks（Cortes, Vapnik, 1995）](https://doi.org/10.1007/BF00994018)
- [scikit-learn User Guide: Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)（数式、カーネル、$C$・$\gamma$ の説明）
- [The Elements of Statistical Learning（Hastie, Tibshirani, Friedman）](https://hastie.su.domains/ElemStatLearn/)：第12章 Support Vector Machines and Flexible Discriminants
