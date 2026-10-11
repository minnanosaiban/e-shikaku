# 固有値分解・特異値分解

<p class="kw">キーワード：固有値、固有ベクトル、対角化、特異値、特異ベクトル、固有値分解、特異値分解、低ランク近似</p>

!!! abstract "要点"
    - **固有ベクトル**は、行列をかけても向きが変わらない方向。伸びる倍率が**固有値**。正方行列を $A=P\Lambda P^{-1}$ と分けるのが**固有値分解**（対角化）。
    - **特異値分解**は任意の $m\times n$ 行列を $A=U\Sigma V^{\top}$（回転・伸縮・回転）と分ける。特異値は $\sigma_i=\sqrt{\lambda_i(A^{\top}A)}$。
    - 特異値を大きい順に $k$ 個だけ残すと、**最良の低ランク近似**になる。共分散行列の固有値分解が主成分分析。

---

## 固有値と固有ベクトル {#eigen}

$$
A\mathbf{u}=\lambda\mathbf{u}\ \ (\mathbf{u}\ne\mathbf{0}),\qquad
\det(A-\lambda I)=0\ \ (\text{固有方程式})
$$

- $\mathbf{u}$ は $(A-\lambda I)\mathbf{u}=\mathbf{0}$ の自明でない解。**向きだけが決まり**、定数倍も固有ベクトルになる。
- $n$ 次正方行列には、重複や複素数を含めて $n$ 個の固有値がある。
- $\operatorname{tr}A=\sum_i\lambda_i$、$\det A=\prod_i\lambda_i$。2×2 なら固有方程式は $\lambda^2-\operatorname{tr}(A)\,\lambda+\det A=0$。

| $A$ | 固有値 | 固有ベクトル |
|:--|:--|:--|
| $\begin{pmatrix}4&1\\-2&1\end{pmatrix}$ | $2,\ 3$ | $(1,-2)^{\top},\ (1,-1)^{\top}$ |
| $\begin{pmatrix}1&4\\2&3\end{pmatrix}$ | $5,\ -1$ | $(1,1)^{\top},\ (-2,1)^{\top}$ |
| $\begin{pmatrix}3&4\\4&3\end{pmatrix}$（対称） | $7,\ -1$ | $(1,1)^{\top},\ (1,-1)^{\top}$（直交する） |

??? note "例：固有値と固有ベクトルの計算"

    $A=\begin{pmatrix}4&1\\-2&1\end{pmatrix}$ のとき。

    $$
    \begin{aligned}
    \det(A-\lambda I)&=(4-\lambda)(1-\lambda)-(1)(-2)=\lambda^2-5\lambda+6=(\lambda-2)(\lambda-3)=0\\[2mm]
    \lambda=2:&\ \begin{pmatrix}2&1\\-2&-1\end{pmatrix}\mathbf{u}=\mathbf{0}\Rightarrow 2u_1+u_2=0\Rightarrow \mathbf{u}=\begin{pmatrix}1\\-2\end{pmatrix}\\[2mm]
    \lambda=3:&\ \begin{pmatrix}1&1\\-2&-2\end{pmatrix}\mathbf{v}=\mathbf{0}\Rightarrow v_1+v_2=0\Rightarrow \mathbf{v}=\begin{pmatrix}1\\-1\end{pmatrix}
    \end{aligned}
    $$

    検算：$\operatorname{tr}A=5=2+3$、$\det A=6=2\cdot3$、$A\mathbf{u}=(2,-4)^{\top}=2\mathbf{u}$。

---

## 固有値分解（対角化） {#eigendecomp}

$$
A=P\Lambda P^{-1},\qquad P^{-1}AP=\Lambda,\qquad
P=(\mathbf{u}_1\ \cdots\ \mathbf{u}_n),\quad \Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n)
$$

- 上の例では $P=\begin{pmatrix}1&1\\-2&-1\end{pmatrix}$、$\Lambda=\begin{pmatrix}2&0\\0&3\end{pmatrix}$、$P^{-1}=\begin{pmatrix}-1&-1\\2&1\end{pmatrix}$ で、$P\Lambda P^{-1}=A$ が確かめられる。
- $P$ は**固有ベクトルを列に並べる**（行ではない）。
- **対角化できる条件**は、線形独立な固有ベクトルが $n$ 本あること。固有値がすべて異なれば必ず可能。重解で固有ベクトルが足りない行列（例：$\begin{pmatrix}1&1\\0&1\end{pmatrix}$）は対角化できない。
- うれしさ：$A^{k}=P\Lambda^{k}P^{-1}$（べき乗が対角成分のべき乗になる）。$A^{-1}=P\Lambda^{-1}P^{-1}$（固有値に 0 がなければ）。
    - 同じ行列を何度もかける RNN で、勾配が消えたり爆発したりするのは、重み行列の固有値の絶対値が 1 より小さい・大きいため（[RNN](../dl/rnn.md)）。

### 対称行列と半正定値

- 実対称行列の固有値は**すべて実数**。異なる固有値の固有ベクトルは**直交**し、**直交行列** $Q$ で $A=Q\Lambda Q^{\top}$（スペクトル分解）と書ける（$Q^{-1}=Q^{\top}$）。
- $\mathbf{x}^{\top}A\mathbf{x}\ge0$（$\forall\mathbf{x}$）となる**半正定値**行列は、固有値がすべて**非負**（正定値なら正）。共分散行列は半正定値。
- 最適化では、ヘッセ行列が正定値なら極小点（[最適化理論](optimization.md)）。
- 実装では `np.linalg.eig` が固有値と、固有ベクトル（**列**）を返す。対称行列には `eigh` を使う。データと固有ベクトルの内積をとると、新しい座標（主成分得点）になる。

### 共分散行列の固有値分解と PCA {#pca}

$$
\boldsymbol{\Sigma}=Q\Lambda Q^{\top},\qquad
\text{寄与率}_i=\frac{\lambda_i}{\sum_{j}\lambda_j},\qquad
\text{累積寄与率}_k=\frac{\sum_{j=1}^{k}\lambda_j}{\sum_{j}\lambda_j}
$$

- 固有ベクトル $\mathbf{u}_i$ は**主成分の方向**、固有値 $\lambda_i$ はその方向の**分散**。
- 固有値の大きい順に $k$ 個の固有ベクトルを選んで射影すると、次元削減になる（[次元圧縮](../ml/dimension-reduction.md)）。
- 共分散行列の固有ベクトルは、正規分布の等高線（楕円）の主軸の向きでもある（[多変量正規分布](probability.md#mvn)）。
- 記号に注意：共分散行列は太字の $\boldsymbol{\Sigma}$、特異値分解の対角行列は細字の $\Sigma$。

---

## 特異値分解 {#svd}

任意の $m\times n$ 行列 $A$ について、次を満たす $\sigma\ge0$ と $\mathbf{u},\mathbf{v}\ne\mathbf{0}$ を**特異値**・**左特異ベクトル**・**右特異ベクトル**という。

$$
A\mathbf{v}_i=\sigma_i\mathbf{u}_i,\quad A^{\top}\mathbf{u}_i=\sigma_i\mathbf{v}_i
\qquad\Longrightarrow\qquad
A=U\Sigma V^{\top},\quad U^{\top}U=I_m,\ \ V^{\top}V=I_n
$$

- $U$：$m\times m$ の直交行列（左特異ベクトル）、$V$：$n\times n$ の直交行列（右特異ベクトル）。
- $\Sigma$：$m\times n$ で、対角成分に $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$ が並び、残りは 0。$r=\operatorname{rank}A$。

| | 固有値分解 | 特異値分解 |
|:--|:--|:--|
| 対象 | 正方行列（対角化できるもの） | **任意の** $m\times n$ 行列 |
| 式 | $A=P\Lambda P^{-1}$ | $A=U\Sigma V^{\top}$ |
| 左右の行列 | 同じ基底 $P$ と $P^{-1}$ | 別々の直交行列 $U,V$ |
| 値 | 固有値 $\lambda$（負・複素もある） | 特異値 $\sigma\ge0$ |
| 幾何 | 向きが変わらない方向が $\lambda$ 倍 | 回転 $V^{\top}$ → 軸ごとに $\sigma$ 倍 → 回転 $U$ |

### 求め方

$$
A^{\top}A=V(\Sigma^{\top}\Sigma)V^{\top}
\quad\Longrightarrow\quad
\sigma_i=\sqrt{\lambda_i(A^{\top}A)},\qquad
\mathbf{v}_i=(A^{\top}A\text{ の固有ベクトル}),\qquad
\mathbf{u}_i=\frac{A\mathbf{v}_i}{\sigma_i}
$$

??? note "例：特異値分解の計算"

    $A=\begin{pmatrix}3&0\\0&2\\0&0\end{pmatrix}$ のとき。

    $$
    A^{\top}A=\begin{pmatrix}9&0\\0&4\end{pmatrix}\ \Rightarrow\ \lambda=9,\,4\ \Rightarrow\ \sigma_1=3,\ \sigma_2=2,\qquad
    \mathbf{v}_1=\begin{pmatrix}1\\0\end{pmatrix},\ \mathbf{v}_2=\begin{pmatrix}0\\1\end{pmatrix}
    $$

    $$
    \mathbf{u}_1=\tfrac13A\mathbf{v}_1=\begin{pmatrix}1\\0\\0\end{pmatrix},\quad
    \mathbf{u}_2=\tfrac12A\mathbf{v}_2=\begin{pmatrix}0\\1\\0\end{pmatrix},\qquad
    A=\begin{pmatrix}1&0\\0&1\\0&0\end{pmatrix}\begin{pmatrix}3&0\\0&2\end{pmatrix}\begin{pmatrix}1&0\\0&1\end{pmatrix}
    $$

    ここでは $r=2$ 個だけの形（コンパクト SVD）。$\mathbf{u}_3=(0,0,1)^{\top}$ を足せば $U$ が $3\times3$ の完全な SVD になる。

### 性質

- $\operatorname{rank}A=$ **0 でない特異値の個数**。
- $\|A\|_F^2=\sum_i\sigma_i^2$、$\|A\|_2=\sigma_1$（スペクトルノルム）。
- 条件数 $\kappa=\sigma_{\max}/\sigma_{\min}$ が大きいと、逆問題が数値的に不安定になる。
- 疑似逆行列 $A^{+}=V\Sigma^{+}U^{\top}$（$\Sigma^{+}$ は 0 でない $\sigma$ を逆数にして転置）。最小二乗解は $\mathbf{x}=A^{+}\mathbf{b}$。
- 半正定値な対称行列では、特異値は固有値に一致する。一般の対称行列では $\sigma_i=|\lambda_i|$。

<div class="ew" data-widget="eigen-svd"></div>

!!! tip "動かしてみる"
    - 「特異値分解」では、円が**楕円**に変わります。楕円の軸が $\sigma_1\mathbf{u}_1,\ \sigma_2\mathbf{u}_2$ で、円の上の $\mathbf{v}_1,\mathbf{v}_2$ がそこへ写ります。$|\det A|=\sigma_1\sigma_2$ も確かめられます。
    - $a=b=c=d=1$ のようにすると $\sigma_2=0$ になり、楕円が**線につぶれます**（ランク 1）。0 でない特異値は 1 個だけです。
    - 「固有値分解」に切り替えると、**向きが変わらない直線**（灰の点線）が見えます。その上の点 $\mathbf{v}$ は $\lambda\mathbf{v}$ に写ります。$A$ を対称にする（$b=c$）と、2 本の直線が直交します。
    - $a=d=0,\ b=-1,\ c=1$（90° 回転）では実の固有ベクトルがなく、$a=b=d=1,\ c=0$ では固有ベクトルが 1 本しかなく対角化できません。

---

## 低ランク近似 {#lowrank}

特異値を大きい順に $k$ 個だけ残して、行列を近似する。

$$
A_k=\sum_{i=1}^{k}\sigma_i\mathbf{u}_i\mathbf{v}_i^{\top}=U_k\Sigma_kV_k^{\top},\qquad
\|A-A_k\|_F=\sqrt{\sum_{i=k+1}^{r}\sigma_i^2},\qquad
\|A-A_k\|_2=\sigma_{k+1}
$$

- **Eckart–Young の定理**：$A_k$ は、ランクが $k$ 以下の行列の中で、フロベニウスノルムでもスペクトルノルムでも $A$ に**最も近い**。
- 保存する数は $mn$ 個から $k(m+n+1)$ 個に減る。
- 応用：画像の圧縮、PCA（中心化したデータ行列 $X_c=U\Sigma V^{\top}$ の $V$ が主成分、分散は $\sigma_i^2/(N-1)$）、潜在意味解析、推薦の行列分解。
- 大規模モデルの微調整で、重みの更新を低ランク行列 $\Delta W=BA$ で表す **LoRA** もこの考え方（[軽量化・高速化](../app/efficiency.md)）。

<div class="ew" data-widget="eigen-svd-lowrank"></div>

!!! tip "動かしてみる"
    - $k=1$ でも棒グラフの 1 本目が圧倒的に大きく、**累積寄与率が約 97 %** あります。$k=4$ ごろで元の行列とほとんど区別がつきません。
    - 誤差 $\|A-A_k\|_F$ は、**切り捨てた特異値だけ**から計算できます（右の「直接計算」と一致します）。
    - この 12×12 の行列は、$k=7$ 以降の特異値がほぼ 0 です。つまり実質的なランクは 6 で、残りの次元は冗長です。

---

## 試験の着眼点

- 固有値分解は**正方行列だけ**、特異値分解は**任意の形**の行列に使える。
- $\sigma_i=\sqrt{\lambda_i(A^{\top}A)}$。右特異ベクトルは $A^{\top}A$、左特異ベクトルは $AA^{\top}$ の固有ベクトル。
- ランク ＝ 0 でない特異値の個数。低ランク近似の誤差は、切り捨てた特異値の**二乗和の平方根**。
- 対称行列の固有ベクトルは直交。**半正定値 ⇔ 固有値が非負**。
- 共分散行列の固有値分解が PCA。寄与率は $\lambda_i/\sum_j\lambda_j$。
- $\operatorname{tr}A=\sum\lambda_i$、$\det A=\prod\lambda_i$ で、固有値の計算を検算する。

## 参考

- [Deep Learning, Chapter 2 Linear Algebra（Goodfellow, Bengio, Courville, 2016）](https://www.deeplearningbook.org/contents/linear_algebra.html)
- [Mathematics for Machine Learning, Chapter 4 Matrix Decompositions（Deisenroth, Faisal, Ong, 2020）](https://mml-book.com/)
- [LoRA: Low-Rank Adaptation of Large Language Models（Hu ほか, 2021）](https://arxiv.org/abs/2106.09685)
