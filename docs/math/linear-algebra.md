# 行列演算・ランク

<p class="kw">キーワード：行列・テンソルの積、勾配、行列のランク、テンソル、アダマール積、ノルム、距離</p>

!!! abstract "要点"
    - 行列積は**内側の次元がそろう**ときだけ定義され、結果の形は**外側の次元**になる。要素ごとの積（**アダマール積**）は同じ形どうしにだけ使える。
    - **行列のランク**は、線形独立な行（列）の最大本数。像の次元に等しく、$\det A=0$（ランク落ち）なら空間が1つ以上つぶれる。
    - 勾配は「スカラーを各変数で偏微分したベクトル」。**元の変数と同じ形**になることが、式の検算に使える。

---

## 行列・テンソルの積 {#matmul}

| 名前 | 記法 | 条件・結果の形 | 要素 |
|:--|:--|:--|:--|
| 行列積 | $C=AB$ | $(m\times n)(n\times p)=m\times p$ | $c_{ij}=\sum_{k}a_{ik}b_{kj}$ |
| アダマール積 | $C=A\odot B$ | 同じ形どうし | $c_{ij}=a_{ij}\,b_{ij}$ |
| 内積 | $\mathbf{x}^{\top}\mathbf{y}$ | 同じ長さのベクトル → スカラー | $\sum_i x_iy_i$ |
| 外積 | $\mathbf{x}\mathbf{y}^{\top}$ | $(D)\,(M)\to D\times M$ | $x_iy_j$（ランク 1 の行列） |

$$
\begin{aligned}
\begin{pmatrix}2&3\\5&6\end{pmatrix}\begin{pmatrix}1&4\\2&3\end{pmatrix}
&=\begin{pmatrix}2\cdot1+3\cdot2&2\cdot4+3\cdot3\\5\cdot1+6\cdot2&5\cdot4+6\cdot3\end{pmatrix}
=\begin{pmatrix}8&17\\17&38\end{pmatrix}\\[2mm]
\begin{pmatrix}2&3\\5&6\end{pmatrix}\odot\begin{pmatrix}1&4\\2&3\end{pmatrix}
&=\begin{pmatrix}2\cdot1&3\cdot4\\5\cdot2&6\cdot3\end{pmatrix}
=\begin{pmatrix}2&12\\10&18\end{pmatrix}
\end{aligned}
$$

- 行列積は**交換できない**（$AB\ne BA$）が、結合はできる（$(AB)C=A(BC)$）。
- $(AB)^{\top}=B^{\top}A^{\top}$、$(AB)^{-1}=B^{-1}A^{-1}$。**順序が逆になる**。
- 内積は $\mathbf{x}^{\top}\mathbf{y}=\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$。直交するベクトルの内積は 0。
- トレースは $\operatorname{tr}(AB)=\operatorname{tr}(BA)$、行列式は $\det(AB)=\det A\det B$。固有値との関係は [固有値分解・特異値分解](eigen-svd.md) で扱う。

### テンソル

- **テンソル**は行列を一般化した多次元配列。添字の数を**階数（次元数）**という。

| 階数 | 呼び方 | 例（形） |
|:--|:--|:--|
| 0 | スカラー | 損失 $L$ |
| 1 | ベクトル | 1サンプルの特徴 $(D)$ |
| 2 | 行列 | 重み $\mathbf{W}$ $(D\times M)$、バッチ $\mathbf{X}$ $(N\times D)$ |
| 3 | 3階テンソル | 系列のバッチ $(N,T,D)$ |
| 4 | 4階テンソル | 画像のバッチ $(N,C,H,W)$ |

- **テンソル積**（外積）は添字を増やす演算 $(\mathbf{a}\otimes\mathbf{b})_{ij}=a_ib_j$。**縮約**は同じ添字を掛けて和をとる演算で、行列積は縮約の一例（$c_{ij}=\sum_k a_{ik}b_{kj}$）。
- バッチ行列積は、先頭のバッチ次元をそろえて行列積を並列に行う：$(B,n,m)\times(B,m,p)\to(B,n,p)$。
- **ブロードキャスト**は、形が違う配列を足すときに小さい方を自動で複製して合わせる規則。全結合層のバイアス加算は $(N\times M)+(M)$ で、このため逆伝播ではバッチ方向に和をとる（[誤差逆伝播法](../dl/backprop.md)）。

---

## 勾配・ヤコビ行列・ヘッセ行列 {#gradient}

$$
\begin{aligned}
&\text{勾配}\ (f:\mathbb{R}^D\to\mathbb{R}) && \nabla_{\mathbf{x}}f=\frac{\partial f}{\partial \mathbf{x}}=\left(\frac{\partial f}{\partial x_1},\ \dots,\ \frac{\partial f}{\partial x_D}\right)^{\!\top} \\[2mm]
&\text{ヤコビ行列}\ (\mathbf{f}:\mathbb{R}^D\to\mathbb{R}^M) && J_{ij}=\frac{\partial f_i}{\partial x_j}\quad (M\times D) \\[2mm]
&\text{ヘッセ行列} && H_{ij}=\frac{\partial^2 f}{\partial x_i\,\partial x_j}\quad (D\times D,\ \text{対称})
\end{aligned}
$$

- 勾配は**$f$ が最も増える向き**を指す。勾配降下法は $\mathbf{x}\leftarrow\mathbf{x}-\eta\nabla f$（[最適化](../dl/optimizer.md)）。
- 連鎖律は $\dfrac{\partial L}{\partial \mathbf{x}}=J^{\top}\dfrac{\partial L}{\partial \mathbf{z}}$（$\mathbf{z}=\mathbf{f}(\mathbf{x})$、$J$ は $\mathbf{f}$ のヤコビ行列）。誤差逆伝播法はこれを層ごとに繰り返している。
- ヘッセ行列は曲率を表す。極小点では半正定値になる（[最適化理論](optimization.md)）。

よく使う微分の公式（$\mathbf{x}\in\mathbb{R}^D$、$A$ は定数行列、$\mathbf{a},\mathbf{b}$ は定数ベクトル）。

| $f$ | 勾配 | 検算のしかた |
|:--|:--|:--|
| $\mathbf{a}^{\top}\mathbf{x}$ | $\mathbf{a}$ | 1変数なら $ax\to a$ |
| $\mathbf{x}^{\top}A\mathbf{x}$ | $(A+A^{\top})\mathbf{x}$（$A$ が対称なら $2A\mathbf{x}$） | $A=I$ なら $\Vert \mathbf{x}\Vert ^2\to2\mathbf{x}$ |
| $\tfrac12\Vert A\mathbf{x}-\mathbf{b}\Vert ^2$ | $A^{\top}(A\mathbf{x}-\mathbf{b})$ | 0 とおくと正規方程式 $A^{\top}A\mathbf{x}=A^{\top}\mathbf{b}$ |
| $\operatorname{tr}(AB)$（$A$ で微分） | $B^{\top}$ | 形が $A$ と同じ |
| $\log\det A$（$A$ で微分） | $A^{-\top}$ | 1×1 なら $\log a\to1/a$ |

- 最小二乗法の正規方程式は、2行目の応用（[線形回帰](../ml/linear-regression.md)、[パラメータ推定](estimation.md)）。

---

## 行列のランク {#rank}

線形独立な行（＝列）の最大本数。行列が表す線形写像の**像の次元**に等しい。

$$
\begin{aligned}
&\operatorname{rank}(A)\le\min(m,n)\quad(\text{等号なら「フルランク」}) \\[2mm]
&\operatorname{rank}(AB)\le\min\bigl(\operatorname{rank}A,\ \operatorname{rank}B\bigr),\qquad
\operatorname{rank}(A+B)\le\operatorname{rank}A+\operatorname{rank}B \\[2mm]
&\operatorname{rank}(A^{\top})=\operatorname{rank}(A)=\operatorname{rank}(A^{\top}A)\quad(\text{行ランク}=\text{列ランク}) \\[2mm]
&\operatorname{rank}(A)+\dim\ker A=n\quad(\text{階数・退化次数の定理})
\end{aligned}
$$

- 求め方：行基本変形（掃き出し法）で階段行列にし、**非ゼロ行（ピボット）の数**を数える。特異値分解では「0 でない特異値の個数」（[固有値分解・特異値分解](eigen-svd.md)）。
- 例：$\begin{pmatrix}1&2&3\\2&4&6\\1&0&1\end{pmatrix}$ は第2行が第1行の2倍なので、ランクは 2（$\det=0$）。
- $n$ 次正方行列では、次の4つは同値：フルランク（$\operatorname{rank}=n$）／正則（逆行列がある）／$\det A\ne0$／固有値に 0 がない。
- ランク落ちは「列が互いに従属している＝情報が重複している」状態。**低ランク近似**で冗長な次元を削れる（特異値分解）。

<div class="ew" data-widget="linalg-map"></div>

!!! tip "動かしてみる"
    - 要素 $a,b,c,d$ を動かすと、単位正方形が平行四辺形に変わります。面積の倍率が $|\det A|$ です。
    - $a=1,\ b=1,\ c=1,\ d=1$ のように 2 つの列を同じ向きにそろえると、$\det A=0$ になり、像が**線につぶれ**ます（ランク 1）。つぶれた方向の情報は戻せないので、逆行列がありません。
    - $\det A$ が負になると、平行四辺形の向きが裏返ります（符号は向きの反転を表す）。

---

## ノルムと距離 {#norm}

| 名前 | 式 | 備考 |
|:--|:--|:--|
| $L_p$ ノルム | $\Vert \mathbf{x}\Vert _p=\bigl(\sum_i\lvert x_i\rvert^p\bigr)^{1/p}\ (p\ge1)$ | $p<1$ は三角不等式を満たさずノルムでない |
| $L_1$ ノルム | $\Vert \mathbf{x}\Vert _1=\sum_i\lvert x_i\rvert$ | 絶対値の和。疎な解を導く（[正則化](../dl/regularization.md)） |
| $L_2$ ノルム | $\Vert \mathbf{x}\Vert _2=\sqrt{\sum_i x_i^2}=\sqrt{\mathbf{x}^{\top}\mathbf{x}}$ | 長さ。$\Vert \mathbf{x}\Vert _2^2=\mathbf{x}^{\top}\mathbf{x}$ |
| $L_\infty$ ノルム | $\Vert \mathbf{x}\Vert _\infty=\max_i\lvert x_i\rvert$ | $p\to\infty$ の極限 |
| フロベニウスノルム | $\Vert A\Vert _F=\sqrt{\sum_{i,j}a_{ij}^2}=\sqrt{\operatorname{tr}(A^{\top}A)}$ | 行列版の $L_2$。$=\sqrt{\sum_i\sigma_i^2}$ |
| スペクトルノルム | $\Vert A\Vert _2=\sigma_{\max}(A)$ | 最大特異値 |

距離は「差のノルム」$d(\mathbf{x},\mathbf{y})=\|\mathbf{x}-\mathbf{y}\|$ で定める。

| 距離 | 式 | 特徴 |
|:--|:--|:--|
| ユークリッド距離 | $\sqrt{\sum_i(x_i-y_i)^2}$（$p=2$） | 最も一般的。外れ値に敏感 |
| マンハッタン距離 | $\sum_i\lvert x_i-y_i\rvert$（$p=1$） | 格子状の道のり |
| ミンコフスキー距離 | $\bigl(\sum_i\lvert x_i-y_i\rvert^p\bigr)^{1/p}$ | 上の2つの一般化 |
| チェビシェフ距離 | $\max_i\lvert x_i-y_i\rvert$（$p=\infty$） | 最も離れた1軸だけで決まる |
| マハラノビス距離 | $\sqrt{(\mathbf{x}-\mathbf{y})^{\top}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\mathbf{y})}$ | データの相関・ばらつきを考慮。$\boldsymbol{\Sigma}=I$ ならユークリッド距離 |

- マハラノビス距離の $\boldsymbol{\Sigma}$ はデータの**共分散行列**（和記号ではない）。分散の大きい方向のずれは小さく評価され、データ集合からの「外れ具合」を測れる（[多変量正規分布](probability.md#mvn)）。
- コサイン類似度 $\dfrac{\mathbf{x}^{\top}\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$ は向きだけを比べる類似度で、距離ではない（[評価指標](../ml/evaluation.md)）。
- 距離の選択は k-NN やクラスタリングの結果を左右する（[パターン認識・k-NN](../ml/pattern-knn.md)）。

<div class="ew" data-widget="linalg-norm"></div>

!!! tip "動かしてみる"
    - $p$ を 1 から大きくしていくと、紺の破線が**ひし形から円、さらに正方形へ**ふくらみます。同じ半径でも、$p$ が大きいほど「角に近い点」まで近いと判定されます。
    - 点を軸の上（$x_2=0$）に置くと、$L_1,L_2,L_\infty$ がすべて同じ値になります。軸から離れるほど 3 つの値に差が出ます。
    - 常に $\|\mathbf{x}\|_\infty\le\|\mathbf{x}\|_2\le\|\mathbf{x}\|_1$ が成り立ちます（図では内側から順に入れ子）。$L_1$ のひし形の**とがった角が軸上**にあるため、$L_1$ 正則化は解を軸の上（成分が 0）に導きます。

---

## 試験の着眼点

- 行列積の**形の検算**：$(m\times n)(n\times p)=m\times p$。勾配は元の変数と同じ形。
- アダマール積（要素ごと）と行列積（行・列の内積）を混同しない。記号は $\odot$。
- $\operatorname{rank}(AB)\le\min(\operatorname{rank}A,\operatorname{rank}B)$ と、正方行列で「フルランク ⇔ 正則 ⇔ $\det\ne0$」。
- $L_1$ は絶対値の和、$L_2$ は二乗和の平方根、$L_\infty$ は最大値。マハラノビス距離は共分散行列の**逆行列**を挟む。
- $(AB)^{\top}=B^{\top}A^{\top}$ のように、転置・逆行列では**積の順序が逆**になる。

## 参考

- [Deep Learning, Chapter 2 Linear Algebra（Goodfellow, Bengio, Courville, 2016）](https://www.deeplearningbook.org/contents/linear_algebra.html)
- [Mathematics for Machine Learning（Deisenroth, Faisal, Ong, 2020）](https://mml-book.com/)
- [JDLA 深層学習協会 E資格（シラバス掲載ページ）](https://www.jdla.org/certificate/engineer/)
