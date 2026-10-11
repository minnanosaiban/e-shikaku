# 多層パーセプトロン・出力層と損失関数

<p class="kw">キーワード：全結合層、重み、バイアス、回帰、平均二乗誤差(MSE)、平均絶対誤差(MAE)、2値分類、バイナリクロスエントロピー、多クラス分類、クロスエントロピー誤差、ソフトマックス関数、one-hotベクトル、マルチラベル分類、順序回帰</p>

!!! abstract "要点"
    - **多層パーセプトロン**は、「全結合層（Affine）＋活性化関数」を重ねたネットワーク。層ごとに $\mathbf{u}=\mathbf{W}\mathbf{z}+\mathbf{b},\ \mathbf{z}=f(\mathbf{u})$ を繰り返す。
    - 出力層の活性化関数と損失は、**タスクの確率モデルに合わせて組で選ぶ**。回帰＝恒等＋MSE、2値＝シグモイド＋BCE、多クラス＝ソフトマックス＋交差エントロピー。
    - この3つの組はどれも、出力層のデルタが **$\mathbf{y}-\mathbf{t}$**（予測 − 正解）になり、飽和しても学習が止まりにくい。

---

## 多層パーセプトロン {#mlp}

入力層・隠れ層・出力層を並べ、隣り合う層の**すべてのノード同士**を重み付きの辺でつなぐ（**全結合層**）。層の数は入力層を数えず、重みを持つ層の数で数える。

$$
\mathbf{z}^{(0)}=\mathbf{x},\qquad
\mathbf{u}^{(l)}=\mathbf{W}^{(l)}\mathbf{z}^{(l-1)}+\mathbf{b}^{(l)},\qquad
\mathbf{z}^{(l)}=f\bigl(\mathbf{u}^{(l)}\bigr)\qquad(l=1,\dots,L),\qquad
\mathbf{y}=\mathbf{z}^{(L)}
$$

- $\mathbf{W}^{(l)}$：**重み**（$n_l\times n_{l-1}$）、$\mathbf{b}^{(l)}$：**バイアス**（$n_l$）。$n_l$ は第 $l$ 層のノード数。
- バッチ $\mathbf{X}$（$N\times D$）でまとめて計算するときは行ごとに並べ、$\mathbf{Z}^{(l)}=f\bigl(\mathbf{Z}^{(l-1)}\mathbf{W}^{(l)\top}+\mathbf{b}^{(l)}\bigr)$ と書く（[誤差逆伝播法](backprop.md)の Affine は $\mathbf{W}$ を $D\times M$ とおく流儀で、転置の位置が違うだけ）。
- 出力層の $f$ は隠れ層と別に選ぶ。隠れ層の $f$ は [活性化関数](activation.md) で選ぶ。

| 層 | 役割 | 例 |
|:--|:--|:--|
| 入力層 | データをそのまま受ける。計算はしない | 画像 $28\times28$ → 784 ノード |
| 隠れ層 | 非線形な特徴を作る。数・幅は設計で決める（ハイパーパラメータ） | Affine → ReLU を繰り返す |
| 出力層 | タスクに合った形（実数・確率・確率ベクトル）にする | 10 クラス → ソフトマックス |

### パラメータ数

$$
\sum_{l=1}^{L}\bigl(n_{l-1}\,n_l+n_l\bigr)=\sum_{l=1}^{L}(n_{l-1}+1)\,n_l
$$

- 例：$784\to100\to100\to100\to100\to10$（隠れ層 4 層）なら、$78500+3\times10100+1010=109810$ 個。
- 重みの数は**隣り合う層のノード数の積**で増える。画像を全結合で受けるとすぐ巨大になる（これを減らすのが [CNN](cnn.md) の重み共有）。

- 活性化関数が非線形なら、**隠れ層が 1 層でも、幅を十分に大きくすれば任意の連続関数を近似できる**（万能近似定理）。ただし必要な幅は急増しうる。層を重ねると、少ないノード数で複雑な関数を表せることが多い。
- 学習は「順伝播で損失を出す → [誤差逆伝播法](backprop.md)で勾配を出す → [最適化](optimizer.md)で更新」の繰り返し。

---

## 出力層と損失の組 {#output-loss}

| タスク | 出力層 | 損失 | 仮定する確率モデル | デルタ $\partial L/\partial\mathbf{u}$ |
|:--|:--|:--|:--|:--|
| 回帰 | 恒等関数 | MSE（または MAE） | 正規分布（ラプラス分布） | $\mathbf{y}-\mathbf{t}$（$\tfrac12$ MSE のとき） |
| 2値分類 | シグモイド（1 出力） | バイナリクロスエントロピー | ベルヌーイ分布 | $y-t$ |
| 多クラス分類 | ソフトマックス（$K$ 出力） | クロスエントロピー | カテゴリカル分布 | $\mathbf{y}-\mathbf{t}$ |
| マルチラベル分類 | シグモイド（$K$ 出力、独立） | BCE の和 | 独立なベルヌーイ分布 | $\mathbf{y}-\mathbf{t}$ |
| 順序回帰 | シグモイド（$K-1$ 出力） | 累積した BCE の和 | 閾値ごとのベルヌーイ分布 | 閾値ごとに $y_k-t_k$ |

- 損失は、**確率モデルのもとでの負の対数尤度**（最尤推定）として導ける。正規分布なら MSE、ベルヌーイ分布なら BCE、カテゴリカル分布ならクロスエントロピーになる（[パラメータ推定](../math/estimation.md)）。
- 記号は、正解 $\mathbf{t}$（または $t$）、予測 $\mathbf{y}$（または $y$）、サンプル数 $N$、クラス数 $K$ に統一する。損失の詳細集は [損失関数まとめ](../math/loss.md) を見る。

---

## 回帰の損失 {#regression-loss}

$$
\begin{aligned}
&\text{MSE} && L=\frac1N\sum_{i=1}^{N}(y_i-t_i)^2 \qquad\text{（逆伝播では }\tfrac12\text{ を付けて }\tfrac12(y-t)^2\text{ とすることが多い）} \\[2mm]
&\text{MAE} && L=\frac1N\sum_{i=1}^{N}|y_i-t_i| \\[2mm]
&\text{RMSE} && \sqrt{\text{MSE}} \\[2mm]
&\text{Huber} && L_\delta(r)=\begin{cases}\tfrac12 r^2 & (|r|\le\delta)\\[1mm] \delta\bigl(|r|-\tfrac12\delta\bigr) & (|r|>\delta)\end{cases}\qquad r=y-t
\end{aligned}
$$

| 損失 | 勾配 $\partial L/\partial y$ | 性質 |
|:--|:--|:--|
| MSE | $y-t$（$	frac12$ を付けたとき。誤差に比例） | 誤差が大きいほど強く修正。**外れ値に弱い**。予測は条件付き**平均**に近づく |
| MAE | $\mathrm{sign}(y-t)$（常に $\pm1$） | **外れ値に頑健**。予測は**中央値**に近づく。勾配の大きさが一定で、$y=t$ で微分できない |
| Huber | $\lvert r\rvert\le\delta$ なら $r$、それ以外は $\delta\,\mathrm{sign}(r)$ | 誤差が小さいうちは MSE、大きいと MAE として働く |

- 出力が誤差 $t=y+\varepsilon,\ \varepsilon\sim\mathcal{N}(0,\sigma^2)$ に従うと仮定すると、負の対数尤度は $\dfrac{(t-y)^2}{2\sigma^2}+\text{定数}$ で、MSE に一致する。ラプラス分布なら MAE。
- 恒等関数の出力層に $\tfrac12\|\mathbf{y}-\mathbf{t}\|^2$ を付けると、デルタは $\mathbf{y}-\mathbf{t}$。

---

## 分類の損失 {#classification-loss}

$$
\begin{aligned}
&\text{BCE（2値）} && L=-\frac1N\sum_{i=1}^{N}\Bigl[t_i\ln y_i+(1-t_i)\ln(1-y_i)\Bigr],\qquad y_i=\sigma(u_i) \\[2mm]
&\text{ソフトマックス} && y_k=\frac{e^{u_k}}{\sum_{j=1}^{K}e^{u_j}} \\[2mm]
&\text{交差エントロピー} && L=-\frac1N\sum_{i=1}^{N}\sum_{k=1}^{K}t_{i,k}\ln y_{i,k} \\[2mm]
&\text{マルチラベル} && L=\sum_{k=1}^{K}\mathrm{BCE}_k=-\sum_{k=1}^{K}\Bigl[t_k\ln y_k+(1-t_k)\ln(1-y_k)\Bigr],\quad y_k=\sigma(u_k)
\end{aligned}
$$

- **one-hot ベクトル**：正解クラス $c$ だけ 1、他は 0 の $\mathbf{t}$。このとき交差エントロピーは $-\ln y_c$（正解クラスの確率の負の対数）だけになる。
- BCE は $K=2$ の交差エントロピーと同じ（$t_1=t,\ t_2=1-t$）。
- **多クラス**はクラスが排他的（ちょうど 1 つ）なのでソフトマックスで $\sum_k y_k=1$ にする。**マルチラベル**は複数が同時に正解になりうるので、クラスごとに独立なシグモイドを使い、BCE を足す。
- ソフトマックスは入力の定数倍の差だけに依存する（最大値を引いてから指数をとると桁あふれしない）。温度を変えた形は [活性化関数](activation.md) を見る。実装では、ソフトマックスと対数をまとめた**ロジットから直接計算する関数**（log-sum-exp）を使うのが安定。

??? note "導出：ソフトマックス＋交差エントロピーのデルタ"
    $L=-\sum_k t_k\ln y_k$、$\dfrac{\partial y_k}{\partial u_j}=y_k(\delta_{kj}-y_j)$（[活性化関数](activation.md)のヤコビ行列）より

    $$
    \frac{\partial L}{\partial u_j}=-\sum_k\frac{t_k}{y_k}\,y_k(\delta_{kj}-y_j)=-t_j+y_j\sum_k t_k=y_j-t_j
    $$

    最後は $\sum_k t_k=1$（one-hot）を使った。シグモイド＋BCE も、$\dfrac{\partial L}{\partial y}=\dfrac{y-t}{y(1-y)}$ に $\sigma'(u)=y(1-y)$ を掛けて $y-t$ になる。

- 損失を**バッチ平均**で定義した場合、デルタは $\dfrac{1}{N}(\mathbf{y}-\mathbf{t})$ になる（$1/N$ を忘れない）。

### なぜ分類では MSE でなく交差エントロピーか

シグモイド出力に MSE を使うと、$\delta=(y-t)\,y(1-y)$ で、**$y$ が間違った側に飽和すると $y(1-y)\approx0$ で勾配が消える**。交差エントロピーなら $y(1-y)$ が約分されて $\delta=y-t$ になり、間違いが大きいほど強く修正される。

<div class="ew" data-widget="mlp-loss"></div>

!!! tip "動かしてみる"
    - $t=1$ で $u=-3$（確信を持って間違えている）にすると、$\delta_{\mathrm{CE}}\approx-0.95$ に対して $\delta_{\mathrm{MSE}}\approx-0.04$ しかありません。MSE では修正がほとんど進みません。
    - $u$ を 0 に近づけると、2 つのデルタの差は縮まります（$y=0.5$ では MSE 側が $1/4$ 倍）。
    - $t=0$ に切り替えると左右が反転します（$u=3$ で間違い）。式の行にマウスを乗せると、曲線と点が光ります。

---

## 順序回帰 {#ordinal}

ラベルに**順序**があり（例：評価 1〜5、年齢層）、間違いの大きさに意味があるタスク。分類として解くと順序を使えず、回帰として解くと間隔が等しいと仮定してしまう。

- 順位 $c\in\{1,\dots,K\}$ を、**「$c>k$ か？」という $K-1$ 個の2値問題**に分ける（$k=1,\dots,K-1$）。目標は $t_k=\mathbb{1}[c>k]$。
- 1 つのスコア $g(\mathbf{x})$ と、昇順の閾値（バイアス）$\theta_1<\cdots<\theta_{K-1}$ を使い、$y_k=P(c>k)=\sigma\bigl(g(\mathbf{x})-\theta_k\bigr)$ とする。
- 損失は各閾値の BCE の和。予測は $\hat c=1+\sum_k\mathbb{1}[y_k>0.5]$。

$$
L=-\sum_{k=1}^{K-1}\Bigl[t_k\ln y_k+(1-t_k)\ln(1-y_k)\Bigr],\qquad t_k=\mathbb{1}[c>k]
$$

- 重みを全閾値で**共有**し、バイアスだけを閾値ごとに変えると、$y_1\ge y_2\ge\cdots$ が保証される（順序の矛盾が起きない）。これが CORAL の考え方。
- 元の Qiita 記事の "Ranking Loss" の式 $-\frac1N\sum\log\frac{1}{\mathrm{rank}_i}$ は、$\frac1N\sum\log\mathrm{rank}_i$ と同じで、学習できる損失の形になっていない（予測がどこにも現れない）。上の累積 BCE に置き換えた。
- 順位学習（ranking）で使う、2 つの出力の大小を比べる損失（マージンランキング損失）とは別のもの。

---

## ラベル平滑化 {#label-smoothing}

one-hot の正解を少しなだらかにして、**過信（確率を 1 に近づけすぎる）と過学習を抑える**。

- 平滑化の強さを $\epsilon$、クラス数を $K$ とする。定義は 2 通りある。

| 流儀 | 正解クラス | 不正解クラス | 式 |
|:--|:--|:--|:--|
| (a) 不正解へ均等に分ける | $1-\epsilon$ | $\epsilon/(K-1)$ | — |
| (b) 一様分布と混ぜる（一般的） | $1-\epsilon+\epsilon/K$ | $\epsilon/K$ | $t_k^{\mathrm{LS}}=(1-\epsilon)\,t_k+\epsilon\,u_k,\ u_k=1/K$ |

$$
L=-\sum_{k=1}^{K}t_k^{\mathrm{LS}}\ln y_k=(1-\epsilon)\,\underbrace{\Bigl(-\sum_k t_k\ln y_k\Bigr)}_{\text{通常の交差エントロピー}}+\epsilon\,\underbrace{\Bigl(-\sum_k u_k\ln y_k\Bigr)}_{\text{一様分布との交差エントロピー}}\qquad\text{（流儀 (b)）}
$$

- どちらも全クラスの和が 1。元の記事は (a) の説明の直後に $u(k)$ を使う (b) の式を置いており、2 つが混ざっていた。ページ内では分けて書いた。
- 正解の目標が 1 でなくなるため、ロジットの差が無限に開かず、確率が 1 に張り付かない。モデルの確率が実際の正解率に近づく（校正がよくなる）効果もある。
- 損失の第 2 項は、予測を一様分布に引き寄せる**正則化**として働く（[正則化](regularization.md) の一種）。蒸留では、教師の確率分布を目標にするので、似た効果を得る。

---

## 試験の着眼点

- 組み合わせは必ず覚える：**回帰＝恒等＋MSE、2値＝シグモイド＋BCE、多クラス＝ソフトマックス＋交差エントロピー、マルチラベル＝シグモイド＋BCE の和**。
- 出力層のデルタは、この3つの組で $\mathbf{y}-\mathbf{t}$。バッチ平均の損失なら $1/N$ が付く。
- **多クラスは排他的（ソフトマックス）、マルチラベルは独立（シグモイド）**。
- MSE は外れ値に弱く平均を、MAE は頑健で中央値を狙う。Huber は両者の折衷。
- 交差エントロピーは one-hot なら $-\ln$（正解クラスの確率）。2クラスで BCE と一致する。
- 順序回帰は「$c>k$ か」の累積2値分類。ラベル平滑化は正解を $1-\epsilon$ にして過信を抑える。
- パラメータ数は $\sum_l(n_{l-1}+1)\,n_l$。

---

## 参考

- [Deep Learning, Chapter 6 Deep Feedforward Networks（Goodfellow, Bengio, Courville, 2016）](https://www.deeplearningbook.org/contents/mlp.html)
- [Rethinking the Inception Architecture for Computer Vision（Szegedy ほか, 2016）](https://arxiv.org/abs/1512.00567)：ラベル平滑化の提案
- [When Does Label Smoothing Help?（Müller, Kornblith, Hinton, 2019）](https://arxiv.org/abs/1906.02629)
- [Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation（Cao, Mirjalili, Raschka, 2019）](https://arxiv.org/abs/1901.07884)：CORAL
