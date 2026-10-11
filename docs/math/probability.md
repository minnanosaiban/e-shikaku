# 確率分布・ベイズの定理

<p class="kw">キーワード：確率の基礎、確率変数、同時確率、条件付き確率、周辺確率、確率質量関数、確率密度関数、期待値、分散、共分散、ベルヌーイ分布・多項分布、ベルヌーイ試行、二項分布、カテゴリカル分布、ガウス分布・t分布、混合ガウス分布、多変量正規分布、中心極限定理、ベイズ則</p>

!!! abstract "要点"
    - 確率の計算は**加法定理・乗法定理・周辺化**の3つで組み立てる。そこから**ベイズの定理** $P(A\mid B)=P(B\mid A)P(A)/P(B)$ が出る。
    - 分布は**期待値と分散**で特徴づける。離散なら確率質量関数、連続なら確率密度関数（密度は 1 を超えてよい）。
    - 独立な量の平均は、元の分布によらず**正規分布に近づく**（中心極限定理）。だから正規分布、特に**多変量正規分布**がモデルの基本になる。

---

## 確率の基礎 {#basics}

$$
\begin{aligned}
&\text{加法定理} && P(A\cup B)=P(A)+P(B)-P(A\cap B),\qquad P(\bar A)=1-P(A) \\[2mm]
&\text{条件付き確率} && P(A\mid B)=\frac{P(A,B)}{P(B)} \\[2mm]
&\text{乗法定理・同時確率} && P(A,B)=P(A\mid B)\,P(B)=P(B\mid A)\,P(A) \\[2mm]
&\text{独立} && P(A,B)=P(A)\,P(B)\quad\Longleftrightarrow\quad P(A\mid B)=P(A) \\[2mm]
&\text{周辺確率（離散）} && P(A)=\sum_{B}P(A,B)=\sum_{B}P(A\mid B)\,P(B) \\[2mm]
&\text{周辺確率（連続）} && p(a)=\int p(a,b)\,db=\int p(a\mid b)\,p(b)\,db
\end{aligned}
$$

- 同時確率を $P(A)P(B)$ と書けるのは**独立のときだけ**。一般の乗法定理は条件付き確率を使う。
- **確率変数**は、とる値が確率的に決まる変数。
    - 離散：**確率質量関数**（PMF）$p(x)=P(X=x)$。$\sum_xp(x)=1$。
    - 連続：**確率密度関数**（PDF）$f(x)\ge0$、$P(a\le X\le b)=\int_a^bf(x)\,dx$。$f(x)$ 自体は確率ではなく、1 を超えることもある。
    - **累積分布関数**（CDF）$F(x)=P(X\le x)$。

---

## 期待値・分散・共分散 {#moments}

| 量 | 式 | 性質 |
|:--|:--|:--|
| 期待値 | $\mathbb{E}[X]=\sum_x x\,p(x)$（連続は $\int x f(x)\,dx$） | 線形：$\mathbb{E}[aX+bY]=a\mathbb{E}[X]+b\mathbb{E}[Y]$（独立でなくてもよい） |
| 分散 | $\mathbb{V}[X]=\mathbb{E}\bigl[(X-\mathbb{E}[X])^2\bigr]=\mathbb{E}[X^2]-\mathbb{E}[X]^2$ | $\mathbb{V}[aX+b]=a^2\mathbb{V}[X]$ |
| 共分散 | $\operatorname{Cov}(X,Y)=\mathbb{E}[XY]-\mathbb{E}[X]\,\mathbb{E}[Y]$ | $\mathbb{V}[X+Y]=\mathbb{V}[X]+\mathbb{V}[Y]+2\operatorname{Cov}(X,Y)$ |
| 相関係数 | $\rho=\dfrac{\operatorname{Cov}(X,Y)}{\sqrt{\mathbb{V}[X]\,\mathbb{V}[Y]}}\in[-1,1]$ | 独立 ⇒ 無相関。**逆は成り立たない** |
| 条件付き期待値 | $\mathbb{E}[X\mid Y=y]=\sum_x x\,P(X=x\mid Y=y)$ | 全期待値の法則：$\mathbb{E}[X]=\sum_y\mathbb{E}[X\mid Y=y]P(Y=y)$ |
| 条件付き分散 | $\mathbb{V}[X\mid Y]=\mathbb{E}[X^2\mid Y]-\mathbb{E}[X\mid Y]^2$ | 全分散の法則：$\mathbb{V}[X]=\mathbb{E}\bigl[\mathbb{V}[X\mid Y]\bigr]+\mathbb{V}\bigl[\mathbb{E}[X\mid Y]\bigr]$ |

### データから計算する場合

$$
\begin{aligned}
&\text{母集団}\quad && \mu=\frac1N\sum_{i=1}^{N}x_i,\qquad \sigma^2=\frac1N\sum_{i=1}^{N}(x_i-\mu)^2 \\[2mm]
&\text{標本}\quad && \bar{x}=\frac1n\sum_{i=1}^{n}x_i,\qquad s^2=\frac1n\sum_{i=1}^{n}(x_i-\bar{x})^2,\qquad \text{不偏分散}=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2 \\[2mm]
&\text{標本共分散}\quad && \sigma_{xy}=\frac1n\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})=\frac1n\sum_{i=1}^{n}x_iy_i-\bar{x}\,\bar{y} \\[2mm]
&\text{相関係数}\quad && r=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i(x_i-\bar{x})^2}\sqrt{\sum_i(y_i-\bar{y})^2}}=\frac{\sigma_{xy}}{\sigma_x\sigma_y} \\[2mm]
&\text{決定係数}\quad && R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}
\end{aligned}
$$

- $\frac1n$ で割る標本分散は、真の分散を**少し小さく見積もる**（$\mathbb{E}[s^2]=\frac{n-1}{n}\sigma^2$）。偏りをなくすのが $\frac{1}{n-1}$ の不偏分散。
- **共分散行列**は、$D$ 次元ベクトル $\mathbf{x}$ の分散と共分散を並べた $D\times D$ 行列。**対称で半正定値**。

$$
\boldsymbol{\Sigma}=\mathbb{E}\bigl[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^{\top}\bigr]
=\begin{pmatrix}\sigma_1^2&\sigma_{12}&\cdots&\sigma_{1D}\\\sigma_{12}&\sigma_2^2&\cdots&\sigma_{2D}\\\vdots&\vdots&\ddots&\vdots\\\sigma_{1D}&\sigma_{2D}&\cdots&\sigma_D^2\end{pmatrix},
\qquad
S=\frac{1}{n-1}\sum_{i=1}^{n}(\mathbf{x}_i-\bar{\mathbf{x}})(\mathbf{x}_i-\bar{\mathbf{x}})^{\top}
$$

- 右は標本共分散行列 $S$（データ行列 $X_c$ の各列から平均を引いて $X_c^{\top}X_c/(n-1)$）。固有値分解すると PCA になる（[固有値分解・特異値分解](eigen-svd.md#pca)）。

---

## 主な確率分布 {#distributions}

| 分布 | 確率関数 | 期待値 | 分散 |
|:--|:--|:--|:--|
| ベルヌーイ $\mathrm{Ber}(x\mid p)$ | $p^{x}(1-p)^{1-x},\ x\in\{0,1\}$ | $p$ | $p(1-p)$ |
| 二項 $\mathrm{Bin}(x\mid n,p)$ | $\binom{n}{x}p^{x}(1-p)^{n-x},\ x=0,\dots,n$ | $np$ | $np(1-p)$ |
| カテゴリカル $\mathrm{Cat}(\mathbf{x}\mid\mathbf{p})$ | $\prod_{k=1}^{K}p_k^{x_k}$（$\mathbf{x}$ は one-hot） | $p_k$ | $p_k(1-p_k)$ |
| 多項 $\mathrm{Mult}(\mathbf{x}\mid\mathbf{p},n)$ | $\dfrac{n!}{x_1!\cdots x_K!}\prod_{k=1}^{K}p_k^{x_k},\ \sum_kx_k=n$ | $np_k$ | $np_k(1-p_k)$ |
| ポアソン $\mathrm{Poisson}(x\mid\lambda)$ | $\dfrac{\lambda^{x}e^{-\lambda}}{x!},\ x=0,1,2,\dots$ | $\lambda$ | $\lambda$ |
| 一様 $\mathrm{Unif}(x\mid a,b)$ | $\dfrac{1}{b-a},\ a\le x\le b$ | $\dfrac{a+b}{2}$ | $\dfrac{(b-a)^2}{12}$ |
| 指数 $\mathrm{Exp}(x\mid\lambda)$ | $\lambda e^{-\lambda x},\ x\ge0$ | $\dfrac{1}{\lambda}$ | $\dfrac{1}{\lambda^{2}}$ |
| 正規（ガウス）$\mathcal{N}(x\mid\mu,\sigma^2)$ | $\dfrac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Bigl(-\dfrac{(x-\mu)^2}{2\sigma^2}\Bigr)$ | $\mu$ | $\sigma^2$ |
| t $\mathrm{t}(x\mid\nu,\mu,\sigma^2)$ | $\dfrac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})\sqrt{\pi\nu\sigma^2}}\Bigl(1+\dfrac{(x-\mu)^2}{\nu\sigma^2}\Bigr)^{-\frac{\nu+1}{2}}$ | $\mu\ (\nu>1)$ | $\dfrac{\nu\sigma^2}{\nu-2}\ (\nu>2)$ |
| 混合ガウス | $\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x\mid\mu_k,\sigma_k^2),\ \sum_k\pi_k=1$ | $\sum_k\pi_k\mu_k$ | $\sum_k\pi_k(\sigma_k^2+\mu_k^2)-\bigl(\sum_k\pi_k\mu_k\bigr)^2$ |

- **ベルヌーイ試行**は、成功か失敗かの2通りだけの試行。1回の結果が $\mathrm{Ber}(p)$、$n$ 回の成功回数が $\mathrm{Bin}(n,p)$。$\mathrm{Bin}(1,p)=\mathrm{Ber}(p)$。
- **カテゴリカル**は $K$ クラスから1つを選ぶ（ベルヌーイの多クラス版）。分類ネットワークのソフトマックス出力がこの確率 $\mathbf{p}$ にあたる（[出力層と損失関数](../dl/mlp-loss.md)）。$n$ 回の試行で各クラスが出る回数が**多項分布**（二項分布の多クラス版）で、$K=2$ なら二項分布。多項分布の成分間の共分散は $-np_ip_j\ (i\ne j)$。
- **ポアソン**は一定の割合で起きる事象の回数。$n$ が大きく $p$ が小さい二項分布は $\lambda=np$ のポアソン分布で近似できる。待ち時間は**指数分布**に従う（[ポアソン過程](stochastic-process.md#poisson-process)）。指数分布は**無記憶性**をもつ：$P(X>s+t\mid X>s)=P(X>t)$。
- **t 分布**は正規分布より**裾が重い**。$\nu\to\infty$ で正規分布に一致し、$\nu=1$ はコーシー分布（期待値も分散も存在しない）。小標本の推定や外れ値に強いモデルに使う。
- **混合ガウス分布**は、複数の正規分布を重み $\pi_k$ で混ぜた多峰の分布。「まずクラス $k$ をカテゴリカルで選び、その正規分布から値を出す」と考える（[クラスタリング](../ml/clustering.md)）。
- 二項分布の例：コインを 5 回投げて表が 2 回 $=\binom52\bigl(\tfrac12\bigr)^5=\tfrac{10}{32}=\tfrac{5}{16}$。赤玉 2 個・白玉 1 個の袋から、取り出して戻すことを 5 回行って赤玉が 4 回 $=\binom54\bigl(\tfrac23\bigr)^4\tfrac13=\tfrac{80}{243}$。
- 多項分布の例：サイコロを 4 回投げて 1 が 2 回・4 が 1 回・5 が 1 回 $=\dfrac{4!}{2!\,1!\,1!}\bigl(\tfrac16\bigr)^4=\tfrac{12}{1296}=\tfrac{1}{108}$。赤 4 個・青 3 個・白 1 個の袋から、取り出して戻すことを 4 回行って赤 1 回・青 2 回・白 1 回 $=\dfrac{4!}{1!\,2!\,1!}\cdot\tfrac48\bigl(\tfrac38\bigr)^2\tfrac18=\tfrac{27}{256}$。

<div class="ew" data-widget="prob-dist"></div>

!!! tip "動かしてみる"
    - 「正規」で $\sigma$ を大きくすると、山は低く広がります（面積は常に 1）。$\mu$ は左右に動かすだけです。
    - 「t」で $\nu$ を小さくすると、灰の正規分布より**裾が高く**なります。$\nu\le2$ では分散が、$\nu=1$ では期待値も存在せず、対応する線が消えます。$\nu=30$ ではほぼ正規分布です。
    - 「二項」で $p=0.5$ のまま $n$ を増やすと、棒の形が**釣鐘型**に近づきます（中心極限定理）。「ポアソン」では $\lambda$ を大きくしても、期待値と分散が一致したまま正規分布に近づきます。

---

## 中心極限定理 {#clt}

$X_1,\dots,X_n$ が独立で同じ分布に従い（i.i.d.）、期待値 $\mu$・分散 $\sigma^2$ が有限なら、標本平均 $\bar{X}_n=\frac1n\sum_iX_i$ を標準化した量は標準正規分布に近づく。

$$
Z_n=\frac{\bar{X}_n-\mu}{\sigma/\sqrt{n}},\qquad
\lim_{n\to\infty}P(Z_n\le z)=\Phi(z)=\int_{-\infty}^{z}\frac{1}{\sqrt{2\pi}}e^{-x^2/2}\,dx
$$

- 元の分布がどんな形でも、$n$ が大きければ $\bar{X}_n\approx\mathcal{N}\bigl(\mu,\ \sigma^2/n\bigr)$。**標準誤差** $\sigma/\sqrt{n}$ は、$n$ を 4 倍にすると半分になる。
- 大数の法則（$\bar{X}_n\to\mu$）は「どこに収束するか」、中心極限定理は「その周りのばらつきの形」を述べる。
- 分散が無限の分布（コーシー分布など）には使えない。
- ミニバッチ勾配の平均のばらつきがバッチサイズ $B$ に対して $1/\sqrt{B}$ で減るのも、この定理による（[最適化](../dl/optimizer.md)）。

---

## 多変量正規分布 {#mvn}

$D$ 次元ベクトル $\mathbf{x}$ について、平均 $\boldsymbol{\mu}$ と共分散行列 $\boldsymbol{\Sigma}$ で決まる分布。

$$
\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu},\boldsymbol{\Sigma})
=\frac{1}{(2\pi)^{D/2}\,\lvert\boldsymbol{\Sigma}\rvert^{1/2}}
\exp\!\Bigl(-\frac12(\mathbf{x}-\boldsymbol{\mu})^{\top}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\Bigr)
$$

- 指数部の $d^2=(\mathbf{x}-\boldsymbol{\mu})^{\top}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ は**マハラノビス距離**の二乗。等高線は $d$ が一定の**楕円（体）**になる。
- 楕円の**主軸の向きは $\boldsymbol{\Sigma}$ の固有ベクトル**、主軸方向の半径は $d\sqrt{\lambda_i}$（固有値 $\lambda_i$ は主軸方向の分散）。$D=1$ にすると通常の正規分布に戻る。
- $\boldsymbol{\Sigma}=\sigma^2I$ なら円（等方）、対角行列なら軸に沿った楕円、非対角成分があると傾く。
- **無相関なら独立**になるのは、正規分布（同時に正規分布に従う場合）の特別な性質。
- 次の性質があるため、扱いやすい。
    - 線形変換：$A\mathbf{x}+\mathbf{b}\sim\mathcal{N}(A\boldsymbol{\mu}+\mathbf{b},\ A\boldsymbol{\Sigma}A^{\top})$。
    - 周辺分布は正規分布：$\mathbf{x}_a\sim\mathcal{N}(\boldsymbol{\mu}_a,\boldsymbol{\Sigma}_{aa})$。
    - 条件付き分布も正規分布：$\boldsymbol{\mu}_{a\mid b}=\boldsymbol{\mu}_a+\boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}(\mathbf{x}_b-\boldsymbol{\mu}_b)$、$\boldsymbol{\Sigma}_{a\mid b}=\boldsymbol{\Sigma}_{aa}-\boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}$（ガウス過程回帰の予測式のもと。[ベイズ・ガウス過程](../ml/bayes-gp-crf.md)）。
- 乱数は、$\boldsymbol{\Sigma}=LL^{\top}$（コレスキー分解）として $\mathbf{x}=\boldsymbol{\mu}+L\mathbf{z}$（$\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$）で作れる。VAE の再パラメータ化トリックも同じ形（[生成モデル](../app/generative.md)）。

<div class="ew" data-widget="prob-mvn"></div>

!!! tip "動かしてみる"
    - $\rho=0$ にすると楕円が軸に沿います。$\sigma_x=\sigma_y$ にすれば円になります。$\rho$ の符号で、右上がり・右下がりに傾きます。
    - $|\rho|$ を 1 に近づけると、楕円が細長くなり、$\lambda_2$ が 0 に近づきます。$\det\boldsymbol{\Sigma}\to0$ なので山の高さ $1/(2\pi\sqrt{\det\boldsymbol{\Sigma}})$ が上がります。
    - どう動かしても $\lambda_1+\lambda_2=\sigma_x^2+\sigma_y^2$（トレース）、$\lambda_1\lambda_2=\det\boldsymbol{\Sigma}$ が成り立ちます。

---

## ベイズの定理 {#bayes}

$$
\overset{\text{事後確率}}{P(A\mid B)}
=\frac{\overset{\text{尤度}}{P(B\mid A)}\ \overset{\text{事前確率}}{P(A)}}{\underset{\text{周辺尤度（エビデンス・正規化定数）}}{P(B)}},
\qquad
P(B)=\sum_iP(B\mid A_i)P(A_i)
$$

- 導出は乗法定理 $P(A,B)=P(A\mid B)P(B)=P(B\mid A)P(A)$ を $P(B)$ で割るだけ。
- 事象 $A$ とその余事象 $\bar A$ の2択なら、$P(A\mid B)=\dfrac{P(B\mid A)P(A)}{P(B\mid A)P(A)+P(B\mid\bar A)P(\bar A)}$。
- **事後 ∝ 尤度 × 事前**。分母 $P(B)$ は $A$ によらない正規化定数なので、どの $A$ が最大かを比べるときは省ける。
- **ベイズ更新**：観測 $B$ のあとの事後確率を、次の観測 $C$ の事前確率として使う。$C$ が $A$ を与えると $B$ と独立なら次のように書ける。

$$
P(A\mid B,C)=\frac{P(C\mid A)\,P(A\mid B)}{P(C\mid B)},\qquad
P(C\mid B)=\int P(C\mid A)\,P(A\mid B)\,dA
$$

- 事前確率が低いと、検査の精度が高くても、陽性になった人が本当に罹患している確率（事後確率）は低い。

<div class="ew" data-widget="prob-bayes"></div>

!!! tip "動かしてみる"
    - $P(A)$（事前確率）を 0.05 から 0.001 に下げると、感度 $P(B\mid A)$ が 0.9 のままでも、陽性の点のほとんどが灰色（健康・陽性）になり、$P(A\mid B)$ が急に下がります。
    - 偽陽性率 $P(B\mid\bar A)$ を下げると、灰色の点が減って、事後確率が上がります。
    - $P(A\mid\bar B)$（陰性だったのに罹患している確率）は、感度が高いほど小さくなります。

### 例題

| 設定 | 事後確率 |
|:--|:--|
| 迷惑メールは全体の 10 %。「広告」を含む割合は、迷惑メールで 30 %、一般メールで 5 %。「広告」を含むメールが迷惑メールである確率 | $\dfrac{0.3\cdot0.1}{0.3\cdot0.1+0.05\cdot0.9}=\dfrac{0.03}{0.075}=0.4$ |
| 1 万人に 1 人の病気。検査の誤判定は 1 %。陽性のとき本当に罹患している確率 | $\dfrac{0.99\cdot0.0001}{0.99\cdot0.0001+0.01\cdot0.9999}\approx0.0098$ |
| 工場 A・B・C の生産比は 50・30・20 %、不良率は 1・2・5 %。不良品が工場 C 製である確率 | $\dfrac{0.05\cdot0.2}{0.01\cdot0.5+0.02\cdot0.3+0.05\cdot0.2}=\dfrac{0.010}{0.021}\approx0.476$ |
| 工場 A〜D の生産比は 30・40・20・10 %、不良率は 5・6・4・10 %。不良品の出所として最も可能性が高い工場 | 分母は $0.015+0.024+0.008+0.010=0.057$。A:0.263、**B:0.421**、C:0.140、D:0.175 → 工場 B |
| 罹患率 0.001。罹患者の 95 % が陽性、非罹患者の 5 % が陽性。陽性のとき罹患している確率 | $\dfrac{0.95\cdot0.001}{0.95\cdot0.001+0.05\cdot0.999}\approx0.019$ |
| 罹患率 0.010。罹患者の 90 % が陽性、非罹患者の 10 % が陽性。陽性のとき罹患している確率 | $\dfrac{0.9\cdot0.010}{0.9\cdot0.010+0.1\cdot0.990}=\dfrac{0.009}{0.108}\approx0.083$ |

- どの例題も「**分母は、すべての場合の確率の和**（全確率の定理）」の形。表に出ている確率は条件の向き（$P(B\mid A)$ か $P(A\mid B)$ か）に注意する。
- ベイズの定理をパラメータ推定に使うと、MAP推定・ベイズ推定になる（[パラメータ推定](estimation.md)）。

---

## 試験の着眼点

- 同時確率 $P(A,B)=P(A)P(B)$ は**独立のとき**だけ。一般には $P(A\mid B)P(B)$。
- $\mathbb{V}[X]=\mathbb{E}[X^2]-\mathbb{E}[X]^2$、$\mathbb{V}[aX+b]=a^2\mathbb{V}[X]$。独立 ⇒ 無相関で、逆は不成立。
- 標本分散（$\frac1n$）は小さめに偏る。不偏分散は $\frac1{n-1}$。
- 分布の期待値・分散：二項 $np,\ np(1-p)$、ポアソン $\lambda,\ \lambda$、指数 $1/\lambda,\ 1/\lambda^2$、一様 $\frac{a+b}2,\ \frac{(b-a)^2}{12}$。
- t 分布は正規分布より裾が重く、$\nu\to\infty$ で正規分布に一致。混合ガウス分布は多峰。
- 中心極限定理：標本平均の分布は $\mathcal{N}(\mu,\sigma^2/n)$ に近づく（元の分布は何でもよい）。
- 多変量正規分布の等高線の主軸は、共分散行列の**固有ベクトル**の向き。
- ベイズの定理の分母は正規化定数。**事前確率が小さい**と、陽性でも事後確率は小さい。

## 参考

- [Deep Learning, Chapter 3 Probability and Information Theory（Goodfellow, Bengio, Courville, 2016）](https://www.deeplearningbook.org/contents/prob.html)
- [Pattern Recognition and Machine Learning, Chapter 1–2（Bishop, 2006）](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
- [Mathematics for Machine Learning, Chapter 6 Probability and Distributions（Deisenroth, Faisal, Ong, 2020）](https://mml-book.com/)
