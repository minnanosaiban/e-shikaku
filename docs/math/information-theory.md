# 情報理論

<p class="kw">キーワード：自己情報量、相互情報量、エントロピー、条件付きエントロピー、結合エントロピー、クロスエントロピー、KLダイバージェンス、JSダイバージェンス</p>

!!! abstract "要点"
    - **自己情報量**は「めずらしい事象ほど大きい」量 $-\log p(x)$。その期待値が**エントロピー**（分布全体の予測しにくさ）。
    - **KLダイバージェンス**は分布のズレで、**交差エントロピー − エントロピー**。エントロピーは $Q$ に依存しないので、**交差エントロピーの最小化＝KLの最小化＝最尤推定**になる。
    - **相互情報量**は「一方を知ると、もう一方の予測しにくさがどれだけ減るか」。エントロピーの差・和で書ける。

---

## 情報量とエントロピー {#entropy}

$$
\begin{aligned}
&\text{自己情報量} && I(x)=-\log p(x)=\log\frac{1}{p(x)} \\[2mm]
&\text{エントロピー} && H(X)=\mathbb{E}_{x\sim p}\bigl[I(x)\bigr]=-\sum_{x}p(x)\log p(x)
\end{aligned}
$$

- 対数の底は **2 ならビット、$e$ なら ナット**。機械学習では自然対数が多い。底を変えると全体が定数倍になるだけで、大小関係は変わらない（$1\ \text{nat}=1/\ln 2\approx1.44$ ビット）。
- $p(x)=0$ の項は $0\log0=0$ と約束する。
- 確率が1の事象の情報量は $0$、確率が小さいほど大きい。独立な事象の情報量は**足し算**になる（$-\log(p\,q)=-\log p-\log q$）。
- エントロピーは $0\le H(X)\le\log|\mathcal{X}|$。**確実**（1つの値だけ）なら $0$、**一様分布**のとき最大 $\log|\mathcal{X}|$。
- コイン投げ（表の確率 $p$）の $H=-p\log_2p-(1-p)\log_2(1-p)$ は、$p=0.5$ で最大 $1$ ビット、$p\to0,1$ で $0$。
- 連続変数には**微分エントロピー** $-\int p(x)\log p(x)\,dx$ を使う（負にもなる）。分散 $\sigma^2$ の正規分布では $\tfrac12\ln(2\pi e\sigma^2)$。分散を固定したとき、エントロピーが最大の連続分布は正規分布。

| 概念 | 式 | ひとことで |
|:--|:--|:--|
| 自己情報量 | $-\log p(x)$ | ある事象がどれだけ予想しにくいか |
| エントロピー | $-\sum_x p(x)\log p(x)$ | 分布全体がどれだけ予想しにくいか（平均情報量） |
| 結合エントロピー | $-\sum_{x,y}p(x,y)\log p(x,y)$ | 同時分布の予想しにくさ |
| 条件付きエントロピー | $-\sum_{x,y}p(x,y)\log p(x\mid y)$ | $Y$ を知ったうえでの $X$ の予想しにくさ |
| 相互情報量 | $\sum_{x,y}p(x,y)\log\dfrac{p(x,y)}{p(x)p(y)}$ | $Y$ を知ることで減る $X$ の予想しにくさ |
| 交差エントロピー | $-\sum_x P(x)\log Q(x)$ | $Q$ で作った符号で $P$ を測ったときの予想しにくさ |
| KLダイバージェンス | $\sum_x P(x)\log\dfrac{P(x)}{Q(x)}$ | $P$ と $Q$ のズレ（非対称） |
| JSダイバージェンス | $\tfrac12\bigl(D_{\mathrm{KL}}(P\Vert M)+D_{\mathrm{KL}}(Q\Vert M)\bigr)$ | KL を対称にしたもの |

---

## 結合エントロピーと条件付きエントロピー {#joint-conditional}

$$
\begin{aligned}
H(X\mid Y)&=-\sum_{x,y}p(x,y)\log p(x\mid y)=\sum_{y}p(y)\,\Bigl[-\sum_x p(x\mid y)\log p(x\mid y)\Bigr] \\[2mm]
H(X,Y)&=H(Y)+H(X\mid Y)=H(X)+H(Y\mid X)\qquad\text{（連鎖律）}
\end{aligned}
$$

- 条件付きエントロピーは、**$Y$ の値ごとのエントロピーを $p(y)$ で重み付けして平均**したもの。
- $H(X\mid Y)\le H(X)$（情報を足しても、予想しにくさは増えない）。**$Y$ から $X$ が完全に決まるなら $H(X\mid Y)=0$**。
- $X,Y$ が独立なら $H(X\mid Y)=H(X)$、$H(X,Y)=H(X)+H(Y)$。
- 注意：$H(X\mid Y)\le H(X)$ は平均の話で、特定の $Y=y$ では $H(X\mid Y=y)$ が $H(X)$ より大きくなることはある。

---

## 相互情報量 {#mutual-information}

$$
I(X;Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)\,p(y)}=D_{\mathrm{KL}}\bigl(p(x,y)\,\|\,p(x)p(y)\bigr)
$$

$$
\begin{aligned}
I(X;Y)&=H(X)-H(X\mid Y) \\
&=H(Y)-H(Y\mid X) \\
&=H(X)+H(Y)-H(X,Y)
\end{aligned}
$$

- 「同時分布」と「独立だと仮定した分布 $p(x)p(y)$」の**KLダイバージェンス**。独立なら $0$、依存が強いほど大きい。
- $I(X;Y)=I(Y;X)\ge0$（対称）。$I(X;X)=H(X)$。
- 情報量の関係は、集合の重なりの図のように読める。$H(X)$ と $H(Y)$ の**重なりが $I(X;Y)$**、重ならない部分が条件付きエントロピー、全体が $H(X,Y)$。

<div class="ew" data-widget="info-mi"></div>

!!! tip "動かしてみる"
    - $c=0$（独立）にすると $I(X;Y)=0$ で、$H(X,Y)=H(X)+H(Y)$ になります（重なりがない）。
    - $c=1$（$X=Y$）にすると $H(X\mid Y)=0$ で、$I(X;Y)=H(X)$ になります（$Y$ から $X$ が完全に分かる）。
    - 上から2本目の棒の3つの区間は、左から $H(X\mid Y)$、$I(X;Y)$、$H(Y\mid X)$ です。式の行にマウスを乗せると対応する区間が光ります。

- **データ処理不等式**：$X\to Y\to Z$ の順に情報が伝わる（$Z$ は $Y$ だけから作る）なら $I(X;Z)\le I(X;Y)$。加工しても元の情報は増えない。
- 決定木の**情報利得**は、「分割で $H$ がどれだけ減るか」で、ラベルと分割の相互情報量にあたる（[決定木・アンサンブル](../ml/tree-ensemble.md)）。

---

## 交差エントロピーとKLダイバージェンス {#kl}

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)&=\sum_xP(x)\log\frac{P(x)}{Q(x)}=-\sum_xP(x)\log\frac{Q(x)}{P(x)} \\[2mm]
&=\underbrace{-\sum_xP(x)\log Q(x)}_{H(P,\,Q)\ \text{交差エントロピー}}-\underbrace{\Bigl(-\sum_xP(x)\log P(x)\Bigr)}_{H(P)\ \text{エントロピー}}
\end{aligned}
$$

- 交差エントロピーは $H(P,Q)=H(P)+D_{\mathrm{KL}}(P\|Q)$。**$P$ 自身の予想しにくさ ＋ $Q$ を使ったことによる無駄**。
- $D_{\mathrm{KL}}(P\|Q)\ge0$、等号は $P=Q$ のとき（ギブスの不等式）。$\log$ が凹であることから、ジェンセンの不等式で $-D_{\mathrm{KL}}=\sum P\log\frac{Q}{P}\le\log\sum Q\le\log1=0$。
- **非対称**：$D_{\mathrm{KL}}(P\|Q)\ne D_{\mathrm{KL}}(Q\|P)$。距離ではない（三角不等式も満たさない）。
- $P(x)>0$ なのに $Q(x)=0$ の点があると $D_{\mathrm{KL}}(P\|Q)=\infty$。逆に $P(x)\approx0$ の領域は $P$ で重み付けされるので、そこでの $Q$ のズレはほとんど影響しない。
- 数値例（$P=(1,0,0)$、自然対数）：$Q=(0.8,0.1,0.1)$ なら $H(P,Q)=-\ln0.8=0.223$、$Q=(0.3,0.4,0.3)$ なら $-\ln0.3=1.204$。one-hot の $P$ なら $H(P)=0$ なので、これが KL にも等しい。

$$
\min_Q D_{\mathrm{KL}}(P\|Q)\iff\min_Q H(P,Q)\qquad(H(P)\text{ は }Q\text{ に依存しない定数})
$$

- 学習データの経験分布を $P$、モデルを $Q_{\boldsymbol\theta}$ とすると、$H(P,Q_{\boldsymbol\theta})$ は**負の対数尤度の平均**そのもの。つまり**最尤推定＝KL の最小化＝交差エントロピーの最小化**（[パラメータ推定](estimation.md)、[損失関数まとめ](loss.md)）。
- 向きで性質が変わる。$D_{\mathrm{KL}}(P\|Q)$ を $Q$ について最小にすると、$Q$ は $P$ の**全体を覆う**ように広がる（mass-covering）。$D_{\mathrm{KL}}(Q\|P)$ を最小にすると、$Q$ は $P$ の**1つの山に寄る**（mode-seeking）。

<div class="ew" data-widget="info-kl"></div>

!!! tip "動かしてみる"
    - スライダーは相対的な重みで、自動で合計が 1 になるよう正規化しています。
    - $Q$ を $P$ と同じ形にすると、$D_{\mathrm{KL}}(P\|Q)$ も $D_{\mathrm{KL}}(Q\|P)$ も $0$ になり、$H(P,Q)=H(P)$ です。
    - $P_1$ を大きく、$Q_1$ を小さくすると、$D_{\mathrm{KL}}(P\|Q)$ と $D_{\mathrm{KL}}(Q\|P)$ の値が食い違います（非対称）。
    - $D_{\mathrm{KL}}(P\|Q)$ の行は、**交差エントロピーからエントロピーを引いた値**になっていることを確かめられます。

---

## JSダイバージェンス {#js}

$$
M=\frac{P+Q}{2},\qquad
D_{\mathrm{JS}}(P\|Q)=\frac12\Bigl(D_{\mathrm{KL}}(P\|M)+D_{\mathrm{KL}}(Q\|M)\Bigr)
$$

- **対称**（$P$ と $Q$ を入れ替えても同じ）。**有界**で $0\le D_{\mathrm{JS}}\le\log2$（ビットなら $1$）。
- $M(x)\ge P(x)/2$ なので、$P(x)>0$ の点で $M(x)>0$。KL と違って**無限大にならない**。
- $P$ と $Q$ の台が**重ならない**とき、どれだけ離れていても $D_{\mathrm{JS}}=\log2$ の**一定値**になる。このとき勾配が 0 になり、生成モデルの学習信号が失われる。元祖の **GAN** の目的関数は JS に対応しており、この問題が学習の不安定さの一因とされる（対策として Wasserstein 距離を使う WGAN がある。[生成モデル](../app/generative.md)）。

---

## 機械学習での登場場面 {#ml-usage}

| 場面 | 情報理論での見方 |
|:--|:--|
| 分類の損失（交差エントロピー） | 正解分布 $P$（one-hot）と予測 $Q$ の $H(P,Q)$。$H(P)=0$ なので KL に一致 |
| 最尤推定 | 経験分布とモデルの KL 最小化（上記） |
| 言語モデルの**パープレキシティ** | $\mathrm{PPL}=\exp\bigl(H(P,Q)\bigr)$（交差エントロピーの指数）。分岐数のような意味 |
| VAE の正則化項 | 近似事後分布 $q_\phi(\mathbf{z}\mid\mathbf{x})$ と事前分布の KL。正規分布どうしなら閉じた式で書ける |
| 知識蒸留 | 教師の確率分布と生徒の確率分布の KL（温度つきソフトマックス。[活性化関数](../dl/activation.md)） |
| 強化学習のエントロピー正則化 | 目的に方策のエントロピー $H\bigl(\pi(\cdot\mid s)\bigr)=-\sum_a\pi(a\mid s)\log\pi(a\mid s)$ を加え、行動の多様性（探索）を促す |
| 決定木 | 情報利得（エントロピーの減少） |
| 対照学習（InfoNCE） | 相互情報量の下界を最大化する見方ができる（[損失関数まとめ](loss.md#metric-learning)） |

$$
D_{\mathrm{KL}}\bigl(\mathcal{N}(\boldsymbol\mu,\operatorname{diag}(\boldsymbol\sigma^2))\,\big\|\,\mathcal{N}(\mathbf{0},\mathbf{I})\bigr)=\frac12\sum_{j=1}^{J}\Bigl(\mu_j^2+\sigma_j^2-1-\ln\sigma_j^2\Bigr)
$$

- 検算：$\mu_j=0,\ \sigma_j=1$ で各項が $0+1-1-0=0$（同じ分布なら KL は 0）。$\mu_j$ や $\sigma_j$ がずれるほど正に大きくなる。
- 強化学習では、方策 $\pi$ のもとで $\sum_t\mathbb{E}\bigl[r_t+\alpha H(\pi(\cdot\mid s_t))\bigr]$ を最大化する形（最大エントロピー強化学習）がある（[強化学習](../app/rl.md)）。

---

## 試験の着眼点

- **交差エントロピー ＝ エントロピー ＋ KL**。$H(P)$ は定数なので、$Q$ についての最小化は KL でも交差エントロピーでも同じ。
- KL は**非対称**で、**距離ではない**。常に $\ge0$。JS は**対称**で $\log2$ 以下。
- **相互情報量**は3通りに書ける：$H(X)-H(X\mid Y)$、$H(Y)-H(Y\mid X)$、$H(X)+H(Y)-H(X,Y)$。独立なら $0$。
- **$Y$ から $X$ が完全に予測できるなら** $H(X\mid Y)=0$。
- エントロピーは**一様分布で最大**、確実な分布で $0$。2 値なら最大 1 ビット。
- 底が 2 かネイピア数かで値は変わる（単位がビットかナットか）。問題の指定に従う。
- $D_{\mathrm{KL}}(P\|Q)$ の引数の**順番**を取り違えない（重み付けは $P$ の側）。

---

## 参考

- [Deep Learning, Chapter 3 Probability and Information Theory（Goodfellow, Bengio, Courville, 2016）](https://www.deeplearningbook.org/)
- [Auto-Encoding Variational Bayes（Kingma, Welling, 2014）](https://arxiv.org/abs/1312.6114)：VAE と正規分布どうしの KL
- [Generative Adversarial Networks（Goodfellow ほか, 2014）](https://arxiv.org/abs/1406.2661)：GAN の目的関数と JS
- [Wasserstein GAN（Arjovsky, Chintala, Bottou, 2017）](https://arxiv.org/abs/1701.07875)：台が重ならないときの JS の問題
- [Soft Actor-Critic（Haarnoja ほか, 2018）](https://arxiv.org/abs/1801.01290)：エントロピー正則化つき強化学習
- Cover, Thomas, *Elements of Information Theory*（教科書、第2版、2006）
