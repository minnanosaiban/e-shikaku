# 生成モデル（VAE・GAN・拡散モデル・フロー）

<p class="kw">キーワード：生成モデル、潜在変数、VAE（ELBO・再パラメータ化トリック）、VQ-VAE、GAN（ミニマックス・モード崩壊）、WGAN、拡散モデル（DDPM・前向き過程・逆過程・ノイズ予測・DDIM・ガイダンス・潜在拡散）、スコアマッチング、正規化フロー（変数変換・カップリング層）、自己回帰モデル、FID</p>

!!! abstract "要点"
    - 生成モデルは、データの分布 $p(\mathbf{x})$ を学び、**そこから新しいデータをサンプリング**する。尤度の扱い方で、VAE・GAN・フロー・拡散モデル・自己回帰に分かれる。
    - **VAE**：尤度の下界（ELBO）を最大化。**GAN**：生成器と識別器が競う。**フロー**：可逆な変換で尤度を厳密に求める。**拡散モデル**：ノイズを足す過程の**逆向き**を、ノイズ予測のネットワークで学ぶ。
    - 拡散モデルは、前向き過程が**閉じた式** $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol{\varepsilon}$ で書けるため、学習が**ノイズの回帰**になって安定する。

---

## 生成モデルの全体像 {#overview}

| 系統 | 尤度 | 学習 | サンプリング | 特徴 |
|:--|:--|:--|:--|:--|
| 自己回帰（PixelCNN・WaveNet・GPT） | 厳密 | 最尤 | **1 要素ずつ**（遅い） | $p(\mathbf{x})=\prod_i p(x_i\mid x_{<i})$ |
| 正規化フロー | **厳密** | 最尤 | 1 回で速い | 可逆な変換。構造に制約 |
| VAE | 下界（ELBO） | 最尤の近似 | 速い | 潜在空間が滑らか。**ぼやけ**やすい |
| GAN | なし（暗黙的） | **敵対的** | 速い | 高品質。**学習が不安定**・モード崩壊 |
| 拡散モデル | 下界 | ノイズ予測の回帰 | **多数のステップ**（遅い） | 高品質・多様・安定 |

- VAE・VQ-VAE・GAN の基本は [自己教師あり学習・生成モデル](../ml/ssl-generative.md) にまとめてある。ここでは要点を確認し、拡散モデルとフローを詳しく扱う。評価指標は [性能指標](../ml/evaluation.md#generation)（**FID**・Inception Score）。

---

## VAE と GAN の要点 {#vae-gan}

### VAE

$$
\log p_\theta(\mathbf{x})=\underbrace{\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\bigl[\log p_\theta(\mathbf{x}\mid\mathbf{z})\bigr]-D_{\mathrm{KL}}\bigl(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\bigr)}_{\mathrm{ELBO}}+D_{\mathrm{KL}}\bigl(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x})\bigr)
$$

- 事後分布 $p_\theta(\mathbf{z}\mid\mathbf{x})$ は計算できないので $q_\phi$ で近似する。右端の KL は 0 以上なので、ELBO は対数尤度の**下界**。ELBO の最大化 $=$ 損失（**再構成誤差 $+$ KL**）の最小化。
- 事前分布 $p(\mathbf{z})=\mathcal{N}(\mathbf{0},\mathbf{I})$、$q_\phi=\mathcal{N}(\boldsymbol{\mu},\mathrm{diag}\,\boldsymbol{\sigma}^2)$ のとき、KL は閉じた式：$-\tfrac12\sum_j\bigl(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\bigr)$。$\boldsymbol\mu=\mathbf{0},\boldsymbol\sigma=\mathbf{1}$ で 0。
- **再パラメータ化トリック**：$\mathbf{z}=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\varepsilon,\ \boldsymbol\varepsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ とし、確率的なサンプリングを**決まった関数 + 外から来るノイズ**にして、逆伝播を通す。
- 損失にベルヌーイ（2 値）の再構成を使うなら、$L=-\sum_i\bigl[x_i\log\hat{x}_i+(1-x_i)\log(1-\hat{x}_i)\bigr]+D_{\mathrm{KL}}$。**負号は交差エントロピー全体**にかかる。
- 課題：生成画像が**ぼやける**。**事後崩壊**（潜在変数が無視される。KL の重みを小さく始めて増やす）。**VQ-VAE** は潜在変数を離散的な**コードブック**から選び（勾配は**ストレートスルー**で通す）、KL を定数にして事後崩壊を避ける。損失は $-\log p(\mathbf{x}\mid\mathbf{z}_q)+\|\mathrm{sg}[\mathbf{z}_e]-\mathbf{e}\|^2+\beta\|\mathbf{z}_e-\mathrm{sg}[\mathbf{e}]\|^2$。

### GAN

$$
\min_G\max_D V(D,G)=\mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}\bigl[\log D(\mathbf{x})\bigr]+\mathbb{E}_{\mathbf{z}\sim p_z}\bigl[\log\bigl(1-D(G(\mathbf{z}))\bigr)\bigr]
$$

- $G$ が固定のとき、最適な識別器と、そのときの目的関数は次の通り。

$$
D^*(\mathbf{x})=\frac{p_{\mathrm{data}}(\mathbf{x})}{p_{\mathrm{data}}(\mathbf{x})+p_g(\mathbf{x})},\qquad V(D^*,G)=-\log4+2\,\mathrm{JS}\bigl(p_{\mathrm{data}}\,\|\,p_g\bigr)
$$

- $G$ の最適解は $p_g=p_{\mathrm{data}}$。このとき $D^*=1/2$（本物か偽物か区別できない）、$V=-\log4\approx-1.386$。つまり**生成器は JS ダイバージェンスを最小化**している。
- 学習初期は $D$ が簡単に見破るので $\log(1-D(G(\mathbf{z})))$ の勾配が**飽和**する。実際は、$G$ を $-\log D(G(\mathbf{z}))$ の最小化（**非飽和損失**）で更新する。
- 課題：**モード崩壊**（特定の少数のデータしか作らなくなる）、**勾配消失**（識別器が強すぎる）、学習の不安定さ（非凸の 2 人ゲームで、鞍点への収束が保証されない）、損失の値から品質を判断できない。
- 改良：**WGAN**（ワッサースタイン距離。識別器＝批評家に **1-リプシッツ制約**を課す。重みのクリッピングや勾配ペナルティ）、**PGGAN**（低解像度から段階的に大きくする）、**StyleGAN**（スタイルで各層を制御）、**条件付き GAN**・**pix2pix**（画像から画像）。

---

## 拡散モデル {#diffusion}

- **発想**：データに少しずつノイズを足して**ただのノイズ**にする過程（**前向き過程**）は簡単に書ける。その**逆向き**（ノイズから少しずつノイズを取り除いてデータに戻す**逆過程**）をニューラルネットワークで学ぶ。DDPM（Denoising Diffusion Probabilistic Models）が代表。

### 前向き過程 {#forward}

$$
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=\mathcal{N}\bigl(\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t\mathbf{I}\bigr)\quad\Longleftrightarrow\quad \mathbf{x}_t=\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}+\sqrt{\beta_t}\,\boldsymbol\varepsilon
$$

- $\beta_t$：ステップ $t$ で足すノイズの大きさ（**スケジュール**）。DDPM は $\beta_1=10^{-4}$ から $\beta_T=0.02$ まで**線形**に増やし、$T=1000$。$\sqrt{1-\beta_t}$ を掛けるので、データの分散が**増えない**（$\mathrm{Var}=1$ なら 1 のまま）。
- $\alpha_t=1-\beta_t$、$\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$ とおくと、**$\mathbf{x}_0$ から $\mathbf{x}_t$ へ一気に**書ける。

$$
q(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}\bigl(\sqrt{\bar\alpha_t}\,\mathbf{x}_0,\ (1-\bar\alpha_t)\mathbf{I}\bigr)\quad\Longleftrightarrow\quad
\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon,\quad\boldsymbol\varepsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$

??? note "導出（2 ステップ）"
    $\mathbf{x}_2=\sqrt{\alpha_2}\,\mathbf{x}_1+\sqrt{1-\alpha_2}\,\boldsymbol\varepsilon_2=\sqrt{\alpha_1\alpha_2}\,\mathbf{x}_0+\sqrt{\alpha_2(1-\alpha_1)}\,\boldsymbol\varepsilon_1+\sqrt{1-\alpha_2}\,\boldsymbol\varepsilon_2$。独立なガウスノイズの和の分散は足し算なので、$\alpha_2(1-\alpha_1)+(1-\alpha_2)=1-\alpha_1\alpha_2$。よって 1 つのノイズ $\sqrt{1-\bar\alpha_2}\,\boldsymbol\varepsilon$ にまとめられる。帰納法で $t$ ステップへ。

- $t$ が大きいと $\bar\alpha_t\to0$ で、$\mathbf{x}_T\approx\mathcal{N}(\mathbf{0},\mathbf{I})$（元のデータの情報がなくなる）。$t=0$ では $\bar\alpha_0=1$ で $\mathbf{x}_0$ そのもの。**信号対雑音比** $\mathrm{SNR}(t)=\bar\alpha_t/(1-\bar\alpha_t)$ が単調に下がる。

<div class="ew" data-widget="gen-diffusion"></div>

!!! tip "動かしてみる"
    - $t$ を 0 から増やすと、三日月の形が崩れていき、$t=1000$ では**ただの正規分布の塊**になります（$\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon$ の第 1 項の係数が 0 に近づくため）。
    - 灰色の点（元のデータ $\mathbf{x}_0$）と紺の点（$\mathbf{x}_t$）は、**同じ点に同じノイズ** $\boldsymbol\varepsilon$ を混ぜたものです。$t$ を動かしても、ノイズは引き直されず、混ぜる割合（$\sqrt{\bar\alpha_t},\ \sqrt{1-\bar\alpha_t}$）だけが変わるので、点はなめらかに動きます。
    - 「線形スケジュール」では、$t=700$ ですでに $\bar\alpha_t\approx0.007$ とほぼノイズです。「コサインスケジュール」は $\bar\alpha_t$ がなだらかに下がり、**後半のステップが無駄になりにくい**ことが分かります。

### 逆過程と学習 {#reverse}

- 逆向きの 1 ステップも、$\beta_t$ が小さければガウス分布になる。ネットワークで平均を求める。

$$
p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)=\mathcal{N}\bigl(\boldsymbol\mu_\theta(\mathbf{x}_t,t),\ \sigma_t^2\mathbf{I}\bigr),\qquad
\boldsymbol\mu_\theta(\mathbf{x}_t,t)=\frac{1}{\sqrt{\alpha_t}}\Bigl(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol\varepsilon_\theta(\mathbf{x}_t,t)\Bigr)
$$

- ネットワーク $\boldsymbol\varepsilon_\theta$ は、**$\mathbf{x}_t$ に含まれるノイズを当てる**（**ノイズ予測**）。入力は $\mathbf{x}_t$ と時刻 $t$（位置エンコーディングのような埋め込みで与える）。画像では **U-Net** が使われる。
- **導出の筋**：$\mathbf{x}_0$ が既知なら、逆向きの正確な分布 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}(\tilde{\boldsymbol\mu}_t,\tilde\beta_t\mathbf{I})$ がガウスで求まる。

$$
\tilde{\boldsymbol\mu}_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\mathbf{x}_0+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t,\qquad
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$

- この $\tilde{\boldsymbol\mu}_t$ に、$\mathbf{x}_0=(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon)/\sqrt{\bar\alpha_t}$ を代入すると、上の $\boldsymbol\mu_\theta$ で $\boldsymbol\varepsilon_\theta$ を真のノイズ $\boldsymbol\varepsilon$ に置いた式と**一致する**。変分下界は、各ステップの 2 つのガウス分布の KL（平均の二乗誤差）の和になり、ほぼ**ノイズの二乗誤差**に帰着する。重みを省いた簡単な形が実際に使われる。

$$
L_{\mathrm{simple}}=\mathbb{E}_{t,\ \mathbf{x}_0,\ \boldsymbol\varepsilon}\Bigl[\bigl\|\boldsymbol\varepsilon-\boldsymbol\varepsilon_\theta\bigl(\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon,\ t\bigr)\bigr\|^2\Bigr]
$$

**学習（1 回の更新）**

1. データ $\mathbf{x}_0$ を選び、$t\in\{1,\dots,T\}$ を一様に選ぶ。
2. $\boldsymbol\varepsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ を引き、$\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon$ を作る。
3. $\|\boldsymbol\varepsilon-\boldsymbol\varepsilon_\theta(\mathbf{x}_t,t)\|^2$ の勾配で更新する。

**サンプリング（生成）**

1. $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ を引く。
2. $t=T,\dots,1$ について $\mathbf{x}_{t-1}=\boldsymbol\mu_\theta(\mathbf{x}_t,t)+\sigma_t\mathbf{z}$（$\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$、$t=1$ では $\mathbf{z}=\mathbf{0}$。$\sigma_t^2=\beta_t$ か $\tilde\beta_t$）。
3. $\mathbf{x}_0$ を出力する。

- 学習は GAN と違って**ただの回帰**なので安定。弱点は、生成に**数百〜千回のネットワーク評価**が要ること。

### 高速化・条件付け・潜在拡散 {#guidance}

- **DDIM**：逆過程を、**ノイズなしの決定的**な式に変える。まず $\mathbf{x}_0$ を推定し、それを使って 1 つ前の状態へ進む。**ステップを飛ばせる**（1000 → 50 など）。

$$
\hat{\mathbf{x}}_0=\frac{\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\bar\alpha_t}},\qquad
\mathbf{x}_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\hat{\mathbf{x}}_0+\sqrt{1-\bar\alpha_{t-1}}\,\boldsymbol\varepsilon_\theta(\mathbf{x}_t,t)
$$

- **スコアとの関係**：スコア $\nabla_{\mathbf{x}}\log p(\mathbf{x})$ は、確率の高い方向を指すベクトル場。$q(\mathbf{x}_t\mid\mathbf{x}_0)$ のスコアは $-\boldsymbol\varepsilon/\sqrt{1-\bar\alpha_t}$ なので、ノイズ予測 $\boldsymbol\varepsilon_\theta$ はスコアの予測 $\mathbf{s}_\theta=-\boldsymbol\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$ と同じこと。連続時間に拡張したものは、確率微分方程式（SDE）で書かれる。
- **条件付き生成**（テキストなど、条件 $\mathbf{c}$）：$\boldsymbol\varepsilon_\theta(\mathbf{x}_t,t,\mathbf{c})$ にする。**classifier-free guidance** は、条件ありとなし（$\varnothing$）の予測の差を強めて、条件への忠実さを上げる（$w$ は強さ）。

$$
\tilde{\boldsymbol\varepsilon}=(1+w)\,\boldsymbol\varepsilon_\theta(\mathbf{x}_t,t,\mathbf{c})-w\,\boldsymbol\varepsilon_\theta(\mathbf{x}_t,t,\varnothing)
$$

- **潜在拡散モデル**（Latent Diffusion、Stable Diffusion）：画像を**オートエンコーダで小さな潜在表現に圧縮**し、その空間で拡散する。計算量が大幅に減る。テキストの条件は、**クロスアテンション**で U-Net に入れる。

---

## 正規化フロー {#flow}

- **可逆な変換** $\mathbf{z}=f(\mathbf{x})$ で、データを単純な分布（標準正規分布）に変える。**変数変換の公式**で、尤度が**厳密に**求まる。

$$
\log p_X(\mathbf{x})=\log p_Z\bigl(f(\mathbf{x})\bigr)+\log\left|\det\frac{\partial f(\mathbf{x})}{\partial\mathbf{x}}\right|
$$

- 条件：$f$ が**可逆**、ヤコビ行列の**行列式が安く計算できる**。変換を重ねると、$\log|\det|$ は**各変換の和**になる（$f=f_K\circ\dots\circ f_1$）。
- **確認**：1 次元で $z=(x-\mu)/\sigma$ とすると、$\log p_X(x)=\log\mathcal{N}(z;0,1)-\log\sigma$。これは正規分布 $\mathcal{N}(\mu,\sigma^2)$ の対数密度そのもの（$\log|\partial z/\partial x|=-\log\sigma$）。
- **カップリング層**（NICE・RealNVP）：入力を 2 つに分け、片方はそのまま、もう片方だけを、**もう片方から作った**スケールとずらしで変換する。

$$
\mathbf{z}_a=\mathbf{x}_a,\qquad \mathbf{z}_b=\mathbf{x}_b\odot\exp\bigl(s(\mathbf{x}_a)\bigr)+t(\mathbf{x}_a),\qquad
\mathbf{x}_b=\bigl(\mathbf{z}_b-t(\mathbf{z}_a)\bigr)\odot\exp\bigl(-s(\mathbf{z}_a)\bigr),\qquad \log|\det J|=\sum_j s_j(\mathbf{x}_a)
$$

- ヤコビ行列が**三角行列**になるので、行列式は対角成分の積 $\exp(\sum s_j)$。**$s,t$ は可逆でなくてよい**ので、深いネットワークをそのまま使える。分け方を層ごとに入れ替えて、全体を混ぜる。NICE は $s=0$（足し算のみ、$\det=1$）。**Glow** は 1×1 の可逆畳み込みを加えた。
- 学習は**最尤推定**（$-\log p_X$ の最小化）、生成は $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ から**逆変換を 1 回**。弱点は、入出力の次元が同じで、**構造の制約が強い**こと。
- 連続時間にした **連続正規化フロー**（ニューラル ODE）と、それを回帰で学ぶ **Flow Matching** は、拡散モデルの確率フロー ODE と近い関係にある。

---

## 試験の着眼点

- **VAE** の損失 $=$ 再構成誤差 $+$ KL（$=-$ELBO）。KL は $-\frac12\sum(1+\log\sigma^2-\mu^2-\sigma^2)$。**再パラメータ化**は逆伝播を通すため。**GAN** は 2 人ゼロサムゲームで、最適な識別器は $D^*=p_{\mathrm{data}}/(p_{\mathrm{data}}+p_g)$、均衡で $1/2$。
- GAN の課題：**モード崩壊**、**勾配消失**、不安定。VAE の課題：**ぼやけ**。VQ-VAE は**離散**の潜在変数（ストレートスルー）。
- **拡散モデル**：前向きは**固定の**ノイズ付加（学習しない）、逆向きだけを学習。$\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon$。ネットワークは**ノイズ** $\boldsymbol\varepsilon$ を予測する（損失は二乗誤差）。
- 拡散モデルは**学習が安定・品質が高い**が、**サンプリングが遅い**（DDIM でステップを減らす）。**潜在拡散**は圧縮した空間で拡散して軽くする。
- **フロー**：可逆、**尤度が厳密**、変数変換の公式の $\log|\det J|$。カップリング層は**ヤコビアンが三角**で行列式が安い。
- 尤度が厳密：**自己回帰・フロー**。下界：**VAE・拡散**。尤度なし：**GAN**。

---

## 参考

- [Auto-Encoding Variational Bayes（VAE, Kingma, Welling, 2013）](https://arxiv.org/abs/1312.6114)
- [Neural Discrete Representation Learning（VQ-VAE, van den Oord ほか, 2017）](https://arxiv.org/abs/1711.00937)
- [Generative Adversarial Networks（Goodfellow ほか, 2014）](https://arxiv.org/abs/1406.2661)
- [Wasserstein GAN（Arjovsky ほか, 2017）](https://arxiv.org/abs/1701.07875)
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation（Karras ほか, 2017）](https://arxiv.org/abs/1710.10196)
- [A Style-Based Generator Architecture for Generative Adversarial Networks（StyleGAN, Karras ほか, 2018）](https://arxiv.org/abs/1812.04948)
- [Denoising Diffusion Probabilistic Models（Ho ほか, 2020）](https://arxiv.org/abs/2006.11239)
- [Improved Denoising Diffusion Probabilistic Models（Nichol, Dhariwal, 2021）](https://arxiv.org/abs/2102.09672)
- [Denoising Diffusion Implicit Models（DDIM, Song ほか, 2020）](https://arxiv.org/abs/2010.02502)
- [Score-Based Generative Modeling through Stochastic Differential Equations（Song ほか, 2020）](https://arxiv.org/abs/2011.13456)
- [Classifier-Free Diffusion Guidance（Ho, Salimans, 2022）](https://arxiv.org/abs/2207.12598)
- [High-Resolution Image Synthesis with Latent Diffusion Models（Rombach ほか, 2021）](https://arxiv.org/abs/2112.10752)
- [NICE: Non-linear Independent Components Estimation（Dinh ほか, 2014）](https://arxiv.org/abs/1410.8516)
- [Density estimation using Real NVP（Dinh ほか, 2016）](https://arxiv.org/abs/1605.08803)
- [Glow: Generative Flow with Invertible 1x1 Convolutions（Kingma, Dhariwal, 2018）](https://arxiv.org/abs/1807.03039)
- [Flow Matching for Generative Modeling（Lipman ほか, 2022）](https://arxiv.org/abs/2210.02747)
